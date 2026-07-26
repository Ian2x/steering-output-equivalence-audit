"""A1 anchor driver — function vectors (Todd et al. 2310.15213), must classify
Genuine (plan §6, amended §11).

Stage 1 (fv_extract.py): 10-shot ICL prompts; mean task-conditioned head
activations (dense-input head slices, last token, >=100 clean prompts); CIE head
ranking on shuffled-label prompts (correct-answer first-token logprob recovery,
optional layer band); FV = sum of top-10 CIE heads' means through dense.
Insertion: resid_post at layer 11 (~L/3), added to the final prompt token AND
each generated position (E_all), Todd's zero-shot protocol.

Stage 2 (battery.py machinery): zero-shot eval, task-accuracy behavior metric
(8-token greedy; gold word appears, word-boundary, case-insensitive). Conditions:
baseline, E_all, E_first (KV-baked one-shot, verified), calibrated logit-bias
control (Amendment 1: mean teacher-forced per-step KL budget), floor (random
vector at matched norm, 3 seeds, max). Degeneracy gate, bootstrap 10k, rho, kappa.

Verdict (amended §3): Genuine requires rho_hi <= 0.3 AND E_all >= 3x floor
(cell valid). Sanity: FV raw gain vs baseline must be >= 10 pts (reproduce the
method before auditing it); if not, STOP.

Usage:
  python run_a1.py --model pythia-160m --task antonym --smoke --n-eval 20 \
      --n-mean 20 --n-cie 8 --edit-layer 5           # smoke
  python run_a1.py --model EleutherAI/pythia-2.8b --task antonym            # full
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery as B          # noqa: E402
import fv_extract as FV      # noqa: E402


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "external", "function_vectors", "dataset_files", "abstractive")


# ---------------------------------------------------------------------------
# Behavior metric: task accuracy (gold word appears in generation)
# ---------------------------------------------------------------------------

def answer_hit(gen_text: str, gold: str) -> bool:
    """True if ``gold`` appears in ``gen_text`` as a whole word, case-insensitive."""
    if not gold:
        return False
    return re.search(r"\b" + re.escape(gold.lower()) + r"\b",
                     gen_text.lower()) is not None


# ---------------------------------------------------------------------------
# FV method wrapper (reuses battery.A0Method's actlib patch plumbing)
# ---------------------------------------------------------------------------

class FVMethod:
    """FV additive insertion at resid_post[edit_layer], via actlib generate_with_patch.

    add_vector = FV (fixed [hidden]); positions='all_generated' (E_all) or
    'last_prompt' (E_first). Wraps a battery.A0Method with u=FV/||FV||, scale=||FV||,
    alpha=1 so add_vector(alpha=1) == FV exactly — reusing all the battery
    plumbing (generate, generate_with_vector, KV-baked semantics, TF-KL, flips).
    """
    def __init__(self, model, tokenizer, edit_layer, fv_vec, device="cpu",
                 max_new_tokens=8):
        self.fv = fv_vec.to(device)
        self.norm = float(self.fv.norm().item())
        u = self.fv / (self.fv.norm() + 1e-12)
        self.meth = B.A0Method(model, tokenizer, edit_layer, u, self.norm,
                               device=device, max_new_tokens=max_new_tokens)
        self.model, self.tokenizer, self.layer = model, tokenizer, edit_layer
        self.device, self.max_new_tokens = device, max_new_tokens

    def generate(self, prompt, mode):  # mode: 'base','all','first'
        return self.meth.generate(prompt, 1.0, mode)


def generate_condition(fn, prompts, golds):
    texts = [fn(p) for p in prompts]
    hits = [int(answer_hit(t, g)) for t, g in zip(texts, golds)]
    return texts, hits


# ---------------------------------------------------------------------------
# Position-1 logit delta for the FV insertion (for control token-set discovery)
# ---------------------------------------------------------------------------

def fv_position1_logit_delta(fvm: FVMethod, prompts):
    """Mean position-1 logit delta (FV at last prompt token) over prompts. Reuses
    battery.position1_logit_delta via the wrapped A0Method (alpha=1)."""
    return B.position1_logit_delta(fvm.meth, fvm.tokenizer, prompts, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bf16"],
                    help="model param dtype (bf16 strongly recommended on MPS "
                         "for pythia-2.8b: ~8s load, 0.1s/fwd vs float32 thrash)")
    ap.add_argument("--task", default="antonym")
    ap.add_argument("--n-eval", type=int, default=200,
                    help="held-out eval items (split n-calib / rest)")
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--n-mean", type=int, default=100,
                    help="clean ICL prompts for mean head activations")
    ap.add_argument("--n-cie", type=int, default=32,
                    help="shuffled-label prompts for CIE ranking")
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--edit-layer", type=int, default=None,
                    help="FV insertion layer (default = round(n_layers/3))")
    ap.add_argument("--cie-lo", type=int, default=3,
                    help="CIE search band low layer (inclusive)")
    ap.add_argument("--cie-hi", type=int, default=24,
                    help="CIE search band high layer (inclusive)")
    ap.add_argument("--no-cie-band", action="store_true",
                    help="search all layers for CIE (slow)")
    ap.add_argument("--n-top-heads", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stage1-cache", default=None,
                    help="path to save/load Stage-1 outputs (FV + heads)")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = args.outdir or os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-06-a1-anchor")
    os.makedirs(outdir, exist_ok=True)
    tag = f"{args.task}_" + ("smoke" if args.smoke else "full")
    t0 = time.time()
    log(f"model={args.model} task={args.task} device={args.device} "
        f"n_eval={args.n_eval} n_mean={args.n_mean} n_cie={args.n_cie} tag={tag}")

    # --- model ---
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tok = B.load_model(args.model, device=args.device, dtype=_dtype)
    device = args.device
    cfg = FV.neox_config(model)
    n_layers = cfg["n_layers"]
    edit_layer = args.edit_layer if args.edit_layer is not None \
        else round(n_layers / 3)
    log(f"config: layers={n_layers} heads={cfg['n_heads']} "
        f"resid={cfg['resid_dim']} head_dim={cfg['head_dim']} "
        f"edit_layer={edit_layer}")

    # --- dataset split ---
    pairs = FV.load_pairs(os.path.join(DATA_DIR, f"{args.task}.json"))
    train_pool, eval_pairs = FV.split_pairs(pairs, args.n_eval, seed=args.seed)
    eval_pairs = eval_pairs[:args.n_eval]
    log(f"dataset {args.task}: {len(pairs)} pairs -> train_pool "
        f"{len(train_pool)}, eval {len(eval_pairs)}")

    # eval calib/eval split (fixed seed)
    rng = np.random.default_rng(args.seed + 1)
    eidx = rng.permutation(len(eval_pairs))
    calib_pairs = [eval_pairs[i] for i in eidx[:args.n_calib]]
    heldout_pairs = [eval_pairs[i] for i in eidx[args.n_calib:]]
    log(f"eval split: calib {len(calib_pairs)} / eval {len(heldout_pairs)}")

    # =====================================================================
    # STAGE 1 — FV extraction
    # =====================================================================
    s1_cache = args.stage1_cache or os.path.join(
        outdir, f"stage1_{args.task}_{args.model.split('/')[-1]}.pt")
    cie_layers = None if args.no_cie_band else list(
        range(args.cie_lo, min(args.cie_hi, n_layers - 1) + 1))

    if os.path.exists(s1_cache):
        log(f"STAGE 1: loading cached {s1_cache}")
        blob = torch.load(s1_cache)
        fv_vec = blob["fv"]
        top = blob["top_heads"]
        mean_acts = blob["mean_acts"]
        cie = blob["cie"]
    else:
        log("STAGE 1.1: mean head activations (clean ICL)...")
        clean = FV.sample_icl_prompts(train_pool, args.n_mean, args.n_shots,
                                      seed=args.seed + 2, shuffle_labels=False)
        mean_acts = FV.mean_head_activations(model, tok, clean, cfg,
                                             device=device, log=log)
        log(f"  mean_acts shape={tuple(mean_acts.shape)} "
            f"norm={mean_acts.norm().item():.2f}")

        log(f"STAGE 1.2: CIE ranking (shuffled-label, band={cie_layers[0] if cie_layers else 'ALL'}"
            f"..{cie_layers[-1] if cie_layers else 'ALL'})...")
        shuffled = FV.sample_icl_prompts(train_pool, args.n_cie, args.n_shots,
                                         seed=args.seed + 3, shuffle_labels=True)
        cie = FV.compute_indirect_effect(model, tok, shuffled, mean_acts, cfg,
                                         layers=cie_layers, device=device, log=log)
        top = FV.top_heads(cie, args.n_top_heads)
        log(f"  top-{args.n_top_heads} CIE heads: {top}")

        log("STAGE 1.3: build FV...")
        fv_vec = FV.build_function_vector(model, mean_acts, top, cfg, device=device)
        log(f"  FV norm={fv_vec.norm().item():.3f}")

        torch.save({"fv": fv_vec, "top_heads": top, "mean_acts": mean_acts,
                    "cie": cie, "cie_layers": cie_layers,
                    "edit_layer": edit_layer, "cfg": cfg}, s1_cache)
        log(f"  cached Stage 1 -> {s1_cache}")

    fvm = FVMethod(model, tok, edit_layer, fv_vec, device=device,
                   max_new_tokens=args.max_new_tokens)
    fv_norm = fvm.norm
    # concentration stats for report
    top_layers = [L for (L, H, s) in top]
    cie_layer_hist = {}
    for L in top_layers:
        cie_layer_hist[L] = cie_layer_hist.get(L, 0) + 1

    result = {
        "meta": {
            "model": args.model, "device": device, "task": args.task, "tag": tag,
            "n_layers": n_layers, "n_heads": cfg["n_heads"],
            "resid_dim": cfg["resid_dim"], "head_dim": cfg["head_dim"],
            "edit_layer": edit_layer, "n_shots": args.n_shots,
            "n_mean": args.n_mean, "n_cie": args.n_cie,
            "cie_band": cie_layers, "n_top_heads": args.n_top_heads,
            "max_new_tokens": args.max_new_tokens, "n_boot": args.n_boot,
            "seed": args.seed, "n_eval_total": len(eval_pairs),
            "n_calib": len(calib_pairs), "n_eval": len(heldout_pairs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "stage1": {
            "top_heads": [{"layer": L, "head": H, "cie": s} for (L, H, s) in top],
            "top_head_layer_hist": cie_layer_hist,
            "fv_norm": fv_norm,
            "cie_band": cie_layers,
        },
    }

    # --- KV-baked sanity (E_first absence), 2 calib prompts ---
    log("KV-baked one-shot sanity (E_first absence)...")
    sanity = kv_baked_sanity(fvm, tok, [zs(p) for p in calib_pairs[:2]])
    result["kv_baked_sanity"] = sanity
    log(f"  all_match={sanity['all_match']}")

    # =====================================================================
    # STAGE 2 — battery on zero-shot prompts
    # =====================================================================
    def zs_pairs(prs):
        return ([FV.zero_shot_prompt(x) for (x, y) in prs],
                [y for (x, y) in prs])

    calib_prompts, calib_golds = zs_pairs(calib_pairs)
    eval_prompts, eval_golds = zs_pairs(heldout_pairs)

    # --- baseline on eval (task accuracy) ---
    log("EVAL: baseline...")
    base_texts, base_hits = generate_condition(
        lambda p: B.base_generate(model, tok, p, args.max_new_tokens, device),
        eval_prompts, eval_golds)
    r_base = B.bootstrap_rate_ci(base_hits, args.n_boot, seed=7)
    log(f"  baseline acc={r_base[0]*100:.1f}%")

    log("EVAL: E_all (FV all positions)...")
    all_texts, all_hits = generate_condition(
        lambda p: fvm.generate(p, "all"), eval_prompts, eval_golds)
    r_all = B.bootstrap_rate_ci(all_hits, args.n_boot, seed=7)
    log(f"  E_all acc={r_all[0]*100:.1f}%")

    # --- SANITY GATE: did the FV method reproduce? gain >= 10 pts ---
    raw_gain_pts = (r_all[0] - r_base[0]) * 100
    log(f"FV raw gain vs baseline = {raw_gain_pts:+.1f} pts "
        f"(Todd sanity threshold: >= 10 pts)")
    reproduced = raw_gain_pts >= 10.0

    log("EVAL: E_first (KV-baked one-shot)...")
    first_texts, first_hits = generate_condition(
        lambda p: fvm.generate(p, "first"), eval_prompts, eval_golds)
    r_first = B.bootstrap_rate_ci(first_hits, args.n_boot, seed=7)
    log(f"  E_first acc={r_first[0]*100:.1f}%")

    # --- floor: random vectors at matched norm, same site/positions (3 seeds) ---
    log("EVAL: floor (random dir @ matched norm, 3 seeds)...")
    H = cfg["resid_dim"]
    floor_runs = []
    for s in range(3):
        g = torch.Generator().manual_seed(2000 + s)
        rv = torch.randn(H, generator=g)
        rv = rv / rv.norm()
        addv = fv_norm * rv.to(device)
        ftexts = [fvm.meth.generate_with_vector(p, addv, "all_generated")
                  for p in eval_prompts]
        fhits = [int(answer_hit(t, g_)) for t, g_ in zip(ftexts, eval_golds)]
        fr = float(np.mean(fhits))
        floor_runs.append({"seed": 2000 + s, "rate": fr, "hits": fhits,
                           "texts": ftexts})
        log(f"  floor seed {2000+s}: acc={fr*100:.1f}%")
    floor_max = max(floor_runs, key=lambda r: r["rate"])
    r_floor = B.bootstrap_rate_ci(floor_max["hits"], args.n_boot, seed=7)

    # --- I2 control: token-set discovery + TF-KL matched bias (Amendment 1) ---
    log("I2 control: token-set discovery (calibration, position-1 delta)...")
    mean_delta = fv_position1_logit_delta(fvm, calib_prompts)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    top_ctrl_tokens = [tok.decode([i]) for i in ctrl_token_ids[:15]]
    log(f"  control token set size={len(ctrl_token_ids)} top={top_ctrl_tokens}")

    log("I2 control: unsteered continuations on calib (for TF-KL)...")
    calib_cont_ids = [B.base_generate_ids(model, tok, p, args.max_new_tokens,
                                          device) for p in calib_prompts]
    log("I2 control: B* = mean teacher-forced per-step KL (steered)...")
    target_kl = B.teacher_forced_stepkl_steered(
        fvm.meth, tok, calib_prompts, calib_cont_ids, 1.0)
    log(f"  B* (TF per-step KL, FV||base) = {target_kl:.5f}")
    c_scalar, achieved_kl = B.calibrate_bias_scalar_stepkl(
        model, tok, calib_prompts, calib_cont_ids, ctrl_token_ids, mean_delta,
        target_kl, device=device)
    log(f"  bias scalar c={c_scalar:.4f} achieved TF-KL={achieved_kl:.5f}")
    tid_t = torch.tensor(ctrl_token_ids)
    bias_vals = c_scalar * mean_delta[tid_t]
    processor = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)

    # sensitivity: position-1-matched control (demoted per Amendment 1)
    p1_target_kl = B._position1_kl_steered(fvm.meth, tok, calib_prompts, 1.0)
    p1_c, p1_ach = B.calibrate_bias_scalar(
        model, tok, calib_prompts, ctrl_token_ids, mean_delta, p1_target_kl,
        device=device)
    log(f"  [sensitivity] pos-1: target={p1_target_kl:.5f} c={p1_c:.4f} "
        f"achieved={p1_ach:.5f}")

    # first-token argmax flips (mechanism coordinate, reported)
    log("Mechanism: first-token argmax flips vs baseline (eval)...")
    n_flips, n_fp = B.first_token_flip_count(fvm.meth, tok, eval_prompts, 1.0)
    log(f"  flips = {n_flips}/{n_fp}")

    log("EVAL: control (calibrated logit bias)...")
    ctrl_texts, ctrl_hits = generate_condition(
        lambda p: B.control_generate(model, tok, p, processor,
                                     args.max_new_tokens, device),
        eval_prompts, eval_golds)
    r_ctrl = B.bootstrap_rate_ci(ctrl_hits, args.n_boot, seed=7)
    log(f"  control acc={r_ctrl[0]*100:.1f}%")

    # --- kappa, rho ---
    kappa = B.bootstrap_ratio_ci(first_hits, all_hits, base_hits, args.n_boot, seed=11)
    rho = B.bootstrap_ratio_ci(ctrl_hits, all_hits, base_hits, args.n_boot, seed=13)

    # --- degeneracy gate (per eval condition) ---
    log("degeneracy gate (eval conditions)...")
    ev_rep = float(np.mean([B.three_gram_rep_rate(t, tok) for t in base_texts]))
    ev_med = B.median_len_tokens(base_texts, tok)
    ev_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                            for p, t in zip(eval_prompts, base_texts)]))
    gates = {}
    for name, texts in [("E_all", all_texts), ("E_first", first_texts),
                        ("control", ctrl_texts), ("floor_max", floor_max["texts"])]:
        gg = B.degeneracy_gate(texts, eval_prompts, ev_rep, ev_med, ev_nll,
                               model, tok, device=device)
        gates[name] = {"tripped": gg.tripped, "rep": gg.rep_rate,
                       "median_len": gg.median_len, "nll": gg.mean_nll,
                       "reasons": gg.reasons}
        log(f"  {name}: tripped={gg.tripped} rep={gg.rep_rate:.3f} "
            f"med={gg.median_len:.1f} nll={gg.mean_nll:.3f}")

    # --- effect vs floor, verdict ---
    e_all = r_all[0] - r_base[0]
    e_floor = r_floor[0] - r_base[0]
    effect_over_floor = e_all / max(e_floor, 1e-9)
    effect_ge_3x_floor = e_all >= 3 * e_floor
    gate_clean_eall = not gates["E_all"]["tripped"]
    gate_clean_control = not gates["control"]["tripped"]
    cell_valid = effect_ge_3x_floor and gate_clean_eall and gate_clean_control
    # Genuine (amended §3): rho_hi <= 0.3 AND effect >= 3x floor.
    genuine = (rho[2] <= 0.3) and effect_ge_3x_floor
    dissolved = (rho[1] >= 0.9) and cell_valid
    if not reproduced:
        verdict = "STOP-NOT-REPRODUCED"
    elif genuine:
        verdict = "Genuine"
    elif dissolved:
        verdict = "Dissolved"
    else:
        verdict = "Mixed"

    result.update({
        "sanity_reproduction": {
            "raw_gain_pts": raw_gain_pts, "threshold_pts": 10.0,
            "reproduced": bool(reproduced),
            "note": "FV zero-shot accuracy gain vs baseline; must be >=10 pts "
                    "before auditing (do not audit a non-effect).",
        },
        "rates": {
            "baseline": {"rate": r_base[0], "ci_lo": r_base[1], "ci_hi": r_base[2]},
            "E_all": {"rate": r_all[0], "ci_lo": r_all[1], "ci_hi": r_all[2]},
            "E_first": {"rate": r_first[0], "ci_lo": r_first[1], "ci_hi": r_first[2]},
            "control": {"rate": r_ctrl[0], "ci_lo": r_ctrl[1], "ci_hi": r_ctrl[2]},
            "floor_max": {"rate": r_floor[0], "ci_lo": r_floor[1], "ci_hi": r_floor[2]},
        },
        "kappa": {"point": kappa[0], "ci_lo": kappa[1], "ci_hi": kappa[2]},
        "rho": {"point": rho[0], "ci_lo": rho[1], "ci_hi": rho[2]},
        "effect": {"E_all": e_all, "E_floor": e_floor,
                   "effect_over_floor": effect_over_floor},
        "control_calibration": {
            "token_set_size": len(ctrl_token_ids), "token_ids": ctrl_token_ids,
            "top_tokens": top_ctrl_tokens,
            "budget": "mean_teacher_forced_per_step_KL (Amendment 1)",
            "B_star_target_kl": target_kl, "achieved_kl": achieved_kl,
            "bias_scalar": c_scalar,
            "sensitivity_position1": {"target_kl": p1_target_kl,
                                      "achieved_kl": p1_ach, "bias_scalar": p1_c},
        },
        "mechanism_check": {"first_token_flips": n_flips, "n_prompts": n_fp},
        "floor_runs": [{"seed": r["seed"], "rate": r["rate"]} for r in floor_runs],
        "degeneracy_gates": gates,
        "eval_baseline_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
        "verdict": {
            "class": verdict,
            "genuine_criterion": "rho_hi <= 0.3 AND E_all >= 3x floor (cell valid)",
            "rho_hi": rho[2], "rho_lo": rho[1], "kappa_point": kappa[0],
            "effect_ge_3x_floor": bool(effect_ge_3x_floor),
            "cell_valid": bool(cell_valid), "reproduced": bool(reproduced),
            "passes_genuine": bool(genuine),
        },
        "samples": {
            "baseline": list(zip(eval_prompts[:8], base_texts[:8], eval_golds[:8])),
            "E_all": list(zip(eval_prompts[:8], all_texts[:8], eval_golds[:8])),
            "control": list(zip(eval_prompts[:8], ctrl_texts[:8], eval_golds[:8])),
        },
        "runtime_sec": time.time() - t0,
    })

    jpath = os.path.join(outdir, f"results_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"wrote {jpath}")
    if not args.smoke:
        # canonical name for the primary (antonym full) run
        with open(os.path.join(outdir, "results_full.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        rpath = os.path.join(outdir, "report.md")
        write_report(result, rpath)
        log(f"wrote {rpath}")
    log(f"DONE in {result['runtime_sec']:.0f}s  verdict={verdict}  "
        f"raw_gain={raw_gain_pts:+.1f}pts")


def zs(prompt_pair):
    return FV.zero_shot_prompt(prompt_pair[0])


def kv_baked_sanity(fvm: "FVMethod", tokenizer, prompts):
    """E_first absence check: re-derive positions='last_prompt' by hand (patch
    only prefill last token, then no hook) and compare to actlib's output."""
    from actlib.patching import _dynamic_patch_hook
    model, device = fvm.model, fvm.device
    Hh = fvm.fv.shape[0]
    L = fvm.layer
    addv = fvm.fv
    rb = torch.zeros(0, Hh, device=device)
    res = []
    for p in prompts:
        lib = fvm.meth.generate_with_vector(p, addv, "last_prompt")
        enc = tokenizer(p, return_tensors="pt")
        iid = enc["input_ids"].to(device)
        st = {"cache_offset": 0, "targets": [iid.shape[1] - 1]}
        with torch.no_grad(), _dynamic_patch_hook(
                model, "resid_post", L, None, None, "subspace_transplant", st,
                remove_subspace=rb, add_vector=addv):
            out = model(iid, use_cache=True)
        past = out.past_key_values
        nid = int(out.logits[0, -1].argmax())
        gen = [nid]
        cur = torch.tensor([[nid]], device=device)
        with torch.no_grad():
            for _ in range(fvm.max_new_tokens - 1):
                o = model(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nid = int(o.logits[0, -1].argmax())
                gen.append(nid)
                cur = torch.tensor([[nid]], device=device)
        manual = tokenizer.decode(torch.tensor(gen), skip_special_tokens=True)
        res.append(lib.strip() == manual.strip())
    return {"n": len(prompts), "all_match": bool(all(res)), "matches": res}


def _fmt_ci(d):
    return f"{d['rate']*100:.1f}% [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"


def write_report(r, path):
    m = r["meta"]; v = r["verdict"]; s1 = r["stage1"]
    L = []
    A = L.append
    A(f"# A1 anchor — function vectors (steering-content-audit)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` ({m['n_layers']} layers, {m['n_heads']} heads, "
      f"resid {m['resid_dim']}, head_dim {m['head_dim']}), device `{m['device']}`  ")
    A(f"**Task:** {m['task']} (zero-shot eval).  ")
    A(f"**Prompts:** {m['n_eval_total']} held-out items (calib {m['n_calib']} / "
      f"eval {m['n_eval']}); {m['max_new_tokens']} new tokens greedy; "
      f"{m['n_boot']} bootstrap resamples.\n")

    A(f"## VERDICT: **{v['class']}**\n")
    A(f"Genuine (amended plan §3) requires `rho_hi <= 0.3` AND `E_all >= 3x "
      f"floor`. A fixed prompt-independent output push cannot produce "
      f"input-dependent correct answers, so a genuine FV should dissolve NO "
      f"control.\n")
    sr = r["sanity_reproduction"]
    A(f"- **FV reproduction sanity:** raw gain = **{sr['raw_gain_pts']:+.1f} "
      f"pts** vs baseline (threshold >= {sr['threshold_pts']:.0f}; "
      f"reproduced = **{sr['reproduced']}**). "
      f"{'Method reproduced; audit proceeds.' if sr['reproduced'] else 'FV DID NOT reproduce -> STOP, do not audit a non-effect.'}")
    A(f"- rho_hi = **{v['rho_hi']:.3f}** (need <= 0.30 for Genuine)")
    A(f"- E_all >= 3x floor = **{v['effect_ge_3x_floor']}** (cell valid = "
      f"{v['cell_valid']})")
    A(f"- kappa (E_first/E_all) = **{v['kappa_point']:.3f}** (reported "
      f"coordinate, not gating)")
    A(f"- passes Genuine = **{v['passes_genuine']}**\n")

    A(f"## Stage 1 — FV extraction\n")
    A(f"- **ICL template** (Todd et al.): prefixes `{{input:'Q:', output:'A:'}}`, "
      f"separators `{{input:'\\n', output:'\\n\\n'}}`, prepend_space; "
      f"{m['n_shots']}-shot.")
    A(f"- **Mean head activations:** dense-input head slices at the final prompt "
      f"token, averaged over {m['n_mean']} clean ICL prompts.")
    band = m["cie_band"]
    A(f"- **CIE ranking:** {m['n_cie']} shuffled-label ICL prompts; per-head "
      f"patch of the task-mean at the last token, correct-answer first-token "
      f"logprob recovery vs shuffled baseline. Search band = layers "
      f"{band[0]}..{band[-1]} (restricted to bound compute; logged).")
    A(f"- **FV** = sum of top-{m['n_top_heads']} CIE heads' means through "
      f"`dense`; **FV norm = {s1['fv_norm']:.3f}**.")
    A(f"- **Insertion:** resid_post at layer **{m['edit_layer']}** "
      f"(~L/3 of {m['n_layers']}), added at the final prompt token and every "
      f"generated position (E_all).")
    A(f"- **KV-baked one-shot sanity (E_first absence):** all_match="
      f"{r['kv_baked_sanity']['all_match']}.\n")
    A(f"### CIE top-{m['n_top_heads']} heads\n")
    A(f"Concentration by layer: {s1['top_head_layer_hist']} "
      f"(Todd found FV heads concentrate in mid layers).\n")
    A(f"| rank | layer | head | CIE score |")
    A(f"|-----:|------:|-----:|----------:|")
    for i, h in enumerate(s1["top_heads"], 1):
        A(f"| {i} | {h['layer']} | {h['head']} | {h['cie']:.4f} |")
    A("")

    A(f"## Headline rates (eval split, {m['n_eval']} prompts)\n")
    A(f"Task accuracy (gold word in 8-token greedy generation), bootstrap "
      f"95% CI.\n")
    rr = r["rates"]
    A(f"| condition | accuracy [95% CI] |")
    A(f"|---|---|")
    A(f"| baseline (zero-shot, no FV) | {_fmt_ci(rr['baseline'])} |")
    A(f"| E_all (FV all positions) | {_fmt_ci(rr['E_all'])} |")
    A(f"| E_first (KV-baked one-shot) | {_fmt_ci(rr['E_first'])} |")
    A(f"| control (calibrated logit bias) | {_fmt_ci(rr['control'])} |")
    A(f"| floor (random dir @ matched norm, max of 3) | {_fmt_ci(rr['floor_max'])} |")
    A("")

    A(f"## Decomposition\n")
    k = r["kappa"]; rho = r["rho"]; e = r["effect"]
    A(f"- **rho = E(control)/E(FV)** = {rho['point']:.3f} "
      f"[{rho['ci_lo']:.3f}, {rho['ci_hi']:.3f}]  (Genuine needs rho_hi <= 0.30)")
    A(f"- **kappa = E_first/E_all** = {k['point']:.3f} "
      f"[{k['ci_lo']:.3f}, {k['ci_hi']:.3f}] (coordinate)")
    A(f"- E_all = {e['E_all']*100:.1f} pts; floor effect = {e['E_floor']*100:.1f} "
      f"pts; FV / floor = {e['effect_over_floor']:.2f}x (needs >= 3x).\n")

    cc = r["control_calibration"]
    A(f"## I2 control calibration (Amendment 1: teacher-forced per-step KL)\n")
    A(f"- Token set S: {cc['token_set_size']} tokens (90% of ||position-1 "
      f"logit-delta||^2, cap 100). Top: {cc['top_tokens']}")
    A(f"- Budget = mean teacher-forced per-step KL over all "
      f"{m['max_new_tokens']} continuation positions x calib prompts, on fixed "
      f"unsteered continuations.")
    A(f"- B* (FV, TF per-step KL) = {cc['B_star_target_kl']:.5f}; achieved "
      f"control TF-KL = {cc['achieved_kl']:.5f}; bias scalar = "
      f"{cc['bias_scalar']:.4f}.")
    sp = cc["sensitivity_position1"]
    A(f"- Sensitivity (position-1-matched, demoted): target KL "
      f"{sp['target_kl']:.5f}, achieved {sp['achieved_kl']:.5f}, c="
      f"{sp['bias_scalar']:.4f}.")
    mc = r["mechanism_check"]
    A(f"- First-token argmax flips vs baseline under FV = "
      f"{mc['first_token_flips']}/{mc['n_prompts']}.\n")

    A(f"## Degeneracy gate (per eval condition)\n")
    er = r["eval_baseline_refs"]
    A(f"Eval baseline refs: rep={er['rep']:.3f}, median_len={er['median_len']:.1f}, "
      f"nll={er['nll']:.3f}. Gate: rep > 2x+0.1, or median_len < 0.5x, or nll > 3x.\n")
    A(f"| condition | tripped | rep | median_len | nll | reasons |")
    A(f"|---|:---:|---:|---:|---:|---|")
    for name, g in r["degeneracy_gates"].items():
        A(f"| {name} | {'VOID' if g['tripped'] else 'ok'} | {g['rep']:.3f} | "
          f"{g['median_len']:.1f} | {g['nll']:.3f} | {'; '.join(g['reasons'])} |")
    A("")
    A(f"## Floor runs\n")
    for fr in r["floor_runs"]:
        A(f"- seed {fr['seed']}: acc {fr['rate']*100:.1f}%")
    A("")
    A(f"## Sample generations (first 8 eval)\n")
    for cond in ["baseline", "E_all", "control"]:
        A(f"**{cond}:**")
        for (p, t, g) in r["samples"][cond]:
            A(f"- `{p}` -> `{t.strip()[:40]}` (gold: {g})")
        A("")
    A(f"Runtime: {r['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
