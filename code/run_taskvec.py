"""Task-vectors arm driver — Hendel et al. 2310.15916 ("In-Context Learning
Creates Task Vectors", the single-residual-vector variant), on Pythia-2.8B.

Distinct from A1 (Todd function-vectors): theta = MEAN full resid_post[L] at the
last ICL demo-context token; injection = REPLACE resid at the zero-shot query's
final token with theta (Hendel patches, not adds).

Stages
------
Stage 1 (sweep): extract theta at each L in {8,12,14,16,18,20}; measure zero-shot
  REPLACE accuracy on the calib split; reproduction gate = gain >= +25 pts at
  some L. Chosen L = best clean gain. If REPLACE never reaches +25, retry ADD as
  sensitivity and report which reproduced.
Stage 2 (battery, at chosen L / op): baseline, E_first (KV-baked one-shot single
  write = native regime), E_all (forced broadcast, DIAGNOSTIC), calibrated logit-
  bias control (Amendment 1 TF-per-step-KL budget matched to E_native = E_first),
  floor (random dir at matched norm, REPLACE, 3 seeds). Degeneracy gate, bootstrap
  10k, rho, kappa.

Verdict (plan §3, computed from CI bounds + cell_valid exactly):
  Genuine   = rho_hi <= 0.3 AND E_native >= 3x floor.
  Dissolved = rho_lo >= 0.9 AND cell valid (E_native >= 3x floor, gate clean).
  else Mixed. If the primary control cell is VOID and the verdict would be
  Genuine, Amendment-2 dose-response is required (run separately if needed).

kappa (Amendment 3): native = single-position write propagating through the
sampling loop (prompt-injection family), so E_native = E_first and
kappa = E_first/E_native ~= 1. E_all is a DIAGNOSTIC only; kappa_diag =
E_first/E_all is also reported.

rho denominator = E_native (= E_first), the as-published effect.

Usage:
  .venv/bin/python run_taskvec.py --smoke --n-eval 20 --n-mean 20 \
      --layers 12,16                                     # smoke
  .venv/bin/python run_taskvec.py                        # full (2.8b, mps, bf16)
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import battery as B          # noqa: E402
import fv_extract as FV      # noqa: E402
import taskvec as TV         # noqa: E402


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "external", "function_vectors", "dataset_files", "abstractive")

SWEEP_LAYERS = [8, 12, 14, 16, 18, 20]
REPRO_GATE_PTS = 25.0


def answer_hit(gen_text: str, gold: str) -> bool:
    if not gold:
        return False
    return re.search(r"\b" + re.escape(gold.lower()) + r"\b",
                     gen_text.lower()) is not None


def hits_for(texts, golds):
    return [int(answer_hit(t, g)) for t, g in zip(texts, golds)]


# ---------------------------------------------------------------------------
# Disk staging (resume): stage1 sweep + stage2 conditions
# ---------------------------------------------------------------------------

def load_stage(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"theta": {}, "sweep": {}, "stage2": {}}


def save_stage(path, stage):
    with open(path, "w") as f:
        json.dump(stage, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16", choices=["float32", "bf16"])
    ap.add_argument("--task", default="antonym")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--n-mean", type=int, default=100,
                    help="ICL demo contexts for theta")
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--layers", default=None,
                    help="comma list; default 8,12,14,16,18,20")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(SWEEP_LAYERS))

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = args.outdir or os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-07-taskvec-arm")
    os.makedirs(outdir, exist_ok=True)
    tag = f"{args.task}_" + ("smoke" if args.smoke else "full")
    stage_path = os.path.join(outdir, f"stage_{tag}.json")
    stage = load_stage(stage_path)
    t0 = time.time()
    log(f"model={args.model} task={args.task} device={args.device} "
        f"layers={layers} n_eval={args.n_eval} n_mean={args.n_mean} tag={tag}")

    # --- model ---
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tok = B.load_model(args.model, device=args.device, dtype=_dtype)
    device = args.device
    cfg = FV.neox_config(model)
    n_layers = cfg["n_layers"]
    log(f"config: layers={n_layers} heads={cfg['n_heads']} "
        f"resid={cfg['resid_dim']} head_dim={cfg['head_dim']}")

    # --- dataset split (same discipline as A1) ---
    pairs = FV.load_pairs(os.path.join(DATA_DIR, f"{args.task}.json"))
    train_pool, eval_pairs = FV.split_pairs(pairs, args.n_eval, seed=args.seed)
    eval_pairs = eval_pairs[:args.n_eval]
    rng = np.random.default_rng(args.seed + 1)
    eidx = rng.permutation(len(eval_pairs))
    calib_pairs = [eval_pairs[i] for i in eidx[:args.n_calib]]
    heldout_pairs = [eval_pairs[i] for i in eidx[args.n_calib:]]
    log(f"dataset {args.task}: {len(pairs)} pairs -> train_pool "
        f"{len(train_pool)}, eval {len(eval_pairs)} "
        f"(calib {len(calib_pairs)} / eval {len(heldout_pairs)})")

    def zs_pairs(prs):
        return ([FV.zero_shot_prompt(x) for (x, y) in prs],
                [y for (x, y) in prs])

    calib_prompts, calib_golds = zs_pairs(calib_pairs)
    eval_prompts, eval_golds = zs_pairs(heldout_pairs)

    # ICL demo contexts for theta extraction (disjoint from eval: drawn from
    # train_pool, same discipline as A1 mean-head extraction).
    clean = FV.sample_icl_prompts(train_pool, args.n_mean, args.n_shots,
                                  seed=args.seed + 2, shuffle_labels=False)

    # =====================================================================
    # STAGE 1 — theta extraction + layer sweep (REPLACE, then ADD fallback)
    # =====================================================================
    def get_theta(L):
        key = str(L)
        if key in stage["theta"]:
            return torch.tensor(stage["theta"][key], dtype=torch.float32)
        log(f"STAGE1: extract theta at L={L} ({args.n_mean} contexts)...")
        theta = TV.mean_task_vector(model, tok, clean, L, device=device, log=log)
        stage["theta"][key] = theta.tolist()
        save_stage(stage_path, stage)
        return theta

    # calib baseline (shared across sweep ops)
    if "calib_baseline_hits" in stage.get("sweep", {}):
        calib_base_hits = stage["sweep"]["calib_baseline_hits"]
    else:
        log("STAGE1: calib baseline (zero-shot, no injection)...")
        cbt = [TV.base_generate(model, tok, p, args.max_new_tokens, device)
               for p in calib_prompts]
        calib_base_hits = hits_for(cbt, calib_golds)
        stage["sweep"]["calib_baseline_hits"] = calib_base_hits
        save_stage(stage_path, stage)
    calib_base_rate = float(np.mean(calib_base_hits))
    log(f"  calib baseline acc={calib_base_rate*100:.1f}%")

    def sweep_op(op):
        rows = []
        for L in layers:
            key = f"{op}_{L}"
            if key in stage["sweep"]:
                rows.append(stage["sweep"][key])
                log(f"  sweep {op} L={L}: cached gain="
                    f"{stage['sweep'][key]['gain_pts']:+.1f} pts")
                continue
            theta = get_theta(L)
            meth = TV.TaskVecMethod(model, tok, L, theta, op=op, device=device,
                                    max_new_tokens=args.max_new_tokens)
            texts = [meth.generate(p, "first") for p in calib_prompts]  # native
            hits = hits_for(texts, calib_golds)
            rate = float(np.mean(hits))
            gain = (rate - calib_base_rate) * 100
            row = {"op": op, "layer": L, "theta_norm": meth.norm,
                   "calib_acc": rate, "gain_pts": gain,
                   "sample": texts[0][:40] if texts else ""}
            stage["sweep"][key] = row
            save_stage(stage_path, stage)
            rows.append(row)
            log(f"  sweep {op} L={L}: theta_norm={meth.norm:.2f} "
                f"calib_acc={rate*100:.1f}% gain={gain:+.1f} pts")
        return rows

    log("STAGE1: layer sweep with REPLACE (native E_first regime)...")
    replace_rows = sweep_op("replace")
    best_replace = max(replace_rows, key=lambda r: r["gain_pts"])
    replace_reproduced = best_replace["gain_pts"] >= REPRO_GATE_PTS

    add_rows = None
    if replace_reproduced:
        chosen_op = "replace"
        chosen_row = best_replace
    else:
        log(f"REPLACE best gain {best_replace['gain_pts']:+.1f} < "
            f"{REPRO_GATE_PTS} pts; retrying ADD as sensitivity...")
        add_rows = sweep_op("add")
        best_add = max(add_rows, key=lambda r: r["gain_pts"])
        if best_add["gain_pts"] >= REPRO_GATE_PTS:
            chosen_op, chosen_row = "add", best_add
        else:
            # neither reproduced; still report the stronger for transparency
            if best_add["gain_pts"] > best_replace["gain_pts"]:
                chosen_op, chosen_row = "add", best_add
            else:
                chosen_op, chosen_row = "replace", best_replace

    chosen_L = chosen_row["layer"]
    reproduced = chosen_row["gain_pts"] >= REPRO_GATE_PTS
    log(f"CHOSEN: op={chosen_op} L={chosen_L} "
        f"calib_gain={chosen_row['gain_pts']:+.1f} pts reproduced={reproduced}")

    theta = get_theta(chosen_L)
    meth = TV.TaskVecMethod(model, tok, chosen_L, theta, op=chosen_op,
                            device=device, max_new_tokens=args.max_new_tokens)
    theta_norm = meth.norm

    # =====================================================================
    # STAGE 2 — battery on zero-shot eval
    # =====================================================================
    def s2(key, fn):
        if key in stage["stage2"]:
            return stage["stage2"][key]
        v = fn()
        stage["stage2"][key] = v
        save_stage(stage_path, stage)
        return v

    log("EVAL: baseline...")
    base = s2("base", lambda: {"texts": [
        TV.base_generate(model, tok, p, args.max_new_tokens, device)
        for p in eval_prompts]})
    base_texts = base["texts"]
    base_hits = hits_for(base_texts, eval_golds)
    r_base = B.bootstrap_rate_ci(base_hits, args.n_boot, seed=7)
    log(f"  baseline acc={r_base[0]*100:.1f}%")

    log("EVAL: E_first (native single write, KV-baked)...")
    first = s2("first", lambda: {"texts": [
        meth.generate(p, "first") for p in eval_prompts]})
    first_texts = first["texts"]
    first_hits = hits_for(first_texts, eval_golds)
    r_first = B.bootstrap_rate_ci(first_hits, args.n_boot, seed=7)
    log(f"  E_first (native) acc={r_first[0]*100:.1f}%")

    # native effect (as published) = E_first
    raw_gain_pts = (r_first[0] - r_base[0]) * 100
    log(f"native (E_first) gain vs baseline = {raw_gain_pts:+.1f} pts")

    log("EVAL: E_all (forced broadcast — DIAGNOSTIC)...")
    allc = s2("all", lambda: {"texts": [
        meth.generate(p, "all") for p in eval_prompts]})
    all_texts = allc["texts"]
    all_hits = hits_for(all_texts, eval_golds)
    r_all = B.bootstrap_rate_ci(all_hits, args.n_boot, seed=7)
    log(f"  E_all (diagnostic) acc={r_all[0]*100:.1f}%")

    # --- floor: random dir @ matched norm, same op/native regime, 3 seeds ---
    log("EVAL: floor (random dir @ matched norm, native regime, 3 seeds)...")
    Hd = cfg["resid_dim"]
    floor_runs = []
    for sd in range(3):
        key = f"floor_{sd}"
        if key in stage["stage2"]:
            fr = stage["stage2"][key]
            floor_runs.append(fr)
            log(f"  floor seed {2000+sd}: cached acc={fr['rate']*100:.1f}%")
            continue
        g = torch.Generator().manual_seed(2000 + sd)
        rv = torch.randn(Hd, generator=g)
        rv = rv / rv.norm()
        vec = (theta_norm * rv).to(device)
        ftexts = [meth.generate_with_vector(p, vec, chosen_op, "first")
                  for p in eval_prompts]
        fhits = hits_for(ftexts, eval_golds)
        fr = {"seed": 2000 + sd, "rate": float(np.mean(fhits)),
              "hits": fhits, "texts": ftexts}
        stage["stage2"][key] = fr
        save_stage(stage_path, stage)
        floor_runs.append(fr)
        log(f"  floor seed {2000+sd}: acc={fr['rate']*100:.1f}%")
    floor_max = max(floor_runs, key=lambda r: r["rate"])
    r_floor = B.bootstrap_rate_ci(floor_max["hits"], args.n_boot, seed=7)

    # --- I2 control: token-set discovery + TF-KL matched to E_native (=E_first) ---
    log("I2 control: position-1 logit delta on calib (token-set discovery)...")
    ctrl = stage["stage2"].get("control")
    if ctrl is None:
        mean_delta = meth.position1_logit_delta(calib_prompts)
        ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
        log(f"  token set size={len(ctrl_token_ids)}")

        log("I2 control: unsteered calib continuations (TF-KL)...")
        calib_cont_ids = [B.base_generate_ids(model, tok, p, args.max_new_tokens,
                                              device) for p in calib_prompts]
        # B* = TF per-step KL of the NATIVE injection. Native regime is a single
        # first-position write; but the additive-family control budget is defined
        # (Amendment 1) as mean TF per-step KL of the steering method applied at
        # every continuation position. For a single-position write the honest
        # analogue matching E_native's mechanism is the KV-baked propagation; we
        # follow the A0/A1 convention and use the all-position write's TF-KL as
        # the budget quantity B* (an UPPER bound on the native per-step push,
        # conservative — it can only make the control stronger, never weaker).
        target_kl = meth.teacher_forced_stepkl(calib_prompts, calib_cont_ids)
        log(f"  B* (TF per-step KL, theta||base) = {target_kl:.5f}")
        c_scalar, achieved_kl = B.calibrate_bias_scalar_stepkl(
            model, tok, calib_prompts, calib_cont_ids, ctrl_token_ids,
            mean_delta, target_kl, device=device)
        log(f"  bias scalar c={c_scalar:.4f} achieved TF-KL={achieved_kl:.5f}")

        p1_target = B._position1_kl_biased  # noqa (not used directly)
        # position-1-matched sensitivity control
        p1_kl = _theta_position1_kl(meth, calib_prompts)
        p1_c, p1_ach = B.calibrate_bias_scalar(
            model, tok, calib_prompts, ctrl_token_ids, mean_delta, p1_kl,
            device=device)
        log(f"  [sensitivity] pos-1: target={p1_kl:.5f} c={p1_c:.4f} "
            f"achieved={p1_ach:.5f}")

        tid_t = torch.tensor(ctrl_token_ids)
        bias_vals = c_scalar * mean_delta[tid_t]
        processor = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)
        ctrl_texts = [B.control_generate(model, tok, p, processor,
                                         args.max_new_tokens, device)
                      for p in eval_prompts]
        n_flips, n_fp = meth.first_token_flip_count(eval_prompts)
        ctrl = {
            "token_ids": ctrl_token_ids,
            "top_tokens": [tok.decode([i]) for i in ctrl_token_ids[:15]],
            "B_star_target_kl": target_kl, "achieved_kl": achieved_kl,
            "bias_scalar": c_scalar,
            "p1_target_kl": p1_kl, "p1_bias_scalar": p1_c, "p1_achieved": p1_ach,
            "texts": ctrl_texts, "n_flips": n_flips, "n_fp": n_fp,
            "mean_delta_topvals": mean_delta[tid_t][:15].tolist(),
        }
        stage["stage2"]["control"] = ctrl
        save_stage(stage_path, stage)
    ctrl_token_ids = ctrl["token_ids"]
    ctrl_texts = ctrl["texts"]
    ctrl_hits = hits_for(ctrl_texts, eval_golds)
    r_ctrl = B.bootstrap_rate_ci(ctrl_hits, args.n_boot, seed=7)
    log(f"  control acc={r_ctrl[0]*100:.1f}%; first-token flips="
        f"{ctrl['n_flips']}/{ctrl['n_fp']}")

    # --- kappa (native = E_first), rho (denominator = E_native = E_first) ---
    # kappa native-regime = E_first/E_native = 1 by construction; report the
    # diagnostic kappa_diag = E_first/E_all as the cascade-vs-broadcast coordinate.
    kappa_native = 1.0
    kappa_diag = B.bootstrap_ratio_ci(first_hits, all_hits, base_hits,
                                      args.n_boot, seed=11)
    rho = B.bootstrap_ratio_ci(ctrl_hits, first_hits, base_hits,
                               args.n_boot, seed=13)

    # --- degeneracy gate ---
    log("degeneracy gate (eval conditions)...")
    ev_rep = float(np.mean([B.three_gram_rep_rate(t, tok) for t in base_texts]))
    ev_med = B.median_len_tokens(base_texts, tok)
    ev_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                            for p, t in zip(eval_prompts, base_texts)]))
    gates = {}
    for name, texts in [("E_first", first_texts), ("E_all", all_texts),
                        ("control", ctrl_texts), ("floor_max", floor_max["texts"])]:
        gg = B.degeneracy_gate(texts, eval_prompts, ev_rep, ev_med, ev_nll,
                               model, tok, device=device)
        gates[name] = {"tripped": gg.tripped, "rep": gg.rep_rate,
                       "median_len": gg.median_len, "nll": gg.mean_nll,
                       "reasons": gg.reasons}
        log(f"  {name}: tripped={gg.tripped} rep={gg.rep_rate:.3f} "
            f"med={gg.median_len:.1f} nll={gg.mean_nll:.3f}")

    # --- effect vs floor, verdict (native regime; denominator = E_first) ---
    e_native = r_first[0] - r_base[0]
    e_floor = r_floor[0] - r_base[0]
    effect_over_floor = e_native / max(e_floor, 1e-9)
    effect_ge_3x_floor = e_native >= 3 * e_floor
    gate_clean_native = not gates["E_first"]["tripped"]
    gate_clean_control = not gates["control"]["tripped"]
    cell_valid = effect_ge_3x_floor and gate_clean_native and gate_clean_control

    # Verdict from CI bounds + cell_valid exactly (plan §3, amended):
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
    # Amendment-2 flag: Genuine with a VOID control cell needs a dose-response.
    needs_amend2 = genuine and gates["control"]["tripped"]

    # --- geometry: cos(theta, W_U answer-span) ---
    log("geometry: cos(theta, W_U answer-span)...")
    geom = _geometry(model, tok, theta, heldout_pairs, device)

    result = {
        "meta": {
            "arm": "task-vectors (Hendel et al. 2310.15916, single-residual "
                   "variant, REPLACE)",
            "model": args.model, "device": device, "dtype": args.dtype,
            "task": args.task, "tag": tag,
            "n_layers": n_layers, "n_heads": cfg["n_heads"],
            "resid_dim": cfg["resid_dim"], "head_dim": cfg["head_dim"],
            "sweep_layers": layers, "chosen_layer": chosen_L,
            "chosen_op": chosen_op, "theta_norm": theta_norm,
            "n_shots": args.n_shots, "n_mean": args.n_mean,
            "max_new_tokens": args.max_new_tokens, "n_boot": args.n_boot,
            "seed": args.seed, "repro_gate_pts": REPRO_GATE_PTS,
            "n_eval_total": len(eval_pairs), "n_calib": len(calib_pairs),
            "n_eval": len(heldout_pairs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "stage1_sweep": {
            "replace": replace_rows, "add": add_rows,
            "calib_baseline_acc": calib_base_rate,
            "best_replace_gain_pts": best_replace["gain_pts"],
            "replace_reproduced": bool(replace_reproduced),
        },
        "sanity_reproduction": {
            "raw_gain_pts": raw_gain_pts, "threshold_pts": REPRO_GATE_PTS,
            "reproduced": bool(reproduced), "op": chosen_op,
            "note": "native (E_first) zero-shot accuracy gain vs baseline; must "
                    ">= 25 pts to audit; REPLACE primary, ADD sensitivity.",
        },
        "rates": {
            "baseline": {"rate": r_base[0], "ci_lo": r_base[1], "ci_hi": r_base[2]},
            "E_native_first": {"rate": r_first[0], "ci_lo": r_first[1],
                               "ci_hi": r_first[2]},
            "E_all_diagnostic": {"rate": r_all[0], "ci_lo": r_all[1],
                                 "ci_hi": r_all[2]},
            "control": {"rate": r_ctrl[0], "ci_lo": r_ctrl[1], "ci_hi": r_ctrl[2]},
            "floor_max": {"rate": r_floor[0], "ci_lo": r_floor[1],
                          "ci_hi": r_floor[2]},
        },
        "kappa": {
            "native": kappa_native,
            "diagnostic_first_over_all": {"point": kappa_diag[0],
                                          "ci_lo": kappa_diag[1],
                                          "ci_hi": kappa_diag[2]},
            "note": "kappa native = E_first/E_native = 1 (single-position write, "
                    "prompt-injection family, Amendment 3). E_all is diagnostic.",
        },
        "rho": {"point": rho[0], "ci_lo": rho[1], "ci_hi": rho[2],
                "denominator": "E_native (=E_first)"},
        "effect": {"E_native": e_native, "E_floor": e_floor,
                   "effect_over_floor": effect_over_floor},
        "control_calibration": {
            "token_set_size": len(ctrl_token_ids), "token_ids": ctrl_token_ids,
            "top_tokens": ctrl["top_tokens"],
            "budget": "mean_teacher_forced_per_step_KL matched to E_native "
                      "(Amendment 1)",
            "B_star_target_kl": ctrl["B_star_target_kl"],
            "achieved_kl": ctrl["achieved_kl"],
            "bias_scalar": ctrl["bias_scalar"],
            "sensitivity_position1": {"target_kl": ctrl["p1_target_kl"],
                                      "achieved_kl": ctrl["p1_achieved"],
                                      "bias_scalar": ctrl["p1_bias_scalar"]},
        },
        "mechanism_check": {"first_token_flips": ctrl["n_flips"],
                            "n_prompts": ctrl["n_fp"]},
        "geometry": geom,
        "floor_runs": [{"seed": r["seed"], "rate": r["rate"]} for r in floor_runs],
        "degeneracy_gates": gates,
        "eval_baseline_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
        "verdict": {
            "class": verdict,
            "rule": "Genuine = rho_hi<=0.3 AND E_native>=3x floor; "
                    "Dissolved = rho_lo>=0.9 AND cell_valid; else Mixed "
                    "(plan §3, from CI bounds + cell_valid).",
            "rho_hi": rho[2], "rho_lo": rho[1],
            "kappa_native": kappa_native,
            "kappa_diag_point": kappa_diag[0],
            "effect_ge_3x_floor": bool(effect_ge_3x_floor),
            "cell_valid": bool(cell_valid), "reproduced": bool(reproduced),
            "passes_genuine": bool(genuine), "passes_dissolved": bool(dissolved),
            "control_cell_void": bool(gates["control"]["tripped"]),
            "needs_amendment2_doseresponse": bool(needs_amend2),
        },
        "samples": {
            "baseline": list(zip(eval_prompts[:8], base_texts[:8], eval_golds[:8])),
            "E_native_first": list(zip(eval_prompts[:8], first_texts[:8],
                                       eval_golds[:8])),
            "E_all_diagnostic": list(zip(eval_prompts[:8], all_texts[:8],
                                         eval_golds[:8])),
            "control": list(zip(eval_prompts[:8], ctrl_texts[:8], eval_golds[:8])),
        },
        "runtime_sec": time.time() - t0,
    }

    jpath = os.path.join(outdir, f"results_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"wrote {jpath}")
    if not args.smoke:
        with open(os.path.join(outdir, "results_full.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
        write_report(result, os.path.join(outdir, "report.md"))
        log(f"wrote {os.path.join(outdir, 'report.md')}")
    log(f"DONE in {result['runtime_sec']:.0f}s verdict={verdict} "
        f"op={chosen_op} L={chosen_L} native_gain={raw_gain_pts:+.1f}pts "
        f"rho_hi={rho[2]:.3f}")


def _theta_position1_kl(meth, prompts):
    """Mean position-1 KL(theta-write || unsteered) at the query final token."""
    kls = []
    v = meth.theta.to(meth.device)
    for p in prompts:
        enc = meth.tokenizer(p, return_tensors="pt")
        ids = enc["input_ids"].to(meth.device)
        with torch.no_grad():
            base = meth.model(ids).logits[0, -1]
        steer = meth._single_forward_last_token(ids, v)
        pp = torch.log_softmax(steer, dim=-1)
        qq = torch.log_softmax(base, dim=-1)
        kls.append((pp.exp() * (pp - qq)).sum().item())
    return float(np.mean(kls))


def _geometry(model, tok, theta, heldout_pairs, device):
    """cos(theta, W_U column) for the answer-span tokens (first token of each
    gold answer with prepend-space), summarized. Also cos to the mean answer
    W_U direction."""
    WU = model.get_output_embeddings().weight.detach().float().to("cpu")  # [vocab, hidden]
    th = theta.float().to("cpu")
    th_u = th / (th.norm() + 1e-12)
    ans_ids = []
    for (x, y) in heldout_pairs:
        aid = tok.encode(" " + y, add_special_tokens=False)
        if aid:
            ans_ids.append(aid[0])
    ans_ids = list(dict.fromkeys(ans_ids))  # unique preserve order
    cols = WU[ans_ids]  # [k, hidden]
    cols_u = cols / (cols.norm(dim=-1, keepdim=True) + 1e-12)
    cos_each = (cols_u @ th_u)  # [k]
    mean_dir = cols.mean(0)
    mean_dir_u = mean_dir / (mean_dir.norm() + 1e-12)
    cos_mean_dir = float((mean_dir_u @ th_u).item())
    # what token does theta most resemble (via W_U logit-lens argmax)?
    logits = WU @ th_u  # [vocab]
    top = torch.topk(logits, 10)
    top_tokens = [tok.decode([int(i)]) for i in top.indices]
    return {
        "n_answer_tokens": len(ans_ids),
        "cos_theta_answer_span_mean": float(cos_each.mean().item()),
        "cos_theta_answer_span_max": float(cos_each.max().item()),
        "cos_theta_answer_span_min": float(cos_each.min().item()),
        "cos_theta_mean_answer_dir": cos_mean_dir,
        "theta_logitlens_top10": top_tokens,
    }


def _fmt_ci(d):
    return f"{d['rate']*100:.1f}% [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"


def write_report(r, path):
    m = r["meta"]; v = r["verdict"]; s1 = r["stage1_sweep"]
    L = []
    A = L.append
    A(f"# Task-vectors arm — Hendel et al. 2310.15916 (steering-content-audit)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` ({m['n_layers']} layers, {m['n_heads']} heads, "
      f"resid {m['resid_dim']}), device `{m['device']}` ({m['dtype']}).  ")
    A(f"**Method:** in-context task vector = mean `resid_post[L]` at the last ICL "
      f"demo-context token over {m['n_mean']} {m['n_shots']}-shot contexts; "
      f"injection = **{m['chosen_op'].upper()}** at the zero-shot query's final "
      f"token (Hendel patches). Distinct from the A1 function-vectors arm (Todd "
      f"head-sum, ADD at L/3).  ")
    A(f"**Task:** {m['task']} (zero-shot eval).  ")
    A(f"**Prompts:** {m['n_eval_total']} held-out items (calib {m['n_calib']} / "
      f"eval {m['n_eval']}); {m['max_new_tokens']} new tokens greedy; "
      f"{m['n_boot']} bootstrap resamples.\n")

    A(f"## VERDICT: **{v['class']}**\n")
    A(f"Rule (plan §3, computed from CI bounds + cell_valid): "
      f"Genuine = `rho_hi <= 0.3` AND `E_native >= 3x floor`; "
      f"Dissolved = `rho_lo >= 0.9` AND cell valid; else Mixed. A fixed prompt-"
      f"independent output push cannot produce input-dependent antonyms.\n")
    sr = r["sanity_reproduction"]
    A(f"- **Reproduction (gate {sr['threshold_pts']:.0f} pts):** native "
      f"(E_first, {sr['op'].upper()}) gain = **{sr['raw_gain_pts']:+.1f} pts** "
      f"vs baseline; reproduced = **{sr['reproduced']}**. "
      f"{'Audit proceeds.' if sr['reproduced'] else 'Did NOT reproduce -> STOP.'}")
    A(f"- rho_hi = **{v['rho_hi']:.3f}** (need <= 0.30 for Genuine); "
      f"rho_lo = {v['rho_lo']:.3f}")
    A(f"- E_native >= 3x floor = **{v['effect_ge_3x_floor']}** "
      f"(cell valid = {v['cell_valid']})")
    A(f"- kappa native (E_first/E_native) = **{v['kappa_native']:.3f}**; "
      f"kappa diagnostic (E_first/E_all) = {v['kappa_diag_point']:.3f}")
    A(f"- passes Genuine = **{v['passes_genuine']}**; control cell void = "
      f"**{v['control_cell_void']}**"
      f"{' (Amendment-2 dose-response required)' if v['needs_amendment2_doseresponse'] else ''}\n")

    A(f"## Stage 1 — layer sweep (reproduction)\n")
    A(f"theta at each L extracted from {m['n_mean']} {m['n_shots']}-shot ICL "
      f"contexts (train-pool, disjoint from eval); native-regime (E_first single "
      f"write) accuracy on the {m['n_calib']}-prompt calib split. Reproduction "
      f"gate = gain >= {m['repro_gate_pts']:.0f} pts. Calib baseline acc = "
      f"{s1['calib_baseline_acc']*100:.1f}%.\n")
    A(f"**REPLACE (primary):**\n")
    A(f"| layer | theta norm | calib acc | gain vs baseline |")
    A(f"|------:|-----------:|----------:|-----------------:|")
    for row in s1["replace"]:
        A(f"| {row['layer']} | {row['theta_norm']:.2f} | "
          f"{row['calib_acc']*100:.1f}% | {row['gain_pts']:+.1f} pts |")
    A("")
    if s1["add"]:
        A(f"**ADD (sensitivity — REPLACE best {s1['best_replace_gain_pts']:+.1f} "
          f"pts < gate):**\n")
        A(f"| layer | theta norm | calib acc | gain vs baseline |")
        A(f"|------:|-----------:|----------:|-----------------:|")
        for row in s1["add"]:
            A(f"| {row['layer']} | {row['theta_norm']:.2f} | "
              f"{row['calib_acc']*100:.1f}% | {row['gain_pts']:+.1f} pts |")
        A("")
    A(f"Chosen: **op={m['chosen_op'].upper()}, L={m['chosen_layer']}** "
      f"(theta norm {m['theta_norm']:.2f}).\n")

    A(f"## Headline rates (eval split, {m['n_eval']} prompts)\n")
    A(f"Task accuracy (gold word in {m['max_new_tokens']}-token greedy "
      f"generation), bootstrap 95% CI.\n")
    rr = r["rates"]
    A(f"| condition | accuracy [95% CI] |")
    A(f"|---|---|")
    A(f"| baseline (zero-shot, no injection) | {_fmt_ci(rr['baseline'])} |")
    A(f"| **E_native = E_first** (KV-baked single write) | "
      f"{_fmt_ci(rr['E_native_first'])} |")
    A(f"| E_all (forced broadcast — diagnostic) | "
      f"{_fmt_ci(rr['E_all_diagnostic'])} |")
    A(f"| control (calibrated logit bias) | {_fmt_ci(rr['control'])} |")
    A(f"| floor (random dir @ matched norm, max of 3) | {_fmt_ci(rr['floor_max'])} |")
    A("")

    A(f"## Decomposition\n")
    e = r["effect"]; rho = r["rho"]; kd = r["kappa"]["diagnostic_first_over_all"]
    A(f"- **rho = E(control)/E_native** = {rho['point']:.3f} "
      f"[{rho['ci_lo']:.3f}, {rho['ci_hi']:.3f}] (Genuine needs rho_hi <= 0.30; "
      f"denominator = {rho['denominator']})")
    A(f"- **kappa native = E_first/E_native = {r['kappa']['native']:.3f}** "
      f"(single-position write, prompt-injection family — Amendment 3).")
    A(f"- kappa diagnostic (E_first/E_all) = {kd['point']:.3f} "
      f"[{kd['ci_lo']:.3f}, {kd['ci_hi']:.3f}] (broadcast comparison only).")
    A(f"- E_native = {e['E_native']*100:.1f} pts; floor effect = "
      f"{e['E_floor']*100:.1f} pts; effect/floor = {e['effect_over_floor']:.2f}x "
      f"(needs >= 3x).\n")

    cc = r["control_calibration"]
    A(f"## I2 control calibration (Amendment 1: TF per-step KL matched to "
      f"E_native)\n")
    A(f"- Token set S: {cc['token_set_size']} tokens (90% of ||position-1 "
      f"logit-delta||^2, cap 100). Top: {cc['top_tokens']}")
    A(f"- B* (theta, TF per-step KL) = {cc['B_star_target_kl']:.5f}; achieved "
      f"control TF-KL = {cc['achieved_kl']:.5f}; bias scalar = "
      f"{cc['bias_scalar']:.4f}.")
    sp = cc["sensitivity_position1"]
    A(f"- Sensitivity (position-1-matched, demoted): target KL "
      f"{sp['target_kl']:.5f}, achieved {sp['achieved_kl']:.5f}, c="
      f"{sp['bias_scalar']:.4f}.")
    mc = r["mechanism_check"]
    A(f"- First-token argmax flips vs baseline under theta (single write) = "
      f"{mc['first_token_flips']}/{mc['n_prompts']}.\n")

    g = r["geometry"]
    A(f"## Geometry — cos(theta, W_U answer-span)\n")
    A(f"- Answer-span tokens (unique first-token of gold answers): "
      f"{g['n_answer_tokens']}.")
    A(f"- cos(theta, W_U[answer]) mean = {g['cos_theta_answer_span_mean']:.4f} "
      f"(min {g['cos_theta_answer_span_min']:.4f}, max "
      f"{g['cos_theta_answer_span_max']:.4f}).")
    A(f"- cos(theta, mean answer W_U direction) = "
      f"{g['cos_theta_mean_answer_dir']:.4f}.")
    A(f"- theta logit-lens top-10 (W_U . theta_hat argmax): "
      f"{g['theta_logitlens_top10']}.\n")

    A(f"## Degeneracy gate (per eval condition)\n")
    er = r["eval_baseline_refs"]
    A(f"Eval baseline refs: rep={er['rep']:.3f}, median_len={er['median_len']:.1f}, "
      f"nll={er['nll']:.3f}. Gate: rep > 2x+0.1, or median_len < 0.5x, or nll > 3x.\n")
    A(f"| condition | tripped | rep | median_len | nll | reasons |")
    A(f"|---|:---:|---:|---:|---:|---|")
    for name, gg in r["degeneracy_gates"].items():
        A(f"| {name} | {'VOID' if gg['tripped'] else 'ok'} | {gg['rep']:.3f} | "
          f"{gg['median_len']:.1f} | {gg['nll']:.3f} | {'; '.join(gg['reasons'])} |")
    A("")
    A(f"## Floor runs\n")
    for fr in r["floor_runs"]:
        A(f"- seed {fr['seed']}: acc {fr['rate']*100:.1f}%")
    A("")
    A(f"## Sample generations (first 3 eval per condition)\n")
    for cond in ["baseline", "E_native_first", "E_all_diagnostic", "control"]:
        A(f"**{cond}:**")
        for (p, t, gld) in r["samples"][cond][:3]:
            A(f"- `{p}` -> `{t.strip()[:40]}` (gold: {gld})")
        A("")
    A(f"Runtime: {r['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
