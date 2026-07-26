"""ActAdd arm driver — Turner et al. 2308.10248 Activation Addition, wedding
topic, GPT-2-XL, under the frozen pre-registered battery (plan.md §2-§5, §8).

Faithful ActAdd:
  h_delta = resid_pre_L6(" weddings") - resid_pre_L6(" ")   (canonical demo)
  E_native = add c*h_delta at front prompt positions 0..pad_len-1, KV-baked
             (intrinsically one-shot / KV-persistent — the published form).
  E_all    = add the same c*h_delta at EVERY position (prompt + generated).
  kappa    = E_native / E_all  (ActAdd: native IS the first-only form; kappa
             measures whether sustained application adds anything beyond the
             native one-shot — a semantic INVERSION of additive-every-step
             methods, where kappa = E_first/E_all measures cascade share).

Conditions (150-prompt eval split, 64 tokens greedy):
  baseline, E_native, E_all, floor (random dir at matched norm, 3 seeds max),
  primary control (calibrated static logit bias, teacher-forced per-step KL
  budget of E_native; Amendment 1), W_U secondary control (report-only).

Verdict (amended plan §3): Dissolved = rho_lo >= 0.9 on a valid cell;
Genuine = rho_hi <= 0.3 with effect >= 3x floor (+ Amendment-2 dose-response if
control void); else Mixed with (kappa, rho) coordinates.

Usage:
  python run_actadd.py --model gpt2 --n 10 --tokens 24 --smoke   # smoke
  python run_actadd.py --model openai-community/gpt2-xl          # full arm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery as B  # noqa: E402


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def gen_condition(fn, prompts):
    texts = [fn(p) for p in prompts]
    hits = [int(B.wedding_topic_hit(t)) for t in texts]
    return texts, hits


def kv_baked_sanity_actadd(meth, tokenizer, prompts, coeff):
    """Verify E_native's front-position patch is baked into KV and absent from
    later single-token forwards. Re-derive E_native manually (prefill with the
    front-position injection, then continue with NO hook) and compare to the
    method's native output."""
    model, device, L = meth.model, meth.device, meth.layer
    rows = meth._delta_rows(coeff)
    results = []
    for p in prompts:
        native = meth.generate(p, coeff, "native")
        enc = tokenizer(p, return_tensors="pt")
        iid = enc["input_ids"].to(device)
        P = iid.shape[1]
        n_front = min(meth.pad_len, P)
        vecmap = {j: rows[j] for j in range(n_front)}
        with torch.no_grad(), B._actadd_static_hook(model, L, vecmap):
            out = model(iid, use_cache=True)
        past = out.past_key_values
        nid = int(out.logits[0, -1].argmax())
        gen = [nid]
        cur = torch.tensor([[nid]], device=device)
        with torch.no_grad():
            for _ in range(meth.max_new_tokens - 1):
                o = model(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nid = int(o.logits[0, -1].argmax())
                gen.append(nid)
                cur = torch.tensor([[nid]], device=device)
        manual = tokenizer.decode(torch.tensor(gen), skip_special_tokens=True)
        results.append(native.strip() == manual.strip())
    return {"n": len(prompts), "all_match": bool(all(results)),
            "matches": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--layer", type=int, default=6, help="ActAdd injection layer")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--coeff-grid", default="2,4,6,8,10,12,15,20")
    ap.add_argument("--fixed-coeff", type=float, default=None)
    # --- Method-reproduction sweep (layer x coeff x prompt-pair) ---
    ap.add_argument("--sweep-layers", default="6,10,16,20,24",
                    help="ActAdd injection layers to sweep")
    ap.add_argument("--sweep-coeffs", default="3,4,6,8,10,14",
                    help="coefficients to sweep")
    ap.add_argument("--repro-threshold", type=float, default=30.0,
                    help="min clean wedding-rate gain (pts) over baseline to pass")
    ap.add_argument("--fixed-pair", default=None,
                    help="skip sweep; force pair id (single|contrastive)")
    ap.add_argument("--fixed-layer", type=int, default=None,
                    help="skip sweep; force layer (with --fixed-coeff)")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = args.outdir or os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-06-actadd-arm")
    os.makedirs(outdir, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    expdir = os.path.dirname(os.path.abspath(__file__))

    t0 = time.time()
    log(f"ActAdd arm: model={args.model} device={args.device} n={args.n} "
        f"tokens={args.tokens} layer={args.layer} tag={tag}")

    # --- prompts: reuse frozen canonical set + splits from A0 ---
    pj = os.path.join(expdir, "prompts_neutral.json")
    with open(pj) as f:
        prompts = json.load(f)["prompts"]
    assert len(prompts) == 200
    calib_all, eval_all = B.split_prompts(prompts, args.n_calib)
    if args.n < 200:
        n_eval = args.n - min(args.n_calib, args.n // 4)
        n_cal = args.n - n_eval
        calib, eval_ = calib_all[:n_cal], eval_all[:n_eval]
    else:
        calib, eval_ = calib_all, eval_all
    log(f"calib={len(calib)} eval={len(eval_)}")

    # --- model ---
    model, tok = B.load_model(args.model, device=args.device)
    device = args.device
    H = B.get_model_info(model).hidden_size
    n_layers = B.get_model_info(model).n_layers
    log(f"model n_layers={n_layers} hidden={H}")

    # --- calibration baseline (gate refs + reproduction sweep) ---
    log("baseline generations on calibration...")
    cal_base_texts = [B.base_generate(model, tok, p, args.tokens, device)
                      for p in calib]
    cal_base_rate = float(np.mean([int(B.wedding_topic_hit(t))
                                   for t in cal_base_texts]))
    base_rep = float(np.mean([B.three_gram_rep_rate(t, tok)
                              for t in cal_base_texts]))
    base_median_len = B.median_len_tokens(cal_base_texts, tok)
    base_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                              for p, t in zip(calib, cal_base_texts)]))
    log(f"calib baseline: rate={cal_base_rate:.3f} rep={base_rep:.3f} "
        f"med_len={base_median_len:.1f} nll={base_nll:.3f}")

    # ==================================================================
    #  A. METHOD-REPRODUCTION SWEEP (anti-strawman gate; BEFORE battery)
    #  Two prompt-pairs x layers x coeffs on the calibration split.
    #  Pick the config with the highest CLEAN wedding rate that is
    #  >= baseline + repro_threshold pts. If none reaches it -> BLOCKED.
    #  Sweep cells cached to disk so a timeout resumes without recompute.
    # ==================================================================
    PAIRS = {
        "single": {"p_plus": " weddings", "p_minus": " ",
                   "desc": "single-token (p+=' weddings', p-=' ')"},
        "contrastive": {"p_plus": "I talk about weddings constantly",
                        "p_minus": "I do not talk about weddings constantly",
                        "desc": "canonical length-matched contrastive pair "
                                "(Turner et al. headline demo), front-aligned, "
                                "right-padded to equal token length, per-position "
                                "resid_pre diffs -> multi-position steering tensor"},
    }
    sweep_layers = [int(x) for x in args.sweep_layers.split(",")]
    sweep_coeffs = [float(x) for x in args.sweep_coeffs.split(",")]

    sweep_cache_path = os.path.join(outdir, f"sweep_{tag}.json")
    sweep_cells = {}
    if os.path.exists(sweep_cache_path):
        with open(sweep_cache_path) as f:
            sweep_cells = json.load(f).get("cells", {})
        log(f"resuming: {len(sweep_cells)} sweep cells cached")

    def _cell_key(pair_id, L, c):
        return f"{pair_id}|L{L}|c{c}"

    def _save_sweep():
        with open(sweep_cache_path, "w") as f:
            json.dump({"cells": sweep_cells,
                       "calib_base_rate": cal_base_rate}, f, indent=2)

    if args.fixed_pair is not None:
        # forced config: skip the sweep entirely
        forced_pair = args.fixed_pair
        forced_L = args.fixed_layer if args.fixed_layer is not None else args.layer
        forced_c = args.fixed_coeff
        assert forced_c is not None, "--fixed-pair requires --fixed-coeff"
        log(f"config FROZEN: pair={forced_pair} L={forced_L} c={forced_c}; "
            f"skipping sweep")
        chosen = {"pair_id": forced_pair, "layer": forced_L, "coeff": forced_c,
                  "note": "FROZEN via --fixed-pair/--fixed-layer/--fixed-coeff"}
        hdelta_info = {}
    else:
        for pair_id, pcfg in PAIRS.items():
            for L in sweep_layers:
                # build h_delta once per (pair, layer)
                h_delta_L, hinfo_L = B.build_actadd_hdelta(
                    model, tok, L, pcfg["p_plus"], pcfg["p_minus"],
                    device=device)
                meth_L = B.ActAddMethod(model, tok, L, h_delta_L, device=device,
                                        max_new_tokens=args.tokens)
                for c in sweep_coeffs:
                    key = _cell_key(pair_id, L, c)
                    if key in sweep_cells:
                        rec = sweep_cells[key]
                        log(f"  [cached] {key}: rate={rec['rate']:.3f} "
                            f"gain={rec['gain_pts']:+.1f} gate={rec['gate_tripped']}")
                        continue
                    texts = [meth_L.generate(p, c, "native") for p in calib]
                    hits = [int(B.wedding_topic_hit(t)) for t in texts]
                    rate = float(np.mean(hits))
                    gate = B.degeneracy_gate(texts, calib, base_rep,
                                             base_median_len, base_nll, model,
                                             tok, device=device)
                    gain = (rate - cal_base_rate) * 100
                    sweep_cells[key] = {
                        "pair_id": pair_id, "layer": L, "coeff": c,
                        "rate": rate, "gain_pts": gain,
                        "gate_tripped": bool(gate.tripped),
                        "gate_reasons": gate.reasons,
                        "rep": gate.rep_rate, "median_len": gate.median_len,
                        "nll": gate.mean_nll,
                        "pad_len": hinfo_L["pad_len"],
                        "hdelta_mean_norm": hinfo_L["mean_vector_norm"],
                    }
                    _save_sweep()
                    log(f"  {key}: rate={rate:.3f} gain={gain:+.1f}pts "
                        f"pad_len={hinfo_L['pad_len']} gate={gate.tripped} "
                        f"{gate.reasons}")

        # --- reproduction gate: highest CLEAN gain >= threshold ---
        clean = [v for v in sweep_cells.values()
                 if not v["gate_tripped"]
                 and v["gain_pts"] >= args.repro_threshold]
        if clean:
            best = max(clean, key=lambda v: (v["gain_pts"], -v["coeff"]))
            chosen = {"pair_id": best["pair_id"], "layer": best["layer"],
                      "coeff": best["coeff"],
                      "note": f"highest clean gain >= +{args.repro_threshold} pts "
                              f"({best['gain_pts']:+.1f} pts, pad_len="
                              f"{best['pad_len']})"}
            log(f"REPRODUCED: pair={chosen['pair_id']} L={chosen['layer']} "
                f"c={chosen['coeff']} gain={best['gain_pts']:+.1f}")
        else:
            chosen = None

    # sweep table (ordered) for the report
    sweep_table = sorted(
        sweep_cells.values(),
        key=lambda v: (v["pair_id"], v["layer"], v["coeff"]))

    result = {
        "meta": {
            "arm": "ActAdd (Turner et al. 2308.10248) wedding topic",
            "model": args.model, "device": device, "tag": tag,
            "n_total": len(prompts), "n_calib": len(calib),
            "n_eval": len(eval_), "max_new_tokens": args.tokens,
            "n_boot": args.n_boot, "hidden": H, "n_layers": n_layers,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "calib_baseline": {"rate": cal_base_rate, "rep": base_rep,
                           "median_len": base_median_len, "nll": base_nll},
        "reproduction_sweep": {
            "pairs": {k: v["desc"] for k, v in PAIRS.items()},
            "layers": sweep_layers, "coeffs": sweep_coeffs,
            "threshold_pts": args.repro_threshold,
            "table": sweep_table,
            "chosen": chosen,
        },
    }

    if chosen is None:
        log("ActAdd did NOT reproduce at any swept config (no clean cell "
            f">= +{args.repro_threshold} pts). STOPPING — reportable outcome.")
        result["verdict"] = {
            "class": "NOT-REPRODUCED",
            "reason": (f"No (pair,L,c) reached +{args.repro_threshold} clean "
                       f"wedding-rate gain on calibration; battery NOT run "
                       f"(auditing a dead vector is not informative)."),
        }
        _write(result, outdir, tag, args)
        log("DONE — verdict=NOT-REPRODUCED (battery skipped)")
        return

    # ==================================================================
    #  B. BATTERY at the reproduced config
    # ==================================================================
    pair_id = chosen["pair_id"]
    L = chosen["layer"]
    coeff = chosen["coeff"]
    pcfg = PAIRS[pair_id]
    h_delta, hinfo = B.build_actadd_hdelta(
        model, tok, L, pcfg["p_plus"], pcfg["p_minus"], device=device)
    meth = B.ActAddMethod(model, tok, L, h_delta, device=device,
                          max_new_tokens=args.tokens)
    log(f"BATTERY config: pair={pair_id} L={L} c={coeff} "
        f"pad_len={hinfo['pad_len']} per-pos norms="
        f"{[round(x,2) for x in hinfo['per_position_norms']]}")

    # --- KV-baked native sanity (2 prompts) at chosen config ---
    sanity = kv_baked_sanity_actadd(meth, tok, calib[:2], coeff=coeff)
    log(f"KV-baked native sanity: all_match={sanity['all_match']}")

    result["meta"]["layer"] = L
    result["method_fidelity"] = {
        "pair_id": pair_id, "pair_desc": pcfg["desc"],
        "p_plus": hinfo["p_plus"], "p_minus": hinfo["p_minus"],
        "ids_plus": hinfo["ids_plus"], "ids_minus": hinfo["ids_minus"],
        "pad_len": hinfo["pad_len"], "layer": L, "site": "resid_pre",
        "h_delta_per_position_norms": hinfo["per_position_norms"],
        "h_delta_mean_vector_norm": hinfo["mean_vector_norm"],
        "injection_native": "c*h_delta at front prompt positions "
                            "0..pad_len-1, KV-baked (one-shot/persistent)",
        "injection_all": "c*h_delta broadcast at every position "
                         "(prompt + each generated token)",
    }
    result["kv_baked_sanity"] = sanity
    result["chosen_config"] = chosen

    # --- EVAL: all conditions ---
    log("EVAL: baseline...")
    base_texts, base_hits = gen_condition(
        lambda p: B.base_generate(model, tok, p, args.tokens, device), eval_)
    log("EVAL: E_native (ActAdd published, KV-baked one-shot)...")
    nat_texts, nat_hits = gen_condition(
        lambda p: meth.generate(p, coeff, "native"), eval_)
    log("EVAL: E_all (all-positions variant)...")
    all_texts, all_hits = gen_condition(
        lambda p: meth.generate(p, coeff, "all"), eval_)

    # --- floor: random direction at matched norm c*||h_delta|| at native
    # site/positions, 3 seeds, max. For pad_len>1 the random vector is injected
    # (broadcast) at every front position at the mean per-position magnitude. ---
    log("EVAL: floor (random dir, matched norm, 3 seeds)...")
    per_pos_norms = [float(h_delta[i].norm()) for i in range(h_delta.shape[0])]
    hd_norm = float(np.mean(per_pos_norms))  # mean per-position norm
    matched_norm = coeff * hd_norm
    floor_runs = []
    for s in range(3):
        g = torch.Generator().manual_seed(2000 + s)
        rv = torch.randn(H, generator=g)
        rv = rv / rv.norm() * matched_norm
        ftexts = [meth.generate_with_fixed_vector(p, rv, "native") for p in eval_]
        fhits = [int(B.wedding_topic_hit(t)) for t in ftexts]
        fr = float(np.mean(fhits))
        floor_runs.append({"seed": 2000 + s, "rate": fr, "hits": fhits,
                           "texts": ftexts})
        log(f"  floor seed {2000+s}: rate={fr:.3f}")
    floor_max = max(floor_runs, key=lambda r: r["rate"])

    # --- I2 primary control: token set from E_native position-1 logit deltas,
    # budget = mean teacher-forced per-step KL of E_native (Amendment 1). ---
    log("I2 control: token-set discovery (E_native pos-1 logit delta)...")
    mean_delta = B.actadd_position1_logit_delta(meth, tok, calib, coeff)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    top_ctrl_tokens = [tok.decode([i]) for i in ctrl_token_ids[:15]]
    log(f"  control token set size = {len(ctrl_token_ids)}; top {top_ctrl_tokens}")

    log("I2 control: unsteered continuations on calib (for TF-KL)...")
    calib_cont_ids = [B.base_generate_ids(model, tok, p, args.tokens, device)
                      for p in calib]
    log("I2 control: B* = mean teacher-forced per-step KL (E_native)...")
    target_kl = B.actadd_teacher_forced_stepkl_native(
        meth, tok, calib, calib_cont_ids, coeff)
    log(f"  B* (E_native TF per-step KL) = {target_kl:.5f}")
    c_scalar, achieved_kl = B.calibrate_bias_scalar_stepkl(
        model, tok, calib, calib_cont_ids, ctrl_token_ids, mean_delta,
        target_kl, device=device)
    log(f"  bias scalar c={c_scalar:.4f} achieved TF-KL={achieved_kl:.5f}")
    tid_t = torch.tensor(ctrl_token_ids)
    bias_vals = c_scalar * mean_delta[tid_t]
    processor = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)

    log("EVAL: control (calibrated logit bias)...")
    ctrl_texts, ctrl_hits = gen_condition(
        lambda p: B.control_generate(model, tok, p, processor, args.tokens,
                                     device), eval_)

    # --- W_U secondary control (report-only): project h_delta onto span(W_U
    # wedding-lexicon) pulled back naively; add at native site/positions/norm ---
    log("W_U secondary control (report-only): project h_delta onto W_U span...")
    wu_ids, wu_kept = B.resolve_a0_token_ids(tok)  # single-token wedding set
    wu_basis = B.wu_wedding_span_basis(model, wu_ids)  # [k, hidden]
    hd_vec = h_delta.float().mean(0)  # mean h_delta over positions -> [hidden]
    # QR on MPS may fall back to CPU; align devices for the projection matmul.
    _proj_dev = wu_basis.device
    hd_proj = B.project_onto(
        hd_vec.to(_proj_dev).unsqueeze(0), wu_basis).squeeze(0).to(hd_vec.device)
    # add at native site/positions with matched per-position norm to c*h_delta
    proj_norm = float(hd_proj.norm())
    cos_hdelta_wu = float(torch.nn.functional.cosine_similarity(
        hd_vec.unsqueeze(0), hd_proj.unsqueeze(0)).item()) if proj_norm > 0 else 0.0
    if proj_norm > 0:
        wu_vec = hd_proj / proj_norm * matched_norm
        wu_texts, wu_hits = gen_condition(
            lambda p: meth.generate_with_fixed_vector(p, wu_vec, "native"), eval_)
        wu_rate = float(np.mean(wu_hits))
    else:
        wu_texts, wu_hits, wu_rate = [], [], 0.0
    log(f"  cos(h_delta, W_U-wedding-span proj) = {cos_hdelta_wu:.4f}; "
        f"W_U-control rate={wu_rate:.3f}")

    # --- mechanism: first-token argmax flips under E_native (eval) ---
    log("Mechanism: first-token argmax flips (E_native, eval)...")
    n_flips, n_flip_prompts = B.actadd_first_token_flip_count(
        meth, tok, eval_, coeff)
    log(f"  first-token flips = {n_flips}/{n_flip_prompts}")

    # --- rates + bootstrap CIs ---
    def rate_ci(hits):
        return B.bootstrap_rate_ci(hits, args.n_boot, seed=7)

    r_base = rate_ci(base_hits)
    r_nat = rate_ci(nat_hits)
    r_all = rate_ci(all_hits)
    r_ctrl = rate_ci(ctrl_hits)
    r_floor = rate_ci(floor_max["hits"])
    r_wu = rate_ci(wu_hits) if wu_hits else (wu_rate, float("nan"), float("nan"))

    # kappa = E_native / E_all (ActAdd semantic inversion); rho = E(control)/E(native)
    # NOTE: for ActAdd the "steering" reference for rho is the PUBLISHED method =
    # E_native. rho = E(control) / E(native).
    kappa = B.bootstrap_ratio_ci(nat_hits, all_hits, base_hits, args.n_boot, seed=11)
    rho = B.bootstrap_ratio_ci(ctrl_hits, nat_hits, base_hits, args.n_boot, seed=13)

    # --- degeneracy gate per eval condition ---
    log("degeneracy gate (eval conditions)...")
    ev_base_rep = float(np.mean([B.three_gram_rep_rate(t, tok) for t in base_texts]))
    ev_base_med = B.median_len_tokens(base_texts, tok)
    ev_base_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                                 for p, t in zip(eval_, base_texts)]))
    gates = {}
    gate_conditions = {
        "E_native": (nat_texts, eval_),
        "E_all": (all_texts, eval_),
        "control": (ctrl_texts, eval_),
        "floor_max": (floor_max["texts"], eval_),
    }
    if wu_texts:
        gate_conditions["wu_secondary"] = (wu_texts, eval_)
    for name, (texts, pr) in gate_conditions.items():
        gg = B.degeneracy_gate(texts, pr, ev_base_rep, ev_base_med, ev_base_nll,
                               model, tok, device=device)
        gates[name] = {"tripped": gg.tripped, "rep": gg.rep_rate,
                       "median_len": gg.median_len, "nll": gg.mean_nll,
                       "reasons": gg.reasons}
        log(f"  {name}: tripped={gg.tripped} rep={gg.rep_rate:.3f} "
            f"med={gg.median_len:.1f} nll={gg.mean_nll:.3f}")

    # --- effect vs floor (E_native must be >= 3x floor to be meaningful) ---
    e_native = r_nat[0] - r_base[0]
    e_all = r_all[0] - r_base[0]
    e_floor = r_floor[0] - r_base[0]
    effect_over_floor = e_native / max(e_floor, 1e-9)

    # --- Amendment 2: if control trips the gate, run dose-response ---
    dose = None
    control_tripped = gates["control"]["tripped"]
    if control_tripped:
        log("Amendment 2: control tripped gate -> dose-response (>=3 sub-degen scales)")
        dose = run_dose_response(
            model, tok, calib, eval_, ctrl_token_ids, mean_delta, c_scalar,
            base_rep, base_median_len, base_nll, ev_base_rep, ev_base_med,
            ev_base_nll, e_native, r_base[0], args, device)

    # --- verdict (amended plan §3) ---
    rho_lo = rho[1]
    rho_hi = rho[2]
    kappa_lo = kappa[1]
    kappa_hi = kappa[2]
    effect_ge_3x_floor = e_native >= 3 * e_floor
    gate_clean_native = not gates["E_native"]["tripped"]
    gate_clean_control = not gates["control"]["tripped"]
    cell_valid = effect_ge_3x_floor and gate_clean_native and gate_clean_control

    # Dissolved requires a clean control cell (a degenerate control can never
    # reproduce a behavior). Genuine may survive a void control IF the
    # Amendment-2 dose-response passes (control <= 0.3*E at every clean scale).
    dose_ok = None
    if dose is not None:
        dose_ok = dose["passes_amendment2"]

    dissolved = (rho_lo >= 0.9) and cell_valid
    if control_tripped:
        # control void: Dissolved impossible; Genuine needs dose-response pass.
        genuine = (dose_ok is True) and effect_ge_3x_floor
    else:
        genuine = (rho_hi <= 0.3) and effect_ge_3x_floor
    if dissolved:
        verdict = "Dissolved"
    elif genuine:
        verdict = "Genuine"
    else:
        verdict = "Mixed"

    result.update({
        "chosen_coeff": coeff,
        "eval_baseline_refs": {"rep": ev_base_rep, "median_len": ev_base_med,
                               "nll": ev_base_nll},
        "rates": {
            "baseline": {"rate": r_base[0], "ci_lo": r_base[1], "ci_hi": r_base[2]},
            "E_native": {"rate": r_nat[0], "ci_lo": r_nat[1], "ci_hi": r_nat[2]},
            "E_all": {"rate": r_all[0], "ci_lo": r_all[1], "ci_hi": r_all[2]},
            "control": {"rate": r_ctrl[0], "ci_lo": r_ctrl[1], "ci_hi": r_ctrl[2]},
            "floor_max": {"rate": r_floor[0], "ci_lo": r_floor[1], "ci_hi": r_floor[2]},
            "wu_secondary": {"rate": r_wu[0], "ci_lo": r_wu[1], "ci_hi": r_wu[2]},
        },
        "kappa": {"point": kappa[0], "ci_lo": kappa[1], "ci_hi": kappa[2],
                  "note": "kappa = E_native / E_all. ActAdd SEMANTIC INVERSION: "
                          "native IS the first-only form; kappa here measures "
                          "whether SUSTAINED all-position application adds "
                          "anything beyond the native one-shot (kappa~1 => "
                          "native already captures the full effect; kappa<1 => "
                          "sustained application strengthens it)."},
        "rho": {"point": rho[0], "ci_lo": rho[1], "ci_hi": rho[2],
                "note": "rho = E(control) / E(E_native), the published method."},
        "effect": {"E_native": e_native, "E_all": e_all, "E_floor": e_floor,
                   "effect_over_floor": effect_over_floor},
        "control_calibration": {
            "token_set_size": len(ctrl_token_ids), "token_ids": ctrl_token_ids,
            "top_tokens": top_ctrl_tokens,
            "budget": "mean_teacher_forced_per_step_KL_of_E_native (Amendment 1)",
            "B_star_target_kl": target_kl, "achieved_kl": achieved_kl,
            "bias_scalar": c_scalar,
        },
        "wu_secondary_control": {
            "kept_tokens": wu_kept,
            "cos_hdelta_wu_span": cos_hdelta_wu,
            "proj_norm": proj_norm, "matched_norm": matched_norm,
            "rate": wu_rate,
            "note": "report-only descriptive secondary, not gating (raw W_U "
                    "columns as resid directions, naive pullback, no lens)",
        },
        "geometry": {
            "cos_hdelta_mean_wu_wedding_span": cos_hdelta_wu,
            "h_delta_norm": hd_norm, "matched_floor_norm": matched_norm,
        },
        "mechanism_check": {"first_token_flips": n_flips,
                            "n_prompts": n_flip_prompts,
                            "note": "E_native first-token argmax flips vs baseline"},
        "floor_runs": [{"seed": r["seed"], "rate": r["rate"]} for r in floor_runs],
        "degeneracy_gates": gates,
        "dose_response": dose,
        "verdict": {
            "class": verdict,
            "rho_lo": rho_lo, "rho_hi": rho_hi,
            "kappa_lo": kappa_lo, "kappa_hi": kappa_hi,
            "cell_valid": bool(cell_valid),
            "effect_ge_3x_floor": bool(effect_ge_3x_floor),
            "gate_clean_E_native": bool(gate_clean_native),
            "gate_clean_control": bool(gate_clean_control),
            "control_tripped": bool(control_tripped),
            "dose_response_passes_amendment2": dose_ok,
            "dissolved_rule": "rho_lo>=0.9 on valid cell",
            "genuine_rule": ("rho_hi<=0.3 & effect>=3x floor (control clean); "
                             "OR effect>=3x floor & Amendment-2 dose-response "
                             "passes (control void)"),
        },
        "runtime_sec": time.time() - t0,
    })

    _write(result, outdir, tag, args)
    log(f"DONE in {result['runtime_sec']:.0f}s  verdict={verdict}")


def run_dose_response(model, tok, calib, eval_, ctrl_token_ids, mean_delta,
                      c_full, base_rep, base_med, base_nll, ev_rep, ev_med,
                      ev_nll, e_native, base_rate, args, device):
    """Amendment 2: evaluate the primary control at >=3 sub-degenerate scales
    (fractions of the matched scalar). Passes if effect <= 0.3*E(native) at every
    non-degenerate scale."""
    scales = [0.25, 0.5, 0.75]
    tid_t = torch.tensor(ctrl_token_ids)
    rows = []
    for frac in scales:
        c = frac * c_full
        bias_vals = c * mean_delta[tid_t]
        proc = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)
        texts = [B.control_generate(model, tok, p, proc, args.tokens, device)
                 for p in eval_]
        hits = [int(B.wedding_topic_hit(t)) for t in texts]
        rate = float(np.mean(hits))
        gate = B.degeneracy_gate(texts, eval_, ev_rep, ev_med, ev_nll, model,
                                 tok, device=device)
        eff = rate - base_rate
        ratio = eff / max(e_native, 1e-9)
        rows.append({"frac": frac, "bias_scalar": c, "rate": rate,
                     "effect": eff, "effect_over_native": ratio,
                     "gate_tripped": gate.tripped, "gate_reasons": gate.reasons})
        log(f"  dose frac={frac} c={c:.3f}: rate={rate:.3f} "
            f"eff/native={ratio:.3f} gate={gate.tripped}")
    clean = [r for r in rows if not r["gate_tripped"]]
    passes = (len(clean) >= 3) and all(r["effect_over_native"] <= 0.3
                                       for r in clean)
    return {"scales": scales, "rows": rows, "n_clean": len(clean),
            "passes_amendment2": bool(passes),
            "rule": ">=3 sub-degenerate scales, effect<=0.3*E(native) at each"}


def _write(result, outdir, tag, args):
    jpath = os.path.join(outdir, f"results_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote {jpath}")
    if not args.smoke:
        # canonical name for the full run
        fpath = os.path.join(outdir, "results_full.json")
        with open(fpath, "w") as f:
            json.dump(result, f, indent=2)
        rpath = os.path.join(outdir, "report.md")
        write_report(result, rpath)
        log(f"wrote {rpath}")


def _fmt_ci(d):
    if d.get("ci_lo") != d.get("ci_lo"):  # nan
        return f"{d['rate']*100:.1f}% [n/a]"
    return f"{d['rate']*100:.1f}% [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"


def _append_repro_section(A, r):
    """Emit the Method-reproduction section: full layer x coeff x pair sweep
    table on calibration + the chosen config (or the miss)."""
    rs = r["reproduction_sweep"]
    A(f"## Method reproduction (anti-strawman gate)\n")
    A(f"Two prompt-pairs x layers {rs['layers']} x coeffs {rs['coeffs']} on the "
      f"{r['meta']['n_calib']}-prompt calibration split (E_native, "
      f"{r['meta']['max_new_tokens']} tok greedy). Reproduction gate: highest "
      f"CLEAN (gate-not-tripped) wedding rate with gain >= "
      f"+{rs['threshold_pts']} pts over baseline. Calib baseline wedding rate = "
      f"{r['calib_baseline']['rate']*100:.1f}%.\n")
    for pid, desc in rs["pairs"].items():
        A(f"- **{pid}**: {desc}")
    A("")
    ch = rs.get("chosen")
    if ch:
        A(f"**Chosen config:** pair=`{ch['pair_id']}`, L={ch['layer']}, "
          f"c={ch['coeff']} — {ch['note']}.\n")
    else:
        A(f"**No config reproduced** (no clean cell reached the threshold).\n")
    A(f"| pair | L | c | pad_len | rate | gain (pts) | gate | rep | med_len | nll |")
    A(f"|---|--:|--:|--:|-----:|-----------:|:----:|----:|--------:|----:|")
    for t in rs["table"]:
        star = ""
        if ch and t["pair_id"] == ch["pair_id"] and t["layer"] == ch["layer"] \
                and t["coeff"] == ch["coeff"]:
            star = " **<-**"
        A(f"| {t['pair_id']} | {t['layer']} | {t['coeff']} | {t['pad_len']} | "
          f"{t['rate']*100:.1f}% | {t['gain_pts']:+.1f}{star} | "
          f"{'YES' if t['gate_tripped'] else 'no'} | {t['rep']:.3f} | "
          f"{t['median_len']:.1f} | {t['nll']:.3f} |")
    A("")


def write_report(r, path):
    m = r["meta"]
    lines = []
    A = lines.append
    A(f"# ActAdd arm — Activation Addition wedding steering (steering-content-audit)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Method:** ActAdd (Turner et al., arXiv 2308.10248), wedding topic.  ")
    _lyr = m.get("layer", "swept")
    A(f"**Model:** `{m['model']}` ({m['n_layers']} layers, hidden {m['hidden']}), "
      f"ActAdd injection layer {_lyr} (resid_pre), device `{m['device']}`.  ")
    A(f"**Prompts:** {m['n_total']} neutral (calib {m['n_calib']} / eval "
      f"{m['n_eval']}), {m['max_new_tokens']} new tokens greedy, "
      f"{m['n_boot']} bootstrap resamples. Prompt set/splits reused from A0.\n")

    if r.get("verdict", {}).get("class") == "NOT-REPRODUCED":
        A(f"## VERDICT: ActAdd did NOT reproduce in our harness at the swept "
          f"configs\n")
        A(f"{r['verdict']['reason']}\n")
        _append_repro_section(A, r)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return

    v = r["verdict"]
    mf = r["method_fidelity"]
    A(f"## VERDICT: **{v['class']}**\n")
    A(f"Pre-registered prediction (plan §8): **Dissolved or Mixed** (high kappa, "
      f"high rho). Amended rules (plan §3, §11): Dissolved = rho_lo >= 0.9 on a "
      f"valid cell; Genuine = rho_hi <= 0.3 with effect >= 3x floor "
      f"(+ Amendment-2 dose-response if control void); else Mixed.\n")
    A(f"- rho = E(control)/E(E_native) = **{r['rho']['point']:.3f}** "
      f"[{r['rho']['ci_lo']:.3f}, {r['rho']['ci_hi']:.3f}]  "
      f"(rho_lo={v['rho_lo']:.3f}, rho_hi={v['rho_hi']:.3f})")
    A(f"- kappa = E_native/E_all = **{r['kappa']['point']:.3f}** "
      f"[{r['kappa']['ci_lo']:.3f}, {r['kappa']['ci_hi']:.3f}] "
      f"(coordinate; ActAdd semantic inversion — see note)")
    A(f"- cell valid = **{v['cell_valid']}** (effect>=3x floor: "
      f"{v['effect_ge_3x_floor']}; gate clean E_native: "
      f"{v['gate_clean_E_native']}; gate clean control: "
      f"{v['gate_clean_control']})")
    if v.get("control_tripped"):
        A(f"- control tripped degeneracy gate; Amendment-2 dose-response "
          f"passes = **{v.get('dose_response_passes_amendment2')}**")
    A("")

    # --- Method reproduction section (sweep table + chosen config) ---
    _append_repro_section(A, r)

    A(f"## Method fidelity\n")
    A(f"- **Prompt-pair:** `{mf['pair_id']}` — {mf['pair_desc']}.")
    A(f"- **h_delta** = resid_pre(L{mf['layer']})(`{mf['p_plus']!r}`) - "
      f"resid_pre(L{mf['layer']})(`{mf['p_minus']!r}`), front-aligned, "
      f"right-padded to equal token length (pad_len={mf['pad_len']}).")
    A(f"  - p+ token ids {mf['ids_plus']}, p- token ids {mf['ids_minus']}.")
    A(f"  - h_delta per-position norms = "
      f"{[round(x,3) for x in mf['h_delta_per_position_norms']]}; "
      f"mean-vector norm {mf['h_delta_mean_vector_norm']:.3f}.")
    A(f"- **E_native** (published): {mf['injection_native']}.")
    A(f"- **E_all** (all-positions variant): {mf['injection_all']}.")
    A(f"- **KV-baked native sanity:** E_native reproduced by manual "
      f"prefill-then-continue (all_match={r['kv_baked_sanity']['all_match']}).")
    A(f"- **Chosen config:** pair=`{r['chosen_config']['pair_id']}`, "
      f"L={r['chosen_config']['layer']}, c={r['chosen_config']['coeff']} "
      f"({r['chosen_config']['note']}).\n")

    A(f"## Headline rates (eval split, {m['n_eval']} prompts)\n")
    A(f"Wedding-topic rate, bootstrap 95% CI in brackets.\n")
    rr = r["rates"]
    A(f"| condition | rate [95% CI] |")
    A(f"|---|---|")
    A(f"| baseline (unsteered) | {_fmt_ci(rr['baseline'])} |")
    A(f"| E_native (ActAdd published, KV-baked one-shot) | {_fmt_ci(rr['E_native'])} |")
    A(f"| E_all (all-positions variant) | {_fmt_ci(rr['E_all'])} |")
    A(f"| control (calibrated logit bias) | {_fmt_ci(rr['control'])} |")
    A(f"| floor (random dir matched norm, max of 3) | {_fmt_ci(rr['floor_max'])} |")
    A(f"| W_U secondary control (report-only) | {_fmt_ci(rr['wu_secondary'])} |")
    A("")

    A(f"## Decomposition\n")
    k = r["kappa"]; rho = r["rho"]; e = r["effect"]
    A(f"- **kappa = E_native / E_all** = {k['point']:.3f} "
      f"[{k['ci_lo']:.3f}, {k['ci_hi']:.3f}]")
    A(f"  - {k['note']}")
    A(f"- **rho = E(control) / E(E_native)** = {rho['point']:.3f} "
      f"[{rho['ci_lo']:.3f}, {rho['ci_hi']:.3f}]")
    A(f"- effect E_native = {e['E_native']*100:.1f} pts; E_all = "
      f"{e['E_all']*100:.1f} pts; floor = {e['E_floor']*100:.1f} pts; "
      f"E_native / floor = {e['effect_over_floor']:.2f}x (needs >= 3x).\n")

    cc = r["control_calibration"]
    A(f"## I2 primary control (Amendment 1: teacher-forced per-step KL of E_native)\n")
    A(f"- Token set S: {cc['token_set_size']} tokens (90% of ||E_native "
      f"position-1 logit-delta||^2, cap 100). Top: {cc['top_tokens']}")
    A(f"- Budget = mean teacher-forced per-step KL over all {m['max_new_tokens']} "
      f"continuation positions x calib prompts, on fixed unsteered continuations.")
    A(f"- B* (E_native TF per-step KL) = {cc['B_star_target_kl']:.5f}, achieved "
      f"control TF per-step KL = {cc['achieved_kl']:.5f}, bias scalar = "
      f"{cc['bias_scalar']:.4f}.\n")

    if r.get("dose_response"):
        d = r["dose_response"]
        A(f"## Amendment 2 dose-response (control tripped gate)\n")
        A(f"Rule: {d['rule']}. Passes = **{d['passes_amendment2']}** "
          f"({d['n_clean']} clean scales).\n")
        A(f"| frac | bias scalar | rate | effect/native | gate |")
        A(f"|-----:|------------:|-----:|--------------:|:----:|")
        for row in d["rows"]:
            A(f"| {row['frac']} | {row['bias_scalar']:.3f} | "
              f"{row['rate']*100:.1f}% | {row['effect_over_native']:.3f} | "
              f"{'VOID' if row['gate_tripped'] else 'ok'} |")
        A("")

    wu = r["wu_secondary_control"]
    g = r["geometry"]
    A(f"## Geometry + W_U secondary control (descriptive)\n")
    A(f"- **cos(h_delta, W_U-wedding-span projection)** = "
      f"{g['cos_hdelta_mean_wu_wedding_span']:.4f} (how much of h_delta already "
      f"lies in the naive unembedding span of the wedding lexicon).")
    A(f"- W_U secondary control (h_delta projected onto span(W_U[wedding]), "
      f"re-normed to c*||h_delta||={g['matched_floor_norm']:.1f}, injected at "
      f"native site/positions): rate {wu['rate']*100:.1f}%. Report-only, not "
      f"gating. Kept tokens: {wu['kept_tokens']}.\n")

    A(f"## Mechanism check\n")
    mc = r["mechanism_check"]
    A(f"- First-token argmax flips vs baseline under E_native = "
      f"**{mc['first_token_flips']}/{mc['n_prompts']}**.\n")

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
        A(f"- seed {fr['seed']}: rate {fr['rate']*100:.1f}%")
    A("")
    A(f"Runtime: {r['runtime_sec']:.0f}s.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
