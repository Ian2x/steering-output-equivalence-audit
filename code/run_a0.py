"""A0 anchor driver — synthetic output-push, must classify Dissolved.

Runs the full battery end-to-end (plan §6):
  1. Build/load neutral prompts (200), split 50 calib / 150 eval.
  2. Build the A0 push direction from W_U columns of the wedding token set.
  3. Fix the additive scale from calibration resid norms.
  4. Select alpha on calibration: smallest alpha giving >= +30 pts wedding rate
     over baseline WITHOUT tripping the degeneracy gate. Freeze it.
  5. Eval split: baseline, E_all (steering), E_first, floor, calibrated
     logit-bias control. Compute kappa, rho, floor, all rates + bootstrap CIs,
     degeneracy gate per condition.
  6. Emit results JSON + report.md with the A0 verdict.

Usage:
  python run_a0.py --model gpt2 --n 20 --tokens 32 --smoke   # smoke test
  python run_a0.py --model openai-community/gpt2-xl          # full A0
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


def kv_baked_sanity(meth: "B.A0Method", tokenizer, prompts, alpha: float) -> dict:
    """Verify E_first's patch is truly absent after the first generated token.

    Re-derives E_first independently: prefill with the additive patch applied
    only at the last prompt position (baked into KV), then continue with NO hook.
    If this equals actlib's positions='last_prompt' output, the one-shot patch is
    correctly absent thereafter.
    """
    from actlib.patching import _dynamic_patch_hook
    model = meth.model
    device = meth.device
    H = meth.u.shape[0]
    L = meth.layer
    addv = meth.add_vector(alpha)
    rb = torch.zeros(0, H, device=device)
    results = []
    for p in prompts:
        lib = meth.generate_with_vector(p, addv, "last_prompt")
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
            for _ in range(meth.max_new_tokens - 1):
                o = model(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nid = int(o.logits[0, -1].argmax())
                gen.append(nid)
                cur = torch.tensor([[nid]], device=device)
        manual = tokenizer.decode(torch.tensor(gen), skip_special_tokens=True)
        results.append(lib.strip() == manual.strip())
    return {"n": len(prompts), "all_match": bool(all(results)),
            "matches": results}


def generate_condition(fn, prompts):
    """Apply a generation fn over prompts, return (texts, hit_indicators)."""
    texts = [fn(p) for p in prompts]
    hits = [int(B.wedding_topic_hit(t)) for t in texts]
    return texts, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200, help="total prompts")
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true",
                    help="smoke mode: fewer prompts, log extra")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--alpha-grid",
                    default="0.02,0.04,0.06,0.08,0.10,0.12,0.15,0.20,0.30,0.50")
    ap.add_argument("--fixed-alpha", type=float, default=None,
                    help="skip alpha selection and use this frozen alpha "
                         "(Amendment 1 rerun: alpha=0.04 is frozen)")
    ap.add_argument("--reuse-floor-json", default=None,
                    help="path to run-1 results_full.json to reuse floor rate "
                         "(Amendment 1: do NOT re-run floor)")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = args.outdir or os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-06-a0-anchor-r2")
    os.makedirs(outdir, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    expdir = os.path.dirname(os.path.abspath(__file__))

    t0 = time.time()
    log(f"model={args.model} device={args.device} n={args.n} tokens={args.tokens} tag={tag}")

    # --- prompts ---
    prompts = B.build_neutral_prompts(200)
    # persist the canonical 200 (only on full run to avoid clobbering)
    pj = os.path.join(expdir, "prompts_neutral.json")
    if not args.smoke or not os.path.exists(pj):
        with open(pj, "w") as f:
            json.dump({"seed": 20260705, "n": 200, "prompts": prompts}, f, indent=2)
    lj = os.path.join(expdir, "lexicon_wedding.json")
    with open(lj, "w") as f:
        json.dump({"lexicon": B.WEDDING_LEXICON,
                   "a0_token_strings": B.A0_TOKEN_STRINGS}, f, indent=2)

    calib_all, eval_all = B.split_prompts(prompts, args.n_calib)
    # subsample for smoke / reduced n
    if args.n < 200:
        n_eval = args.n - min(args.n_calib, args.n // 4)
        n_cal = args.n - n_eval
        calib = calib_all[:n_cal]
        eval_ = eval_all[:n_eval]
    else:
        calib, eval_ = calib_all, eval_all
    log(f"calib={len(calib)} eval={len(eval_)}")

    # --- model + direction ---
    model, tok = B.load_model(args.model, device=args.device)
    device = args.device
    L = B.final_layer(model)
    H = B.get_model_info(model).hidden_size
    token_ids, kept = B.resolve_a0_token_ids(tok)
    log(f"A0 tokens kept ({len(kept)}): {kept}")
    u = B.a0_direction(model, token_ids).to(device)
    scale = B.calib_resid_norm(model, tok, calib, L, device=device)
    log(f"final layer L={L} hidden={H} resid-norm scale={scale:.1f}")

    meth = B.A0Method(model, tok, L, u, scale, device=device,
                      max_new_tokens=args.tokens)

    # --- KV-baked one-shot sanity (2 prompts) ---
    sanity = kv_baked_sanity(meth, tok, calib[:2], alpha=3.0)
    log(f"KV-baked one-shot sanity: all_match={sanity['all_match']}")

    # --- baseline on calibration (for gate references + alpha selection) ---
    log("baseline generations on calibration...")
    cal_base_texts = [B.base_generate(model, tok, p, args.tokens, device)
                      for p in calib]
    cal_base_hits = [int(B.wedding_topic_hit(t)) for t in cal_base_texts]
    cal_base_rate = float(np.mean(cal_base_hits))
    base_rep = float(np.mean([B.three_gram_rep_rate(t, tok)
                              for t in cal_base_texts]))
    base_median_len = B.median_len_tokens(cal_base_texts, tok)
    base_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                              for p, t in zip(calib, cal_base_texts)]))
    log(f"calib baseline: rate={cal_base_rate:.3f} rep={base_rep:.3f} "
        f"med_len={base_median_len:.1f} nll={base_nll:.3f}")

    # --- alpha selection on calibration ---
    # Amendment 1 freezes alpha = 0.04 (run-1 selection). With --fixed-alpha we
    # skip the sweep entirely (saves ~half the calibration cost) and record the
    # frozen value; otherwise run the original selection.
    grid = [float(x) for x in args.alpha_grid.split(",")]
    alpha_trace = []
    chosen_alpha = None
    if args.fixed_alpha is not None:
        chosen_alpha = args.fixed_alpha
        log(f"alpha FROZEN (Amendment 1) = {chosen_alpha}; skipping sweep")
    else:
        for a in grid:
            texts = [meth.generate(p, a, "all") for p in calib]
            hits = [int(B.wedding_topic_hit(t)) for t in texts]
            rate = float(np.mean(hits))
            gate = B.degeneracy_gate(texts, calib, base_rep, base_median_len,
                                     base_nll, model, tok, device=device)
            gain = (rate - cal_base_rate) * 100
            rec = {"alpha": a, "rate": rate, "gain_pts": gain,
                   "gate_tripped": gate.tripped, "gate_reasons": gate.reasons,
                   "rep": gate.rep_rate, "median_len": gate.median_len,
                   "nll": gate.mean_nll}
            alpha_trace.append(rec)
            log(f"  alpha={a}: rate={rate:.3f} gain={gain:+.1f}pts "
                f"gate_tripped={gate.tripped} {gate.reasons}")
            if chosen_alpha is None and gain >= 30.0 and not gate.tripped:
                chosen_alpha = a
        log(f"chosen alpha = {chosen_alpha}")

    result = {
        "meta": {
            "model": args.model, "device": device, "tag": tag,
            "n_total": len(prompts), "n_calib": len(calib),
            "n_eval": len(eval_), "max_new_tokens": args.tokens,
            "n_boot": args.n_boot, "layer": L, "hidden": H,
            "resid_norm_scale": scale,
            "a0_tokens_kept": kept, "a0_token_ids": token_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "kv_baked_sanity": sanity,
        "calib_baseline": {
            "rate": cal_base_rate, "rep": base_rep,
            "median_len": base_median_len, "nll": base_nll,
        },
        "alpha_selection": {"grid": grid, "trace": alpha_trace,
                            "chosen_alpha": chosen_alpha,
                            "frozen": args.fixed_alpha is not None,
                            "criterion": (
                                "FROZEN at Amendment-1 value (run-1 selection)"
                                if args.fixed_alpha is not None
                                else ">=+30pts over baseline AND gate not tripped")},
    }

    if chosen_alpha is None:
        log("NO alpha satisfied the selection criterion; recording and stopping "
            "eval (A0 blocked at alpha selection).")
        result["verdict"] = {"class": "BLOCKED",
                             "reason": "no alpha met +30pts w/o degeneracy"}
        _write(result, outdir, tag, repo, model, tok, args)
        return

    alpha = chosen_alpha

    # --- EVAL split: all conditions ---
    log("EVAL: baseline...")
    base_texts, base_hits = generate_condition(
        lambda p: B.base_generate(model, tok, p, args.tokens, device), eval_)
    log("EVAL: E_all (steering)...")
    all_texts, all_hits = generate_condition(
        lambda p: meth.generate(p, alpha, "all"), eval_)
    log("EVAL: E_first...")
    first_texts, first_hits = generate_condition(
        lambda p: meth.generate(p, alpha, "first"), eval_)

    # --- floor: REUSED from run 1 (Amendment 1 §4: do NOT re-run floor) ---
    floor_reused = False
    if args.reuse_floor_json and os.path.exists(args.reuse_floor_json):
        with open(args.reuse_floor_json) as f:
            _r1 = json.load(f)
        floor_runs = _r1["floor_runs"]  # [{seed, rate}, ...]
        floor_max_r1 = max(floor_runs, key=lambda r: r["rate"])
        _n = _r1["meta"]["n_eval"]
        _n_hit = int(round(floor_max_r1["rate"] * _n))
        # reconstruct a per-prompt hit vector consistent with the reused rate
        # (only its mean is load-bearing for the E_all >= 3x floor validity gate;
        # per-prompt identity was not persisted in run 1)
        fhits = [1] * _n_hit + [0] * (_n - _n_hit)
        floor_max = {"seed": floor_max_r1["seed"], "rate": floor_max_r1["rate"],
                     "hits": fhits, "texts": None}
        floor_reused = True
        log(f"EVAL: floor REUSED from run 1 ({args.reuse_floor_json}); "
            f"floor_max rate={floor_max['rate']:.4f} (seed "
            f"{floor_max['seed']}, {_n_hit}/{_n} hits)")
    else:
        log("EVAL: floor (random directions, 3 seeds)...")
        floor_runs = []
        for s in range(3):
            g = torch.Generator().manual_seed(1000 + s)
            rv = torch.randn(H, generator=g)
            rv = rv / rv.norm()
            addv = (alpha * scale) * rv.to(device)
            ftexts = [meth.generate_with_vector(p, addv, "all_generated")
                      for p in eval_]
            fhits = [int(B.wedding_topic_hit(t)) for t in ftexts]
            fr = float(np.mean(fhits))
            floor_runs.append({"seed": 1000 + s, "rate": fr,
                               "hits": fhits, "texts": ftexts})
            log(f"  floor seed {1000+s}: rate={fr:.3f}")
        floor_max = max(floor_runs, key=lambda r: r["rate"])

    # --- I2 control: token-set discovery + KL-matched bias, on calibration ---
    # AMENDMENT 1 (§2): control budget = mean TEACHER-FORCED per-step KL over all
    # 64 continuation positions x 50 calib prompts, on the fixed UNSTEERED
    # continuations. Token set S is recomputed identically to run 1 (position-1
    # logit-delta regression, 90% coverage cap 100).
    log("I2 control: token-set discovery (calibration, identical to run 1)...")
    mean_delta = B.position1_logit_delta(meth, tok, calib, alpha)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    log(f"  control token set size = {len(ctrl_token_ids)}")
    top_ctrl_tokens = [tok.decode([i]) for i in ctrl_token_ids[:15]]
    log(f"  top control tokens: {top_ctrl_tokens}")

    log("I2 control: generating unsteered continuations on calib (for TF-KL)...")
    calib_cont_ids = [B.base_generate_ids(model, tok, p, args.tokens, device)
                      for p in calib]

    log("I2 control: computing B* = mean teacher-forced per-step KL (steered)...")
    target_kl = B.teacher_forced_stepkl_steered(
        meth, tok, calib, calib_cont_ids, alpha)
    log(f"  B* (mean TF per-step KL, steered||unsteered) = {target_kl:.5f}")

    c_scalar, achieved_kl = B.calibrate_bias_scalar_stepkl(
        model, tok, calib, calib_cont_ids, ctrl_token_ids, mean_delta,
        target_kl, device=device)
    log(f"  bias scalar c={c_scalar:.4f} achieved TF-KL={achieved_kl:.5f}")
    tid_t = torch.tensor(ctrl_token_ids)
    bias_vals = c_scalar * mean_delta[tid_t]
    processor = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)

    # Sensitivity: position-1-matched control (run-1 method), demoted per Amend 1.
    p1_target_kl = B._position1_kl_steered(meth, tok, calib, alpha)
    p1_c_scalar, p1_achieved_kl = B.calibrate_bias_scalar(
        model, tok, calib, ctrl_token_ids, mean_delta, p1_target_kl,
        device=device)
    log(f"  [sensitivity] position-1-matched: target={p1_target_kl:.5f} "
        f"c={p1_c_scalar:.4f} achieved={p1_achieved_kl:.5f}")

    # --- Amendment 1 §3: first-token argmax flip count on eval (expected ~0) ---
    log("Mechanism check: first-token argmax flips vs baseline (eval)...")
    n_flips, n_flip_prompts = B.first_token_flip_count(meth, tok, eval_, alpha)
    log(f"  first-token flips = {n_flips}/{n_flip_prompts}")

    log("EVAL: control (calibrated logit bias)...")
    ctrl_texts, ctrl_hits = generate_condition(
        lambda p: B.control_generate(model, tok, p, processor, args.tokens,
                                     device), eval_)

    # --- rates + bootstrap CIs ---
    def rate_ci(hits):
        return B.bootstrap_rate_ci(hits, args.n_boot, seed=7)

    r_base = rate_ci(base_hits)
    r_all = rate_ci(all_hits)
    r_first = rate_ci(first_hits)
    r_ctrl = rate_ci(ctrl_hits)
    r_floor = rate_ci(floor_max["hits"])

    # kappa = E_first / E_all ; rho = E(control) / E(steering)
    kappa = B.bootstrap_ratio_ci(first_hits, all_hits, base_hits,
                                 args.n_boot, seed=11)
    rho = B.bootstrap_ratio_ci(ctrl_hits, all_hits, base_hits,
                               args.n_boot, seed=13)

    # --- degeneracy gate per eval condition ---
    log("degeneracy gate (eval conditions)...")
    # baseline references recomputed on eval baseline
    ev_base_rep = float(np.mean([B.three_gram_rep_rate(t, tok)
                                 for t in base_texts]))
    ev_base_med = B.median_len_tokens(base_texts, tok)
    ev_base_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                                 for p, t in zip(eval_, base_texts)]))
    gates = {}
    gate_conditions = {
        "E_all": (all_texts, eval_),
        "E_first": (first_texts, eval_),
        "control": (ctrl_texts, eval_),
    }
    if not floor_reused:
        gate_conditions["floor_max"] = (floor_max["texts"], eval_)
    for name, (texts, prompts_used) in gate_conditions.items():
        g = B.degeneracy_gate(texts, prompts_used, ev_base_rep, ev_base_med,
                              ev_base_nll, model, tok, device=device)
        gates[name] = {"tripped": g.tripped, "rep": g.rep_rate,
                       "median_len": g.median_len, "nll": g.mean_nll,
                       "reasons": g.reasons}
        log(f"  {name}: tripped={g.tripped} rep={g.rep_rate:.3f} "
            f"med={g.median_len:.1f} nll={g.mean_nll:.3f}")
    if floor_reused:
        gates["floor_max"] = {"tripped": False, "rep": None, "median_len": None,
                              "nll": None,
                              "reasons": ["REUSED from run 1 (clean there)"]}

    # --- effect vs floor (steering must be >= 3x floor to be meaningful) ---
    e_all = r_all[0] - r_base[0]
    e_floor = r_floor[0] - r_base[0]
    effect_over_floor = e_all / max(e_floor, 1e-9)

    # --- verdict (AMENDMENT 1 §2) ---
    # Dissolved = rho_lo >= 0.9 AND cell valid. Cell valid = E_all >= 3x floor
    # AND degeneracy gate clean on E_all AND control. kappa is a reported
    # coordinate, NOT a gate. Genuine definition unchanged.
    kappa_lo_clip = max(0.0, min(1.0, kappa[1]))
    rho_lo = rho[1]
    kappa_lo = kappa[1]
    effect_ge_3x_floor = e_all >= 3 * e_floor
    gate_clean_eall = not gates["E_all"]["tripped"]
    gate_clean_control = not gates["control"]["tripped"]
    cell_valid = effect_ge_3x_floor and gate_clean_eall and gate_clean_control
    dissolved = (rho_lo >= 0.9) and cell_valid
    genuine = (rho[2] <= 0.3) and (e_all >= 3 * e_floor)
    if dissolved:
        verdict = "Dissolved"
    elif genuine:
        verdict = "Genuine"
    else:
        verdict = "Mixed"

    result.update({
        "chosen_alpha": alpha,
        "eval_baseline_refs": {"rep": ev_base_rep, "median_len": ev_base_med,
                               "nll": ev_base_nll},
        "rates": {
            "baseline": {"rate": r_base[0], "ci_lo": r_base[1], "ci_hi": r_base[2]},
            "E_all_steering": {"rate": r_all[0], "ci_lo": r_all[1], "ci_hi": r_all[2]},
            "E_first": {"rate": r_first[0], "ci_lo": r_first[1], "ci_hi": r_first[2]},
            "control": {"rate": r_ctrl[0], "ci_lo": r_ctrl[1], "ci_hi": r_ctrl[2]},
            "floor_max": {"rate": r_floor[0], "ci_lo": r_floor[1], "ci_hi": r_floor[2]},
        },
        "kappa": {"point": kappa[0], "ci_lo": kappa[1], "ci_hi": kappa[2],
                  "ci_lo_clipped": kappa_lo_clip},
        "rho": {"point": rho[0], "ci_lo": rho[1], "ci_hi": rho[2]},
        "effect": {"E_all": e_all, "E_floor": e_floor,
                   "effect_over_floor": effect_over_floor},
        "control_calibration": {
            "token_set_size": len(ctrl_token_ids),
            "token_ids": ctrl_token_ids,
            "top_tokens": top_ctrl_tokens,
            "budget": "mean_teacher_forced_per_step_KL (Amendment 1)",
            "B_star_target_kl": target_kl, "achieved_kl": achieved_kl,
            "bias_scalar": c_scalar,
            "sensitivity_position1": {
                "target_kl": p1_target_kl, "achieved_kl": p1_achieved_kl,
                "bias_scalar": p1_c_scalar,
                "note": "position-1-matched control, demoted to sensitivity "
                        "per Amendment 1 (run-1 primary method)",
            },
        },
        "mechanism_check": {
            "first_token_flips": n_flips,
            "n_prompts": n_flip_prompts,
            "note": "steering push (alpha, position-1) argmax-flip count vs "
                    "baseline on eval; expected ~0 (Amendment 1 §3)",
        },
        "floor_runs": [{"seed": r["seed"], "rate": r["rate"]} for r in floor_runs],
        "floor_reused_from_run1": floor_reused,
        "degeneracy_gates": gates,
        "verdict": {
            "class": verdict,
            "amendment": "Amendment 1 (2026-07-06)",
            "dissolved_criterion": "rho_lo>=0.9 AND cell_valid "
                                   "(E_all>=3x floor AND gate clean on "
                                   "E_all+control); kappa is a coordinate, "
                                   "NOT a gate",
            "rho_lo": rho_lo, "kappa_lo": kappa_lo,
            "cell_valid": bool(cell_valid),
            "effect_ge_3x_floor": bool(effect_ge_3x_floor),
            "gate_clean_E_all": bool(gate_clean_eall),
            "gate_clean_control": bool(gate_clean_control),
            "passes_dissolved": bool(dissolved),
        },
        "runtime_sec": time.time() - t0,
    })

    _write(result, outdir, tag, repo, model, tok, args)
    log(f"DONE in {result['runtime_sec']:.0f}s  verdict={verdict}")


def _write(result, outdir, tag, repo, model, tok, args):
    jpath = os.path.join(outdir, f"results_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2)
    log(f"wrote {jpath}")
    if not args.smoke:
        rpath = os.path.join(outdir, "report.md")
        write_report(result, rpath)
        log(f"wrote {rpath}")


def _fmt_ci(d):
    return f"{d['rate']*100:.1f}% [{d['ci_lo']*100:.1f}, {d['ci_hi']*100:.1f}]"


def write_report(r, path):
    m = r["meta"]
    lines = []
    A = lines.append
    A(f"# A0 anchor — synthetic output-push (steering-content-audit)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` (layer {m['layer']}, hidden {m['hidden']}), "
      f"device `{m['device']}`  ")
    A(f"**Prompts:** {m['n_total']} neutral (calib {m['n_calib']} / eval "
      f"{m['n_eval']}), {m['max_new_tokens']} new tokens greedy, "
      f"{m['n_boot']} bootstrap resamples.\n")

    if r.get("verdict", {}).get("class") == "BLOCKED":
        A(f"## VERDICT: BLOCKED\n")
        A(f"{r['verdict']['reason']}\n")

    v = r["verdict"]
    A(f"## VERDICT: **{v['class']}**\n")
    A(f"A0 is Dissolved by construction (a pure output push). Under the "
      f"**amended** rule (plan §11 Amendment 1, §3): Dissolved = "
      f"`rho_lo >= 0.9` on a **valid cell** (E_all >= 3x floor AND degeneracy "
      f"gate clean on E_all and control). kappa is a reported coordinate, "
      f"NOT a gate.\n")
    A(f"- rho_lo = **{v['rho_lo']:.3f}** (need >= 0.90)")
    A(f"- cell valid = **{v['cell_valid']}** "
      f"(E_all>=3x floor: {v['effect_ge_3x_floor']}; "
      f"gate clean E_all: {v['gate_clean_E_all']}; "
      f"gate clean control: {v['gate_clean_control']})")
    A(f"- kappa_lo = **{v['kappa_lo']:.3f}** (reported coordinate, not gating)")
    A(f"- passes Dissolved: **{v['passes_dissolved']}**\n")

    A(f"## Amendment 1 applied (2026-07-06, debugging round 1)\n")
    mc = r.get("mechanism_check", {})
    cc0 = r["control_calibration"]
    A(f"1. **Verdict:** kappa removed from Dissolved definition; Dissolved = "
      f"rho_lo>=0.9 on a valid cell (see above). kappa reported as coordinate.")
    A(f"2. **Control budget = mean teacher-forced per-step KL** (all 64 "
      f"continuation positions x calib prompts, on fixed unsteered "
      f"continuations), replacing position-1 KL matching.")
    A(f"   - **B\\*** (steering method's mean TF per-step KL) = "
      f"**{cc0['B_star_target_kl']:.5f}**")
    A(f"   - achieved control TF per-step KL = **{cc0['achieved_kl']:.5f}** "
      f"(bias scalar c = {cc0['bias_scalar']:.4f})")
    sp = cc0.get("sensitivity_position1", {})
    A(f"   - sensitivity (position-1-matched, run-1 method): target KL "
      f"{sp.get('target_kl', float('nan')):.5f}, achieved "
      f"{sp.get('achieved_kl', float('nan')):.5f}, c={sp.get('bias_scalar', float('nan')):.4f}")
    A(f"3. **Mechanism check:** first-token argmax flips vs baseline under the "
      f"steering push = **{mc.get('first_token_flips')}/{mc.get('n_prompts')}** "
      f"(expected ~0, confirming the kappa=0 mechanism).")
    A(f"4. Frozen (unchanged): alpha={r['chosen_alpha']}, prompt sets/splits, "
      f"lexicon, token-set discovery, greedy 64 tokens, bootstrap "
      f"{m['n_boot']}, degeneracy thresholds, E_all/E_first/floor defs. "
      f"Floor {'REUSED from run 1' if r.get('floor_reused_from_run1') else 're-run'}.\n")

    A(f"## Config / method\n")
    A(f"- **A0 direction:** L2-normalised sum of W_U columns for tokens "
      f"{m['a0_tokens_kept']} (single-token; multi-token entries dropped).")
    A(f"- **Intervention:** additive push `alpha * u * scale` into the "
      f"final-layer (L={m['layer']}) resid_post, realised via actlib "
      f"`generate_with_patch` (subspace_transplant w/ empty remove-basis == "
      f"additive add_vector).")
    A(f"- **Scale** (fixed): median final-position resid_post norm on "
      f"calibration = {m['resid_norm_scale']:.1f}.")
    A(f"- **Chosen alpha:** {r['chosen_alpha']} "
      f"(smallest alpha with >= +30 pts over baseline w/o tripping the gate).")
    A(f"- **KV-baked one-shot sanity:** E_first patch verified absent after "
      f"the first generated token (all_match="
      f"{r['kv_baked_sanity']['all_match']}).\n")

    A(f"## Alpha selection trace (calibration)\n")
    A(f"Calib baseline wedding rate = {r['calib_baseline']['rate']*100:.1f}%.\n")
    A(f"| alpha | rate | gain (pts) | gate tripped | rep | med_len | nll |")
    A(f"|------:|-----:|-----------:|:------------:|----:|--------:|----:|")
    for t in r["alpha_selection"]["trace"]:
        A(f"| {t['alpha']} | {t['rate']*100:.1f}% | {t['gain_pts']:+.1f} | "
          f"{'YES' if t['gate_tripped'] else 'no'} | {t['rep']:.3f} | "
          f"{t['median_len']:.1f} | {t['nll']:.3f} |")
    A("")

    A(f"## Headline rates (eval split, {m['n_eval']} prompts)\n")
    A(f"Wedding-topic rate, bootstrap 95% CI in brackets.\n")
    rr = r["rates"]
    A(f"| condition | rate [95% CI] |")
    A(f"|---|---|")
    A(f"| baseline (unsteered) | {_fmt_ci(rr['baseline'])} |")
    A(f"| E_all (steering, native) | {_fmt_ci(rr['E_all_steering'])} |")
    A(f"| E_first (KV-baked one-shot) | {_fmt_ci(rr['E_first'])} |")
    A(f"| control (calibrated logit bias) | {_fmt_ci(rr['control'])} |")
    A(f"| floor (random dir, max of 3) | {_fmt_ci(rr['floor_max'])} |")
    A("")

    A(f"## Decomposition\n")
    k = r["kappa"]; rho = r["rho"]; e = r["effect"]
    A(f"- **kappa = E_first / E_all** = {k['point']:.3f} "
      f"[{k['ci_lo']:.3f}, {k['ci_hi']:.3f}]  (CI-lo clipped to [0,1]: "
      f"{k['ci_lo_clipped']:.3f})")
    A(f"- **rho = E(control) / E(steering)** = {rho['point']:.3f} "
      f"[{rho['ci_lo']:.3f}, {rho['ci_hi']:.3f}]")
    A(f"- effect E_all = {e['E_all']*100:.1f} pts; floor effect = "
      f"{e['E_floor']*100:.1f} pts; steering / floor = "
      f"{e['effect_over_floor']:.2f}x (needs >= 3x to be a meaningful effect).\n")

    cc = r["control_calibration"]
    A(f"## I2 control calibration (Amendment 1: teacher-forced per-step KL)\n")
    A(f"- Token set S: {cc['token_set_size']} tokens (90% of ||position-1 "
      f"logit-delta||^2, cap 100; recomputed identically to run 1). "
      f"Top: {cc['top_tokens']}")
    A(f"- Budget = **mean teacher-forced per-step KL** over all 64 continuation "
      f"positions x calib prompts, on fixed unsteered continuations.")
    A(f"- B* (steering, TF per-step KL) = {cc['B_star_target_kl']:.5f}, "
      f"achieved control TF per-step KL = {cc['achieved_kl']:.5f}, "
      f"bias scalar = {cc['bias_scalar']:.4f}.")
    sp = cc.get("sensitivity_position1", {})
    A(f"- Sensitivity (position-1-matched control, demoted per Amend 1): "
      f"target KL {sp.get('target_kl', float('nan')):.5f}, achieved "
      f"{sp.get('achieved_kl', float('nan')):.5f}, bias scalar "
      f"{sp.get('bias_scalar', float('nan')):.4f} "
      f"(run-1 reported c=2.4169 at KL=0.0432).\n")

    A(f"## Degeneracy gate (per eval condition)\n")
    er = r["eval_baseline_refs"]
    A(f"Eval baseline refs: rep={er['rep']:.3f}, median_len="
      f"{er['median_len']:.1f}, nll={er['nll']:.3f}. "
      f"Gate: rep > 2x+0.1, or median_len < 0.5x, or nll > 3x.\n")
    A(f"| condition | tripped | rep | median_len | nll | reasons |")
    A(f"|---|:---:|---:|---:|---:|---|")
    def _fmt(x, spec):
        return "n/a" if x is None else format(x, spec)
    for name, g in r["degeneracy_gates"].items():
        A(f"| {name} | {'VOID' if g['tripped'] else 'ok'} | "
          f"{_fmt(g['rep'], '.3f')} | {_fmt(g['median_len'], '.1f')} | "
          f"{_fmt(g['nll'], '.3f')} | {'; '.join(g['reasons'])} |")
    A("")
    A(f"## Floor runs\n")
    if r.get("floor_reused_from_run1"):
        A(f"REUSED from run 1 (Amendment 1 §4: floor not re-run).")
    for fr in r["floor_runs"]:
        A(f"- seed {fr['seed']}: rate {fr['rate']*100:.1f}%")
    A("")
    A(f"Runtime: {r['runtime_sec']:.0f}s.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
