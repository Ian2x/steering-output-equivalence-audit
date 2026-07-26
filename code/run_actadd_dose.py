"""ActAdd arm — control-budget DOSE curve for the OUTPUT-PUSH pole (plan §11).

The ActAdd run returned **Mixed** with a HIGH rho (E(control)/E(E_native) point
0.959, CI [0.853, 1.071]): the matched-budget calibrated logit-bias control
(built WITHOUT the steering vector, on the run's frozen token set) reproduces
almost all of the +80.7-pt wedding-topic effect. A referee kill for the whole
rho-audit is "rho is not identified — it depends on the arbitrary budget-matching
functional." For the GENUINE pole that is answered by the Amendment-2 dose sweeps
(no admissible budget reproduces the effect). This driver is the MIRROR for the
OUTPUT-PUSH pole: show the high rho is NOT a knife-edge at one tuned budget — that
rho stays high across a BAND of control budgets and does not collapse into Genuine
territory (<= 0.3) below the matched budget.

RE-EVALUATES ONLY THE CONTROL CONDITION at bias scalars c = frac x matched_c on
the SAME 150 eval prompts, SAME 64-token greedy, SAME wedding-topic metric, SAME
degeneracy gate (SAME frozen eval baseline refs), bootstrap 10k. NO model-side
intervention (no steering-vector-side push in the control) — calibrated logit bias
on the run's frozen token set only, exactly as the run's control cell.

Reconstructs the run deterministically: same model (gpt2-xl, MPS), same prompt set
+ split (B.split_prompts, n_calib=50), same chosen config (pair=contrastive, L=20,
coeff=8.0), h_delta rebuilt from the frozen contrastive pair, control token set +
mean position-1 logit-delta recomputed on the SAME calib split and VERIFIED against
the saved run token set (like the taskvec template's `ids_match`). The matched
scalar (1.4505...) is loaded frozen from the run (NOT re-calibrated), so frac 1.00
reproduces the run's control cell exactly (rho ~ 0.959).

Adjudication is the OUTPUT-PUSH-pole read: report the rho(budget) CURVE + its
stability characterizations (min/max clean rho, stays_out_of_genuine, sub-matched
rho). This is NOT the Genuine-pole PASS/FAIL. It does NOT re-adjudicate
Dissolved/Mixed/Genuine.

Nothing else re-run. E_native and baseline are fixed run references. Output:
dose_response_budget.md + dose_response_budget.json + a resumable
dose_budget_stage.json in the actadd-arm dir (DISTINCT `_budget` names so they
never collide with any Genuine-pole dose_response.*). Does not touch
report.md/results_full.json/sweep_full.json.

Usage:
  cd .../exp && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_actadd_dose.py
  ... --smoke --n-eval 20
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import battery as B          # noqa: E402

# Fractions of the run-matched control scalar. Brackets the matched budget BELOW
# (sub-matched regime: does rho stay high?) and ABOVE (degeneracy onset). frac
# 1.00 == the run's control cell (the determinism anchor: must reproduce rho~0.96).
DOSE_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# The Genuine boundary (plan §3): a method whose control reproduces <= 0.3 x
# E_native is Genuine. For the OUTPUT-PUSH pole we characterize whether rho stays
# ABOVE this line across the clean band (i.e. does NOT collapse into Genuine).
GENUINE_RHO = 0.3

# Frozen verdict-rule text kept for reference only (headline is the curve).
FROZEN_VERDICT_RULE = (
    "plan §3/§11: Dissolved = rho_lo >= 0.9 on a valid cell; Genuine = rho_hi <= "
    "0.3 with effect >= 3x floor (+Amendment-2 if control void); else Mixed. The "
    "ActAdd run was Mixed (rho point 0.959, CI [0.853, 1.071]). This driver does "
    "NOT re-adjudicate that verdict — it reports the rho(budget) curve.")

# Determinism tolerance: frac 1.00 rho must land within this of the run's point.
RHO_MATCH_TOL = 0.05


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_stage(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"grid": {}}


def save_stage(path, stage):
    with open(path, "w") as f:
        json.dump(stage, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2-xl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260705,
                    help="prompt-split seed (B.split_prompts default; the ActAdd "
                         "run used the frozen prompts_neutral.json + this split)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = args.outdir or os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-06-actadd-arm")
    stage_path = os.path.join(outdir, "dose_budget_stage.json")
    stage = load_stage(stage_path)
    t0 = time.time()

    # --- load the run result (fixed references + determinism checks) ---
    run = json.load(open(os.path.join(outdir, "results_full.json")))
    chosen_L = run["meta"]["layer"]                      # 20
    coeff = run["chosen_coeff"]                          # 8.0
    pair_id = run["method_fidelity"]["pair_id"]          # contrastive
    p_plus = run["method_fidelity"]["p_plus"]
    p_minus = run["method_fidelity"]["p_minus"]
    matched_c = run["control_calibration"]["bias_scalar"]  # 1.4505... (FROZEN)
    run_token_ids = run["control_calibration"]["token_ids"]
    E_native_frac = run["effect"]["E_native"]            # 0.8066...
    E_native_pts = E_native_frac * 100.0
    baseline_rate = run["rates"]["baseline"]["rate"]     # 0.0
    run_control_rate = run["rates"]["control"]["rate"]   # 0.7733...
    run_rho_point = run["rho"]["point"]                  # 0.9587
    run_rho_lo = run["rho"]["ci_lo"]
    run_rho_hi = run["rho"]["ci_hi"]
    ev_rep = run["eval_baseline_refs"]["rep"]
    ev_med = run["eval_baseline_refs"]["median_len"]
    ev_nll = run["eval_baseline_refs"]["nll"]
    log(f"DOSE (output-push pole): arm=ActAdd pair={pair_id} L={chosen_L} "
        f"coeff={coeff} matched_c={matched_c:.4f} E_native={E_native_pts:.1f}pts "
        f"baseline={baseline_rate:.3f} run_rho={run_rho_point:.3f} fracs={DOSE_FRACS}")

    # --- model (same as run: gpt2-xl on mps) ---
    model, tok = B.load_model(args.model, device=args.device)
    device = args.device

    # --- prompt set + split (identical to the run) ---
    pj = os.path.join(_HERE, "prompts_neutral.json")
    with open(pj) as f:
        prompts = json.load(f)["prompts"]
    assert len(prompts) == 200, f"expected 200 frozen prompts, got {len(prompts)}"
    calib_all, eval_all = B.split_prompts(prompts, args.n_calib, seed=args.seed)
    # The full ActAdd run used n=200 -> calib = calib_all (50), eval = eval_all (150).
    calib = calib_all
    eval_ = eval_all
    if args.smoke:
        eval_ = eval_[:args.n_eval]
    log(f"split: calib {len(calib)} / eval {len(eval_)} "
        f"(smoke={args.smoke}, n_eval cap={args.n_eval if args.smoke else 'full'})")

    # --- rebuild h_delta from the frozen contrastive pair + method (byte-faithful) ---
    h_delta, hinfo = B.build_actadd_hdelta(model, tok, chosen_L, p_plus, p_minus,
                                           device=device)
    meth = B.ActAddMethod(model, tok, chosen_L, h_delta, device=device,
                          max_new_tokens=args.tokens)
    log(f"h_delta pad_len={hinfo['pad_len']} mean_vec_norm="
        f"{hinfo['mean_vector_norm']:.3f} "
        f"(run mean_vec_norm={run['method_fidelity']['h_delta_mean_vector_norm']:.3f})")

    # --- reconstruct control token set + mean position-1 logit-delta on calib ---
    # (native front-position injection, exactly as the run's control discovery)
    log("reconstructing control token set + mean position-1 logit-delta on calib...")
    mean_delta = B.actadd_position1_logit_delta(meth, tok, calib, coeff)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    ids_match = (ctrl_token_ids == run_token_ids)
    log(f"token set size={len(ctrl_token_ids)} matches run set: {ids_match}")
    if not ids_match and not args.smoke:
        # Hard determinism failure: STOP and report (do NOT fudge). Smoke uses a
        # truncated eval only (calib is still full), so the token set should still
        # match; if it doesn't in a full run the reconstruction diverged.
        log("FATAL: reconstructed control token set does NOT match the run set.")
        log(f"  recon[:12]={ctrl_token_ids[:12]}")
        log(f"  run  [:12]={run_token_ids[:12]}")
        raise SystemExit(
            "Determinism check A failed (token set mismatch). Refusing to emit a "
            "dose curve on a divergent reconstruction (project hard rule).")
    tid_t = torch.tensor(ctrl_token_ids)

    # --- dose sweep, control-only ---
    def run_frac(frac):
        key = f"{frac:.2f}"
        if key in stage["grid"]:
            log(f"  frac {frac:.2f}: cached")
            return stage["grid"][key]
        c = frac * matched_c
        bias_vals = c * mean_delta[tid_t]
        processor = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)
        texts = [B.control_generate(model, tok, p, processor, args.tokens, device)
                 for p in eval_]
        hits = [int(B.wedding_topic_hit(t)) for t in texts]
        r = B.bootstrap_rate_ci(hits, args.n_boot, seed=7)
        gg = B.degeneracy_gate(texts, eval_, ev_rep, ev_med, ev_nll, model, tok,
                               device=device)
        gain_pts = (r[0] - baseline_rate) * 100
        rho = gain_pts / max(E_native_pts, 1e-9)
        row = {
            "frac": frac, "bias_scalar": c, "accuracy": r[0],
            "ci_lo": r[1], "ci_hi": r[2], "gain_pts": gain_pts,
            "rho": rho, "effect_ratio_vs_E_native": rho,
            "gate_tripped": bool(gg.tripped), "gate_rep": gg.rep_rate,
            "gate_median_len": gg.median_len, "gate_nll": gg.mean_nll,
            "gate_reasons": gg.reasons,
            "sample": texts[0][:80] if texts else "",
        }
        stage["grid"][key] = row
        save_stage(stage_path, stage)
        log(f"  frac {frac:.2f} (c={c:.4f}): acc={r[0]*100:.1f}% "
            f"gain={gain_pts:+.1f}pts rho={rho:.3f} tripped={gg.tripped} "
            f"rep={gg.rep_rate:.3f}")
        return row

    log("running dose grid (control only)...")
    rows = [run_frac(f) for f in DOSE_FRACS]

    # --- determinism check B: frac 1.00 must reproduce the run's rho ---
    row1 = next(r for r in rows if abs(r["frac"] - 1.00) < 1e-9)
    rho_at_1 = row1["rho"]
    det_b_delta = abs(rho_at_1 - run_rho_point)
    det_b_ok = det_b_delta <= RHO_MATCH_TOL
    log(f"determinism B: frac 1.00 rho={rho_at_1:.3f} vs run rho={run_rho_point:.3f} "
        f"(|delta|={det_b_delta:.3f}, tol {RHO_MATCH_TOL}) -> {'OK' if det_b_ok else 'MISMATCH'}")
    if not det_b_ok and not args.smoke:
        log("FATAL: frac 1.00 rho does not reproduce the run's control cell.")
        raise SystemExit(
            f"Determinism check B failed: frac 1.00 rho {rho_at_1:.3f} not within "
            f"{RHO_MATCH_TOL} of run rho {run_rho_point:.3f}. Refusing to emit a "
            f"dose curve on a non-reproducing anchor (project hard rule).")

    # --- OUTPUT-PUSH-pole characterization (NOT a Genuine-pole PASS/FAIL) ---
    clean = [r for r in rows if not r["gate_tripped"]]
    clean_fracs = [r["frac"] for r in clean]
    clean_rhos = [r["rho"] for r in clean]
    min_clean_rho = min(clean_rhos) if clean_rhos else None
    max_clean_rho = max(clean_rhos) if clean_rhos else None
    # (c) does rho stay ABOVE the Genuine line (0.3) across ALL clean budgets?
    stays_out_of_genuine = bool(clean) and all(r["rho"] > GENUINE_RHO for r in clean)
    # (d) rho at the LARGEST clean SUB-matched budget (frac < 1.00): does the push
    #     still reproduce most of the effect below the matched budget?
    clean_sub = [r for r in clean if r["frac"] < 1.0]
    largest_clean_submatched = (
        max(clean_sub, key=lambda r: r["frac"]) if clean_sub else None)
    rho_at_largest_clean_submatched = (
        largest_clean_submatched["rho"] if largest_clean_submatched else None)
    log(f"CURVE: clean fracs={clean_fracs} rho[min,max]="
        f"[{min_clean_rho if min_clean_rho is None else round(min_clean_rho,3)}, "
        f"{max_clean_rho if max_clean_rho is None else round(max_clean_rho,3)}] "
        f"stays_out_of_genuine={stays_out_of_genuine} "
        f"rho@largest_clean_submatched="
        f"{rho_at_largest_clean_submatched if rho_at_largest_clean_submatched is None else round(rho_at_largest_clean_submatched,3)}")

    out = {
        "meta": {
            "purpose": "ActAdd arm control-budget DOSE curve for the OUTPUT-PUSH "
                       "pole (plan §11): show the high rho is not a knife-edge at "
                       "one tuned budget but persists across a band of control "
                       "budgets and does not collapse into Genuine (<=0.3) below "
                       "the matched budget.",
            "arm": "ActAdd (Turner et al. 2308.10248), wedding topic",
            "pole": "OUTPUT-PUSH (high rho)",
            "model": args.model, "device": device,
            "pair_id": pair_id, "chosen_layer": chosen_L, "chosen_coeff": coeff,
            "n_eval": len(eval_), "n_calib": len(calib),
            "max_new_tokens": args.tokens, "n_boot": args.n_boot,
            "seed": args.seed, "dose_fracs": DOSE_FRACS,
            "matched_control_scalar": matched_c,
            "token_set_size": len(ctrl_token_ids),
            "token_set_matches_run": bool(ids_match),
            "gate_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
            "E_native_pts": E_native_pts, "baseline_rate": baseline_rate,
            "genuine_rho_line": GENUINE_RHO,
            "control_note": "calibrated static logit bias on the run's FROZEN token "
                            "set, built WITHOUT the steering vector; matched scalar "
                            "frozen from the run (not re-calibrated); frac 1.00 == "
                            "the run's control cell.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "run_reference": {
            "rho_point": run_rho_point, "rho_lo": run_rho_lo, "rho_hi": run_rho_hi,
            "control_rate": run_control_rate, "verdict_class": run["verdict"]["class"],
            "frozen_verdict_rule": FROZEN_VERDICT_RULE,
        },
        "determinism_checks": {
            "token_set_matches_run": bool(ids_match),
            "token_set_size": len(ctrl_token_ids),
            "frac_1.00_rho": rho_at_1,
            "run_rho_point": run_rho_point,
            "frac_1.00_rho_delta": det_b_delta,
            "frac_1.00_matches_run": bool(det_b_ok),
            "rho_match_tol": RHO_MATCH_TOL,
        },
        "grid": rows,
        "curve_characterization": {
            "n_clean_budgets": len(clean),
            "clean_fracs": clean_fracs,
            "clean_rhos": clean_rhos,
            "min_clean_rho": min_clean_rho,
            "max_clean_rho": max_clean_rho,
            "genuine_rho_line": GENUINE_RHO,
            "stays_out_of_genuine": stays_out_of_genuine,
            "largest_clean_submatched_frac": (
                largest_clean_submatched["frac"] if largest_clean_submatched else None),
            "rho_at_largest_clean_submatched": rho_at_largest_clean_submatched,
            "note": "OUTPUT-PUSH-pole read: rho(budget) curve + stability, NOT a "
                    "Dissolved/Mixed/Genuine PASS/FAIL. 'stays_out_of_genuine' = "
                    "rho > 0.3 at every clean (non-degenerate) budget.",
        },
        "runtime_sec": time.time() - t0,
    }
    jpath = os.path.join(outdir, "dose_response_budget.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"wrote {jpath}")
    write_md(out, os.path.join(outdir, "dose_response_budget.md"))
    log(f"wrote {os.path.join(outdir, 'dose_response_budget.md')}")
    log(f"DONE in {out['runtime_sec']:.0f}s  "
        f"(clean rho band [{min_clean_rho}, {max_clean_rho}], "
        f"stays_out_of_genuine={stays_out_of_genuine})")


def write_md(out, path):
    m = out["meta"]
    rr = out["run_reference"]
    dc = out["determinism_checks"]
    cc = out["curve_characterization"]
    L = []
    A = L.append
    A("# ActAdd arm — control-budget dose curve (OUTPUT-PUSH pole)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` (pair {m['pair_id']}, L={m['chosen_layer']}, "
      f"coeff={m['chosen_coeff']}), device `{m['device']}`.  ")
    A(f"**Eval:** {m['n_eval']} held-out prompts, {m['max_new_tokens']}-token "
      f"greedy, wedding-topic rate, {m['n_boot']} bootstrap. Control-only re-eval "
      f"— **no model-side intervention** (calibrated logit bias on the run's "
      f"frozen token set only, built WITHOUT the steering vector).  ")
    A(f"**Control token set:** {m['token_set_size']} tokens "
      f"(matches run set: {m['token_set_matches_run']}).  ")
    A(f"**Gate refs (frozen, run eval baseline):** rep={m['gate_refs']['rep']:.3f}, "
      f"median_len={m['gate_refs']['median_len']:.1f}, nll={m['gate_refs']['nll']:.3f}.\n")

    A("## Purpose\n")
    A(f"The ActAdd run was **{rr['verdict_class']}** with a HIGH "
      f"rho = E(control)/E(E_native) = **{rr['rho_point']:.3f}** "
      f"[{rr['rho_lo']:.3f}, {rr['rho_hi']:.3f}]: the matched-budget output-push "
      f"control reproduces almost all of the {m['E_native_pts']:.1f}-pt "
      f"wedding-topic effect. This is the OUTPUT-PUSH pole. A referee kill for the "
      f"rho-audit is that rho is not identified — its value depends on the "
      f"budget-matching functional. Here we test whether the high rho is a "
      f"**knife-edge** at the one tuned (matched) budget or a **band**: we sweep "
      f"the control at fracs of the matched scalar "
      f"({m['matched_control_scalar']:.4f}) below and above matched and report the "
      f"rho(budget) curve. The mirror question to the Genuine pole's Amendment-2 "
      f"sweep — does rho stay ABOVE the Genuine line ({m['genuine_rho_line']}) "
      f"across the clean band and NOT collapse below matched?\n")
    A(f"> Frozen verdict rule (reference only; NOT re-adjudicated here): "
      f"{rr['frozen_verdict_rule']}\n")

    A("## Determinism vs the run\n")
    A(f"- Control token set S matches run exactly: **{dc['token_set_matches_run']}** "
      f"(size {dc['token_set_size']}). h_delta rebuilt from the frozen contrastive "
      f"pair at L={m['chosen_layer']}; matched scalar loaded frozen from the run.")
    A(f"- frac 1.00 reproduction: rho = **{dc['frac_1.00_rho']:.3f}** vs run rho "
      f"**{dc['run_rho_point']:.3f}** (|delta| {dc['frac_1.00_rho_delta']:.3f}, tol "
      f"{dc['rho_match_tol']}) — matches = **{dc['frac_1.00_matches_run']}**.\n")

    A("## Dose-response curve (control only)\n")
    A(f"Bias scalars = frac x matched c ({m['matched_control_scalar']:.4f}); frac "
      f"1.00 == the run's control cell. **rho = gain_pts / E_native "
      f"({m['E_native_pts']:.1f} pts)** is rho AT THAT BUDGET. Same eval prompts / "
      f"generation / metric / gate throughout.\n")
    A("| frac | bias scalar | accuracy [95% CI] | gain vs baseline | **rho** "
      "(effect ratio) | gate |")
    A("|-----:|------------:|:-----------------:|----------------:|"
      "----------------------------:|:----:|")
    for r in out["grid"]:
        gate = "**VOID**" if r["gate_tripped"] else "clean"
        anchor = " *(matched)*" if abs(r["frac"] - 1.00) < 1e-9 else ""
        A(f"| {r['frac']:.2f}{anchor} | {r['bias_scalar']:.4f} | "
          f"{r['accuracy']*100:.1f}% [{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] | "
          f"{r['gain_pts']:+.1f} pts | {r['rho']:.3f} | {gate} |")
    A("")
    A("Per-budget gate detail (rep / median_len / nll):\n")
    for r in out["grid"]:
        reasons = "; ".join(r["gate_reasons"]) if r["gate_reasons"] else "—"
        A(f"- frac {r['frac']:.2f}: rep={r['gate_rep']:.3f}, "
          f"median_len={r['gate_median_len']:.1f}, nll={r['gate_nll']:.3f}  ({reasons})")
    A("")

    A("## Curve characterization (OUTPUT-PUSH pole)\n")
    A(f"Read of the rho(budget) curve — **NOT** a Dissolved/Mixed/Genuine "
      f"PASS/FAIL. The ActAdd verdict ({rr['verdict_class']}) is unchanged.\n")
    A(f"- (a) rho at each clean (non-degenerate) budget: " + ", ".join(
        f"frac {f:.2f} → {rho:.3f}"
        for f, rho in zip(cc["clean_fracs"], cc["clean_rhos"])) + ".")
    _minr = cc["min_clean_rho"]; _maxr = cc["max_clean_rho"]
    if _minr is not None and _maxr is not None:
        A(f"- (b) clean rho band: min = **{_minr:.3f}**, max = **{_maxr:.3f}** "
          f"over {cc['n_clean_budgets']} clean budgets.")
    else:
        A("- (b) clean rho band: n/a (no clean budgets).")
    A(f"- (c) stays ABOVE the Genuine line ({cc['genuine_rho_line']}) at EVERY "
      f"clean budget: **{cc['stays_out_of_genuine']}**.")
    if cc["rho_at_largest_clean_submatched"] is not None:
        A(f"- (d) rho at the largest clean SUB-matched budget "
          f"(frac {cc['largest_clean_submatched_frac']:.2f}): "
          f"**{cc['rho_at_largest_clean_submatched']:.3f}** — the push still "
          f"reproduces this fraction of the effect BELOW the matched budget.")
    else:
        A(f"- (d) rho at the largest clean sub-matched budget: n/a (no clean "
          f"sub-matched budget).")
    A("")
    if cc["stays_out_of_genuine"]:
        _read_a = ("a smooth, budget-graded curve that stays well above the "
                   "Genuine line")
        _read_b = "a BAND, not a knife-edge at the matched budget"
    else:
        _read_a = "NOT uniformly above the Genuine line"
        _read_b = "budget-sensitive; see the per-budget rho above"
    A(f"**Curve read:** across the clean band the control-budget rho is {_read_a} "
      f"— the high rho is {_read_b}.\n")
    A(f"Runtime: {out['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
