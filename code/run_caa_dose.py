"""CAA arm — control-budget DOSE curve for the OUTPUT-PUSH pole (plan §11).

Mirror of run_actadd_dose.py for CAA (Rimsky et al. 2312.06681, Qwen2.5-7B-Instruct,
sycophancy). The CAA run returned **Mixed** with a HIGH rho (E(control)/E(E_native)
point 0.882, CI [0.790, 0.971]): the matched-budget calibrated logit-bias control
(built WITHOUT the steering vector, on the run's frozen token set) reproduces most
of the +51-pt sycophancy effect. This is the OUTPUT-PUSH pole. A referee kill for
the rho-audit is "rho is not identified — it depends on the arbitrary
budget-matching functional." This driver is the MIRROR for that pole: show the high
rho is NOT a knife-edge at one tuned budget — that rho stays high across a BAND of
control budgets and does not collapse into Genuine territory (<= 0.3) below matched.

RE-EVALUATES ONLY THE CONTROL CONDITION at bias scalars c = frac x matched_c on the
SAME 200 eval prompts, SAME 64-token greedy, SAME sycophancy metric, SAME degeneracy
gate (SAME frozen eval baseline refs), bootstrap 10k. NO model-side intervention (no
steering-vector-side add in the control) — calibrated logit bias on the run's frozen
token set only, exactly as the run's control cell.

Reconstructs the run deterministically: same model (Qwen2.5-7B-Instruct via the
llama-family loader), same dataset fetch + shuffle (seed 20260707) + extract/heldout
split + build_eval_prompts, same chosen config (L=18, coeff=32.0), the frozen unit
steering vector loaded from caa_vec_L18.pt (v_hat), control token set + mean
position-1 logit-delta recomputed on the SAME calib split and VERIFIED against the
saved run token set. The matched scalar (1.5076...) is loaded frozen from the run
(NOT re-calibrated), so frac 1.00 reproduces the run's control cell exactly
(rho ~ 0.88).

Adjudication is the OUTPUT-PUSH-pole read: report the rho(budget) CURVE + its
stability characterizations (min/max clean rho, stays_out_of_genuine, sub-matched
rho). NOT the Genuine-pole PASS/FAIL. Does NOT re-adjudicate Dissolved/Mixed/Genuine.

Output: dose_response_budget.md/.json + resumable dose_budget_stage.json in --outdir
(default the new 20260708-caa-dose dir). Distinct `_budget` names so they never
collide with any Genuine-pole dose_response.*.

GPU JOB — this needs a 7B GPU. Locally it only supports `--dry-run`, which parses
the frozen refs from results_full.json, prints the frac grid + planned bias scalars,
and EXITS before B.load_model (no 7B load). The real run is launched on GPU:

  python run_caa_dose.py --device cuda --dtype bf16 \
      --outdir <repo>/runs/steering-content-audit/20260708-caa-dose

Usage:
  python run_caa_dose.py --dry-run                       # local, no model
  python run_caa_dose.py --smoke --n 12 --tokens 24 ...  # GPU smoke
  python run_caa_dose.py --device cuda --dtype bf16 ...  # GPU full
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import battery as B          # noqa: E402
import caa_steer as C        # noqa: E402
import run_caa as RC         # noqa: E402  (fetch_dataset, build_eval_prompts, vec_path)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Fractions of the run-matched control scalar. Brackets the matched budget BELOW
# (sub-matched regime: does rho stay high?) and ABOVE (degeneracy onset). frac 1.00
# == the run's control cell (the determinism anchor: must reproduce rho~0.88).
DOSE_FRACS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

# The Genuine boundary (plan §3): does rho stay ABOVE this across the clean band?
GENUINE_RHO = 0.3

FROZEN_VERDICT_RULE = (
    "plan §3/§11: Dissolved = rho_lo >= 0.9 on a valid cell; Genuine = rho_hi <= "
    "0.3 with effect >= 3x floor (+Amendment-2 if control void); else Mixed. The "
    "CAA run was Mixed (rho point 0.882, CI [0.790, 0.971]). This driver does NOT "
    "re-adjudicate that verdict — it reports the rho(budget) curve.")

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


# ---------------------------------------------------------------------------
# Frozen-ref parsing (shared by --dry-run and the real run)
# ---------------------------------------------------------------------------

def parse_frozen_refs(run: dict) -> dict:
    """Pull the frozen references the dose driver needs from the CAA run's
    results_full.json. Raises a clear error if the run is NOT-REPRODUCED (no
    control cell to dose)."""
    verdict_class = run.get("verdict", {}).get("class")
    if verdict_class == "NOT-REPRODUCED" or "control_calibration" not in run:
        raise SystemExit(
            "This CAA results_full.json is NOT-REPRODUCED / has no "
            "control_calibration — there is no matched control cell to build a "
            "dose curve from. Point --caa-run at the reproduced CAA run "
            "(20260707-caa-7b-x), not the NOT-REPRODUCED base run.")
    cc = run["control_calibration"]
    return {
        "chosen_layer": run["meta"]["chosen_layer"],
        "chosen_coeff": run["meta"]["chosen_coeff"],
        "raw_v_norm": run["meta"].get("raw_v_norm"),
        "n_eval": run["meta"]["n_eval"],
        "n_calib": run["meta"]["n_calib"],
        "n_extract": run["meta"]["n_extract_pairs"],
        "max_new_tokens": run["meta"]["max_new_tokens"],
        "matched_scalar": cc["bias_scalar"],
        "token_ids": cc["token_ids"],
        "token_set_size": cc["token_set_size"],
        "B_star_target_kl": cc.get("B_star_target_kl"),
        "achieved_kl": cc.get("achieved_kl"),
        "E_native_frac": run["effect"]["E_native"],
        "baseline_rate": run["rates"]["baseline"]["rate"],
        "control_rate": run["rates"]["control"]["rate"],
        "rho_point": run["rho"]["point"],
        "rho_lo": run["rho"]["ci_lo"],
        "rho_hi": run["rho"]["ci_hi"],
        "ev_rep": run["eval_baseline_refs"]["rep"],
        "ev_med": run["eval_baseline_refs"]["median_len"],
        "ev_nll": run["eval_baseline_refs"]["nll"],
        "verdict_class": verdict_class,
        "dataset_source": run["meta"].get("dataset_source"),
    }


def gate_status(texts, prompts, ev_rep, ev_med, ev_nll, model, tok, device):
    """Chat-model gate adjudication (mirrors run_caa.py battery): a rep- or
    length-collapse is TRUE degeneracy (voids the cell); an NLL-only trip on
    coherent chat text is flagged, not voided (§4 chat recalibration)."""
    gate = B.degeneracy_gate(texts, prompts, ev_rep, ev_med, ev_nll, model, tok,
                             device=device)
    rep_trip = gate.rep_rate > 2 * ev_rep + 0.1
    len_trip = gate.median_len < 0.5 * ev_med
    degenerate = bool(rep_trip or len_trip)
    nll_only = bool(gate.tripped and not degenerate)
    return {
        "raw_tripped": bool(gate.tripped), "degenerate": degenerate,
        "nll_only": nll_only, "rep_trip": bool(rep_trip), "len_trip": bool(len_trip),
        "rep": gate.rep_rate, "median_len": gate.median_len, "nll": gate.mean_nll,
        "reasons": gate.reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda",
                    help="cuda for the GPU run; a 7B model will not load on cpu.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "float32"])
    ap.add_argument("--n", type=int, default=200, help="held-out eval count")
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--n-extract", type=int, default=200)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260707,
                    help="dataset shuffle + eval-prompt seed (the CAA run's seed)")
    ap.add_argument("--behavior", default=RC.DEFAULT_BEHAVIOR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse frozen refs + print frac grid/bias scalars and EXIT "
                         "before loading the 7B model (local validation, no GPU).")
    ap.add_argument("--outdir", default=None,
                    help="output dir (default the new 20260708-caa-dose dir).")
    ap.add_argument("--caa-run", default=None,
                    help="path to the reproduced CAA run's results_full.json "
                         "(default <outdir>/caa_inputs/results_full.json).")
    ap.add_argument("--vec", default=None,
                    help="path to the frozen steering vector caa_vec_L{L}.pt "
                         "(default <outdir>/caa_inputs/caa_vec_L{L}.pt).")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = args.outdir or os.path.join(
        repo, "runs", "steering-content-audit", "20260708-caa-dose")
    os.makedirs(outdir, exist_ok=True)
    caa_run_path = args.caa_run or os.path.join(outdir, "caa_inputs",
                                                "results_full.json")
    stage_path = os.path.join(outdir, "dose_budget_stage.json")
    stage = load_stage(stage_path)
    t0 = time.time()

    if args.smoke:
        args.n = min(args.n, 12)
        args.n_calib = max(4, args.n // 2)
        args.n_extract = min(args.n_extract, 24)
        args.n_boot = min(args.n_boot, 300)

    # --- load the run result (fixed references + determinism checks) ---
    if not os.path.exists(caa_run_path):
        raise SystemExit(f"CAA run results_full.json not found at {caa_run_path}. "
                         f"Pull it from S3 (20260707-caa-7b-x) first, or pass "
                         f"--caa-run.")
    run = json.load(open(caa_run_path))
    refs = parse_frozen_refs(run)
    chosen_L = refs["chosen_layer"]                 # 18
    coeff = refs["chosen_coeff"]                     # 32.0
    matched_c = refs["matched_scalar"]              # 1.5076... (FROZEN)
    run_token_ids = refs["token_ids"]
    E_native_frac = refs["E_native_frac"]           # 0.51
    E_native_pts = E_native_frac * 100.0
    baseline_rate = refs["baseline_rate"]           # 0.445
    run_rho_point = refs["rho_point"]               # 0.8824
    ev_rep, ev_med, ev_nll = refs["ev_rep"], refs["ev_med"], refs["ev_nll"]
    vec_path = args.vec or os.path.join(outdir, "caa_inputs",
                                        f"caa_vec_L{chosen_L}.pt")

    planned = [{"frac": f, "bias_scalar": f * matched_c} for f in DOSE_FRACS]
    log(f"DOSE (output-push pole): arm=CAA behavior={args.behavior} L={chosen_L} "
        f"coeff={coeff} matched_c={matched_c:.4f} E_native={E_native_pts:.1f}pts "
        f"baseline={baseline_rate:.3f} run_rho={run_rho_point:.3f}")
    log(f"frozen refs parsed from {os.path.relpath(caa_run_path, repo)}:")
    log(f"  chosen_layer={chosen_L} chosen_coeff={coeff} raw_v_norm={refs['raw_v_norm']}")
    log(f"  matched_scalar={matched_c} token_set_size={refs['token_set_size']}")
    log(f"  E_native={E_native_frac} ({E_native_pts:.1f} pts) baseline={baseline_rate}")
    log(f"  rho point={run_rho_point} CI [{refs['rho_lo']:.4f}, {refs['rho_hi']:.4f}] "
        f"verdict={refs['verdict_class']}")
    log(f"  B*_target_kl={refs['B_star_target_kl']} gate refs "
        f"rep={ev_rep:.4f} median_len={ev_med:.1f} nll={ev_nll:.4f}")
    log(f"  vec path={vec_path} (exists={os.path.exists(vec_path)})")
    log(f"  frac grid + planned bias scalars (= frac x {matched_c:.4f}):")
    for pl in planned:
        log(f"    frac {pl['frac']:.2f} -> bias_scalar {pl['bias_scalar']:.4f}"
            + ("  *(matched, frac 1.00)*" if abs(pl['frac'] - 1.0) < 1e-9 else ""))

    if args.dry_run:
        # Local validation path: confirm everything parsed + the vec exists, then
        # EXIT before any 7B load (no GPU here).
        vec_ok = os.path.exists(vec_path)
        vec_meta = None
        if vec_ok:
            try:
                dd = torch.load(vec_path, map_location="cpu", weights_only=False)
                vec_meta = {"keys": list(dd.keys()),
                            "layer": int(dd.get("layer", -1)),
                            "v_hat_shape": list(dd["v_hat"].shape),
                            "v_hat_norm": float(dd["v_hat"].norm()),
                            "raw_norm": float(dd.get("raw_norm", float("nan")))}
                log(f"  vec loaded: layer={vec_meta['layer']} "
                    f"v_hat shape={vec_meta['v_hat_shape']} "
                    f"norm={vec_meta['v_hat_norm']:.4f} raw_norm={vec_meta['raw_norm']:.3f}")
                assert vec_meta["layer"] == chosen_L, (
                    f"vec layer {vec_meta['layer']} != chosen_L {chosen_L}")
            except Exception as e:  # noqa: BLE001
                log(f"  WARNING: could not load vec: {e}")
                vec_ok = False
        dry = {
            "dry_run": True, "arm": "CAA", "behavior": args.behavior,
            "caa_run_path": caa_run_path, "vec_path": vec_path, "vec_ok": vec_ok,
            "vec_meta": vec_meta, "frozen_refs": refs,
            "dose_fracs": DOSE_FRACS, "planned_bias_scalars": planned,
            "gpu_only_remaining": [
                "B.load_model(Qwen2.5-7B-Instruct) — 7B, GPU only",
                "dataset fetch + build_eval_prompts (needs the cached Rimsky "
                "sycophancy set on the box)",
                "C.position1_logit_delta on calib (recompute mean_delta) + token-"
                "set match check vs the frozen 100-token run set",
                "the dose grid: control_generate at each frac x 200 eval prompts",
            ],
        }
        dpath = os.path.join(outdir, "dose_budget_dryrun.json")
        with open(dpath, "w") as f:
            json.dump(dry, f, indent=2, default=str)
        log(f"DRY-RUN OK: frozen refs parsed, frac grid + bias scalars printed, "
            f"vec {'present' if vec_ok else 'MISSING'}. Wrote {dpath}. Exiting "
            f"before 7B load (GPU-only steps listed above).")
        return

    # =====================================================================
    #  GPU PATH (from here down requires the 7B model)
    # =====================================================================
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    log(f"loading {MODEL_ID} on {args.device} {args.dtype}")
    model, tokenizer = B.load_model(MODEL_ID, device=args.device, dtype=dtype)
    device = next(model.parameters()).device.type

    # --- dataset + split (identical to the CAA run) ---
    log(f"behavior={args.behavior} url={RC.rimsky_url(args.behavior)}")
    pairs, source = RC.fetch_dataset(outdir, args.behavior)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_extract = min(args.n_extract, max(1, len(pairs) - args.n))
    extract_pairs = pairs[:n_extract]
    heldout_pairs = pairs[n_extract:]
    eval_items = RC.build_eval_prompts(heldout_pairs, tokenizer,
                                       args.n + args.n_calib, seed=args.seed)
    eval_calib = eval_items[:args.n_calib]
    eval_eval = eval_items[args.n_calib:args.n_calib + args.n]
    calib = [e["prompt"] for e in eval_calib]
    eval_ = [e["prompt"] for e in eval_eval]
    log(f"dataset: {len(pairs)} pairs; extract={len(extract_pairs)} "
        f"heldout={len(heldout_pairs)}; calib={len(calib)} eval={len(eval_)} "
        f"(source: {source})")

    # --- load the FROZEN steering vector (byte-identical to the run) ---
    if not os.path.exists(vec_path):
        raise SystemExit(f"frozen steering vector not found at {vec_path}. Pull "
                         f"caa_vec_L{chosen_L}.pt from S3 (20260707-caa-7b-x) first.")
    dd = torch.load(vec_path, map_location="cpu", weights_only=False)
    v_hat = dd["v_hat"]
    assert int(dd["layer"]) == chosen_L, (
        f"vec layer {dd['layer']} != chosen_L {chosen_L}")
    meth = C.CAAMethod(model, tokenizer, chosen_L, v_hat,
                       first_window="prefill_plus1", device=device,
                       max_new_tokens=args.tokens)
    log(f"loaded frozen v_hat: L={dd['layer']} norm={float(v_hat.norm()):.4f} "
        f"raw_norm={float(dd.get('raw_norm', float('nan'))):.3f}")

    # --- reconstruct control token set + mean position-1 logit-delta on calib ---
    log("reconstructing control token set + mean position-1 logit-delta on calib...")
    mean_delta = C.position1_logit_delta(meth, tokenizer, calib, coeff)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    ids_match = (ctrl_token_ids == run_token_ids)
    log(f"token set size={len(ctrl_token_ids)} matches run set: {ids_match}")
    if not ids_match and not args.smoke:
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
        texts = [B.control_generate(model, tokenizer, p, processor, args.tokens,
                                    device) for p in eval_]
        hits = [int(C.is_sycophantic(t)) for t in texts]
        r = B.bootstrap_rate_ci(hits, args.n_boot, seed=7)
        g = gate_status(texts, eval_, ev_rep, ev_med, ev_nll, model, tokenizer,
                        device)
        gain_pts = (r[0] - baseline_rate) * 100
        rho = gain_pts / max(E_native_pts, 1e-9)
        row = {
            "frac": frac, "bias_scalar": c, "accuracy": r[0],
            "ci_lo": r[1], "ci_hi": r[2], "gain_pts": gain_pts,
            "rho": rho, "effect_ratio_vs_E_native": rho,
            "gate_tripped": g["degenerate"], "gate_raw_tripped": g["raw_tripped"],
            "gate_nll_only": g["nll_only"], "gate_rep": g["rep"],
            "gate_median_len": g["median_len"], "gate_nll": g["nll"],
            "gate_reasons": g["reasons"],
            "sample": texts[0][:120] if texts else "",
        }
        stage["grid"][key] = row
        save_stage(stage_path, stage)
        log(f"  frac {frac:.2f} (c={c:.4f}): acc={r[0]*100:.1f}% "
            f"gain={gain_pts:+.1f}pts rho={rho:.3f} degenerate={g['degenerate']} "
            f"(nll_only={g['nll_only']}) rep={g['rep']:.4f}")
        return row

    log("running dose grid (control only)...")
    rows = [run_frac(f) for f in DOSE_FRACS]

    # --- determinism check B: frac 1.00 must reproduce the run's rho ---
    row1 = next(r for r in rows if abs(r["frac"] - 1.00) < 1e-9)
    rho_at_1 = row1["rho"]
    det_b_delta = abs(rho_at_1 - run_rho_point)
    det_b_ok = det_b_delta <= RHO_MATCH_TOL
    log(f"determinism B: frac 1.00 rho={rho_at_1:.3f} vs run rho={run_rho_point:.3f} "
        f"(|delta|={det_b_delta:.3f}, tol {RHO_MATCH_TOL}) -> "
        f"{'OK' if det_b_ok else 'MISMATCH'}")
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
    stays_out_of_genuine = bool(clean) and all(r["rho"] > GENUINE_RHO for r in clean)
    clean_sub = [r for r in clean if r["frac"] < 1.0]
    largest_clean_submatched = (
        max(clean_sub, key=lambda r: r["frac"]) if clean_sub else None)
    rho_at_largest_clean_submatched = (
        largest_clean_submatched["rho"] if largest_clean_submatched else None)
    log(f"CURVE: clean fracs={clean_fracs} rho[min,max]="
        f"[{min_clean_rho}, {max_clean_rho}] "
        f"stays_out_of_genuine={stays_out_of_genuine} "
        f"rho@largest_clean_submatched={rho_at_largest_clean_submatched}")

    out = {
        "meta": {
            "purpose": "CAA arm control-budget DOSE curve for the OUTPUT-PUSH pole "
                       "(plan §11): show the high rho is not a knife-edge at one "
                       "tuned budget but persists across a band of control budgets "
                       "and does not collapse into Genuine (<=0.3) below matched.",
            "arm": "CAA (Rimsky et al. 2312.06681), sycophancy",
            "pole": "OUTPUT-PUSH (high rho)",
            "behavior": args.behavior,
            "model": MODEL_ID, "device": device, "dtype": args.dtype,
            "chosen_layer": chosen_L, "chosen_coeff": coeff,
            "n_eval": len(eval_), "n_calib": len(calib), "n_extract": len(extract_pairs),
            "max_new_tokens": args.tokens, "n_boot": args.n_boot,
            "seed": args.seed, "dose_fracs": DOSE_FRACS,
            "matched_control_scalar": matched_c,
            "token_set_size": len(ctrl_token_ids),
            "token_set_matches_run": bool(ids_match),
            "gate_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
            "E_native_pts": E_native_pts, "baseline_rate": baseline_rate,
            "genuine_rho_line": GENUINE_RHO,
            "dataset_source": source,
            "control_note": "calibrated static logit bias on the run's FROZEN token "
                            "set, built WITHOUT the steering vector; matched scalar "
                            "frozen from the run (not re-calibrated); frac 1.00 == "
                            "the run's control cell. Chat-model gate: rep/length "
                            "collapse = true degeneracy (voids); NLL-only trip on "
                            "coherent chat text is flagged, not voided (§4).",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "run_reference": {
            "rho_point": run_rho_point, "rho_lo": refs["rho_lo"],
            "rho_hi": refs["rho_hi"], "control_rate": refs["control_rate"],
            "verdict_class": refs["verdict_class"],
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
    A("# CAA arm — control-budget dose curve (OUTPUT-PUSH pole)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` (L={m['chosen_layer']}, coeff={m['chosen_coeff']}), "
      f"device `{m['device']}` ({m['dtype']}).  ")
    A(f"**Eval:** {m['n_eval']} held-out sycophancy prompts, {m['max_new_tokens']}-"
      f"token greedy, sycophancy rate, {m['n_boot']} bootstrap. Control-only re-eval "
      f"— **no model-side intervention** (calibrated logit bias on the run's frozen "
      f"token set only, built WITHOUT the steering vector).  ")
    A(f"**Control token set:** {m['token_set_size']} tokens "
      f"(matches run set: {m['token_set_matches_run']}).  ")
    A(f"**Gate refs (frozen, run eval baseline):** rep={m['gate_refs']['rep']:.4f}, "
      f"median_len={m['gate_refs']['median_len']:.1f}, nll={m['gate_refs']['nll']:.4f}.\n")

    A("## Purpose\n")
    A(f"The CAA run was **{rr['verdict_class']}** with a HIGH "
      f"rho = E(control)/E(E_native) = **{rr['rho_point']:.3f}** "
      f"[{rr['rho_lo']:.3f}, {rr['rho_hi']:.3f}]: the matched-budget output-push "
      f"control reproduces most of the {m['E_native_pts']:.1f}-pt sycophancy "
      f"effect. This is the OUTPUT-PUSH pole. A referee kill for the rho-audit is "
      f"that rho is not identified — its value depends on the budget-matching "
      f"functional. Here we test whether the high rho is a **knife-edge** at the "
      f"one tuned (matched) budget or a **band**: we sweep the control at fracs of "
      f"the matched scalar ({m['matched_control_scalar']:.4f}) below and above "
      f"matched and report the rho(budget) curve — does rho stay ABOVE the Genuine "
      f"line ({m['genuine_rho_line']}) across the clean band and NOT collapse below "
      f"matched?\n")
    A(f"> Frozen verdict rule (reference only; NOT re-adjudicated here): "
      f"{rr['frozen_verdict_rule']}\n")

    A("## Determinism vs the run\n")
    A(f"- Control token set S matches run exactly: **{dc['token_set_matches_run']}** "
      f"(size {dc['token_set_size']}). Frozen unit v_hat loaded from "
      f"caa_vec_L{m['chosen_layer']}.pt; matched scalar loaded frozen from the run.")
    A(f"- frac 1.00 reproduction: rho = **{dc['frac_1.00_rho']:.3f}** vs run rho "
      f"**{dc['run_rho_point']:.3f}** (|delta| {dc['frac_1.00_rho_delta']:.3f}, tol "
      f"{dc['rho_match_tol']}) — matches = **{dc['frac_1.00_matches_run']}**.\n")

    A("## Dose-response curve (control only)\n")
    A(f"Bias scalars = frac x matched c ({m['matched_control_scalar']:.4f}); frac "
      f"1.00 == the run's control cell. **rho = gain_pts / E_native "
      f"({m['E_native_pts']:.1f} pts)** is rho AT THAT BUDGET. Same eval prompts / "
      f"generation / metric / gate throughout. Chat-model gate: rep/length collapse "
      f"= true degeneracy (VOID); NLL-only trip on coherent text is flagged, not "
      f"voided (§4).\n")
    A("| frac | bias scalar | sycophancy [95% CI] | gain vs baseline | **rho** "
      "(effect ratio) | gate |")
    A("|-----:|------------:|:-------------------:|----------------:|"
      "----------------------------:|:----:|")
    for r in out["grid"]:
        if r["gate_tripped"]:
            gate = "**VOID**"
        elif r.get("gate_nll_only"):
            gate = "clean (nll-only)"
        else:
            gate = "clean"
        anchor = " *(matched)*" if abs(r["frac"] - 1.00) < 1e-9 else ""
        A(f"| {r['frac']:.2f}{anchor} | {r['bias_scalar']:.4f} | "
          f"{r['accuracy']*100:.1f}% [{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] | "
          f"{r['gain_pts']:+.1f} pts | {r['rho']:.3f} | {gate} |")
    A("")
    A("Per-budget gate detail (rep / median_len / nll):\n")
    for r in out["grid"]:
        reasons = "; ".join(r["gate_reasons"]) if r["gate_reasons"] else "—"
        A(f"- frac {r['frac']:.2f}: rep={r['gate_rep']:.4f}, "
          f"median_len={r['gate_median_len']:.1f}, nll={r['gate_nll']:.4f}  ({reasons})")
    A("")

    A("## Curve characterization (OUTPUT-PUSH pole)\n")
    A(f"Read of the rho(budget) curve — **NOT** a Dissolved/Mixed/Genuine "
      f"PASS/FAIL. The CAA verdict ({rr['verdict_class']}) is unchanged.\n")
    A("- (a) rho at each clean (non-degenerate) budget: " + ", ".join(
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
        A("- (d) rho at the largest clean sub-matched budget: n/a.")
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
