"""Task-vectors arm — control dose-response sensitivity (plan §11 Amendment 2).

The task-vectors run returned Genuine (rho_hi = -0.061, native REPLACE gain +47.3
pts) with the primary calibrated logit-bias control cell VOID (control degenerated
at the matched budget c=3.592: rep 0.833, "vacancy vacancy..." loops). Amendment 2
requires a dose-response: the primary control, evaluated at >=3 sub-degenerate
scales (below the degeneracy gate), must show effect <= 0.3 x E_native at every
non-degenerate scale. PASS => no admissible scale of the prompt-independent
output-push control reproduces the task-vector effect, so the void control cell
does not unfairly favor Genuine.

RE-EVALUATES ONLY THE CONTROL CONDITION at bias scalars c in fractions of the
matched budget on the SAME 150 eval prompts, SAME 8-token greedy, SAME
task-accuracy metric, SAME degeneracy gate (SAME baseline refs), bootstrap 10k.

Reconstructs the run deterministically: same model (pythia-2.8b, MPS, bf16), same
seed (20260707), same dataset split, same chosen layer/op (L=12, REPLACE), theta
loaded from the run's stage cache, control token set + mean position-1 logit-delta
recomputed on the SAME calib split (verified against the saved run token set), so
the control at the matched scale reproduces the run exactly.

Nothing else re-run. E_native and baseline are fixed run references. Output:
dose_response.md + dose_response.json in the taskvec-arm dir. Does not touch
report.md/results_full.json. Staged to disk (dose_stage.json) for resume.

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python run_taskvec_dose.py
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
import fv_extract as FV      # noqa: E402
import taskvec as TV         # noqa: E402
from run_taskvec import answer_hit, hits_for  # noqa: E402

# Fractions of the run-matched control scalar (frac 1.0 == the run's void cell).
DOSE_FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.00]
AMEND2_RATIO = 0.3       # control effect must be <= 0.3 x E_native at every scale
MIN_CLEAN_SCALES = 3


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "external", "function_vectors", "dataset_files", "abstractive")


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
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16", choices=["float32", "bf16"])
    ap.add_argument("--task", default="antonym")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--n-mean", type=int, default=100)
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-07-taskvec-arm")
    stage_path = os.path.join(outdir, "dose_stage.json")
    stage = load_stage(stage_path)
    t0 = time.time()

    # --- load the run result (fixed references + determinism checks) ---
    run = json.load(open(os.path.join(outdir, "results_full.json")))
    chosen_L = run["meta"]["chosen_layer"]
    chosen_op = run["meta"]["chosen_op"]
    matched_c = run["control_calibration"]["bias_scalar"]
    run_token_ids = run["control_calibration"]["token_ids"]
    E_native_pts = run["effect"]["E_native"] * 100.0
    baseline_rate = run["rates"]["baseline"]["rate"]
    ev_rep = run["eval_baseline_refs"]["rep"]
    ev_med = run["eval_baseline_refs"]["median_len"]
    ev_nll = run["eval_baseline_refs"]["nll"]
    gain_bound_pts = AMEND2_RATIO * E_native_pts
    log(f"DOSE (Amendment 2): op={chosen_op} L={chosen_L} matched_c={matched_c:.4f} "
        f"E_native={E_native_pts:.1f}pts bound={gain_bound_pts:.1f}pts fracs={DOSE_FRACS}")

    # --- model ---
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tok = B.load_model(args.model, device=args.device, dtype=_dtype)
    device = args.device

    # --- dataset split (identical to the run) ---
    pairs = FV.load_pairs(os.path.join(DATA_DIR, f"{args.task}.json"))
    train_pool, eval_pairs = FV.split_pairs(pairs, args.n_eval, seed=args.seed)
    eval_pairs = eval_pairs[:args.n_eval]
    rng = np.random.default_rng(args.seed + 1)
    eidx = rng.permutation(len(eval_pairs))
    calib_pairs = [eval_pairs[i] for i in eidx[:args.n_calib]]
    heldout_pairs = [eval_pairs[i] for i in eidx[args.n_calib:]]

    def zs_pairs(prs):
        return ([FV.zero_shot_prompt(x) for (x, y) in prs],
                [y for (x, y) in prs])
    calib_prompts, calib_golds = zs_pairs(calib_pairs)
    eval_prompts, eval_golds = zs_pairs(heldout_pairs)
    if args.smoke:
        eval_prompts, eval_golds = eval_prompts[:args.n_eval], eval_golds[:args.n_eval]
    log(f"split: calib {len(calib_prompts)} / eval {len(eval_prompts)}")

    # --- theta from stage cache (byte-identical) + method ---
    tv_stage = json.load(open(os.path.join(outdir, "stage_antonym_full.json")))
    theta = torch.tensor(tv_stage["theta"][str(chosen_L)], dtype=torch.float32)
    meth = TV.TaskVecMethod(model, tok, chosen_L, theta, op=chosen_op,
                            device=device, max_new_tokens=args.max_new_tokens)
    log(f"theta L={chosen_L} norm={meth.norm:.3f}")

    # --- reconstruct control token set + mean position-1 logit-delta on calib ---
    log("reconstructing control token set + mean position-1 logit-delta on calib...")
    mean_delta = meth.position1_logit_delta(calib_prompts)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    ids_match = (ctrl_token_ids == run_token_ids)
    log(f"token set size={len(ctrl_token_ids)} matches run set: {ids_match}")
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
        texts = [B.control_generate(model, tok, p, processor,
                                    args.max_new_tokens, device)
                 for p in eval_prompts]
        hits = hits_for(texts, eval_golds)
        r = B.bootstrap_rate_ci(hits, args.n_boot, seed=7)
        gg = B.degeneracy_gate(texts, eval_prompts, ev_rep, ev_med, ev_nll,
                               model, tok, device=device)
        gain_pts = (r[0] - baseline_rate) * 100
        ratio = gain_pts / max(E_native_pts, 1e-9)
        row = {
            "frac": frac, "bias_scalar": c, "accuracy": r[0],
            "ci_lo": r[1], "ci_hi": r[2], "gain_pts": gain_pts,
            "effect_ratio_vs_E_native": ratio,
            "gate_tripped": bool(gg.tripped), "gate_rep": gg.rep_rate,
            "gate_median_len": gg.median_len, "gate_nll": gg.mean_nll,
            "gate_reasons": gg.reasons,
            "sample": texts[0][:60] if texts else "",
        }
        stage["grid"][key] = row
        save_stage(stage_path, stage)
        log(f"  frac {frac:.2f} (c={c:.4f}): acc={r[0]*100:.1f}% gain={gain_pts:+.1f}pts "
            f"ratio={ratio:.3f} tripped={gg.tripped} rep={gg.rep_rate:.3f}")
        return row

    log("running dose grid (control only)...")
    rows = [run_frac(f) for f in DOSE_FRACS]

    # --- Amendment 2 adjudication ---
    nondegen = [r for r in rows if not r["gate_tripped"]]
    n_nondegen = len(nondegen)
    all_within = all(r["gain_pts"] <= gain_bound_pts for r in nondegen)
    passed = (n_nondegen >= MIN_CLEAN_SCALES) and all_within
    verdict = "PASS" if passed else "FAIL"
    over = [r for r in nondegen if r["gain_pts"] > gain_bound_pts]
    fail_reasons = []
    if n_nondegen < MIN_CLEAN_SCALES:
        fail_reasons.append(f"only {n_nondegen} non-degenerate scale(s) (<3)")
    if over:
        fail_reasons.append("scales exceeding bound: " + ", ".join(
            f"frac={r['frac']} gain={r['gain_pts']:+.1f}pts" for r in over))
    max_clean_ratio = max((r["effect_ratio_vs_E_native"] for r in nondegen),
                          default=None)
    log(f"AMENDMENT 2: {verdict} (non-degenerate={n_nondegen}, "
        f"bound={gain_bound_pts:.1f}pts, all_within={all_within})")

    out = {
        "meta": {
            "purpose": "task-vectors arm control dose-response sensitivity "
                       "(plan §11 Amendment 2)",
            "arm": "task-vectors (Hendel 2310.15916, REPLACE)",
            "model": args.model, "device": device, "dtype": args.dtype,
            "task": args.task, "chosen_layer": chosen_L, "chosen_op": chosen_op,
            "n_eval": len(eval_prompts), "n_calib": len(calib_prompts),
            "max_new_tokens": args.max_new_tokens, "n_boot": args.n_boot,
            "seed": args.seed, "dose_fracs": DOSE_FRACS,
            "matched_control_scalar": matched_c,
            "token_set_size": len(ctrl_token_ids),
            "token_set_matches_run": bool(ids_match),
            "gate_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
            "E_native_pts": E_native_pts, "baseline_rate": baseline_rate,
            "amend2_ratio": AMEND2_RATIO, "gain_bound_pts": gain_bound_pts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "grid": rows,
        "verdict": {
            "amendment2": verdict, "passed": bool(passed),
            "rule": f"PASS = >=3 non-degenerate scales AND every non-degenerate "
                    f"scale has gain <= {AMEND2_RATIO} x E_native = "
                    f"{gain_bound_pts:.1f} pts (ratio <= {AMEND2_RATIO}).",
            "n_nondegenerate_scales": n_nondegen,
            "clean_fracs": [r["frac"] for r in nondegen],
            "max_clean_effect_ratio": max_clean_ratio,
            "all_nondegen_within_bound": bool(all_within),
            "fail_reasons": fail_reasons,
        },
        "runtime_sec": time.time() - t0,
    }
    jpath = os.path.join(outdir, "dose_response.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"wrote {jpath}")
    write_md(out, os.path.join(outdir, "dose_response.md"))
    log(f"wrote {os.path.join(outdir, 'dose_response.md')}")
    log(f"DONE in {out['runtime_sec']:.0f}s  AMENDMENT 2: {verdict}")


def write_md(out, path):
    m = out["meta"]; v = out["verdict"]
    L = []
    A = L.append
    A("# Task-vectors arm — control dose-response (Amendment 2)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` (op {m['chosen_op'].upper()}, L={m['chosen_layer']}), "
      f"device `{m['device']}` ({m['dtype']}).  ")
    A(f"**Eval:** {m['n_eval']} held-out prompts, {m['max_new_tokens']}-token greedy, "
      f"task accuracy, {m['n_boot']} bootstrap. Control-only re-eval — no model-side "
      f"intervention (calibrated logit bias on the run's token set only).  ")
    A(f"**Control token set:** {m['token_set_size']} tokens "
      f"(matches run set: {m['token_set_matches_run']}).  ")
    A(f"**Gate refs (frozen, run):** rep={m['gate_refs']['rep']:.3f}, "
      f"median_len={m['gate_refs']['median_len']:.1f}, nll={m['gate_refs']['nll']:.3f}.\n")

    A("## Purpose\n")
    A(f"The task-vectors run returned **Genuine** (rho_hi -0.061) with the primary "
      f"logit-bias control cell VOID (degenerate at matched budget c="
      f"{m['matched_control_scalar']:.3f}: rep 0.833, \"vacancy vacancy...\"). Per "
      f"plan §3 + §11 Amendment 2, a void control does not block Genuine **provided** "
      f"the control at >=3 sub-degenerate scales all show effect <= "
      f"{m['amend2_ratio']} x E_native (= {m['gain_bound_pts']:.1f} pts of the "
      f"{m['E_native_pts']:.1f}-pt effect). This closes the hole that a degenerate "
      f"(deflated) control could otherwise unfairly favor Genuine.\n")

    A("## Dose-response grid (control only)\n")
    A(f"Bias scalars = frac x matched c ({m['matched_control_scalar']:.4f}); frac "
      f"1.00 == the run's void cell. Same eval prompts / generation / metric / gate.\n")
    A("| frac | bias scalar | accuracy [95% CI] | gain vs baseline | effect ratio "
      "(/E_native) | gate |")
    A("|-----:|------------:|:-----------------:|----------------:|"
      "----------------------------:|:----:|")
    for r in out["grid"]:
        gate = "**VOID**" if r["gate_tripped"] else "clean"
        A(f"| {r['frac']:.2f} | {r['bias_scalar']:.4f} | {r['accuracy']*100:.1f}% "
          f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] | {r['gain_pts']:+.1f} pts | "
          f"{r['effect_ratio_vs_E_native']:.3f} | {gate} |")
    A("")
    A("Per-scale gate detail (rep / median_len / nll):\n")
    for r in out["grid"]:
        reasons = "; ".join(r["gate_reasons"]) if r["gate_reasons"] else "—"
        A(f"- frac {r['frac']:.2f}: rep={r['gate_rep']:.3f}, "
          f"median_len={r['gate_median_len']:.1f}, nll={r['gate_nll']:.3f}  ({reasons})")
    A("")

    A("## Adjudication (plan §11 Amendment 2)\n")
    A(f"- Rule: {v['rule']}")
    A(f"- Non-degenerate (sub-gate) scales: **{v['n_nondegenerate_scales']}** "
      f"(fracs {v['clean_fracs']}); need >= {MIN_CLEAN_SCALES}.")
    mr = v['max_clean_effect_ratio']
    A(f"- Max effect ratio among clean scales: "
      f"**{mr:.3f}**" + (f"" if mr is None else "") +
      f" (bound {m['amend2_ratio']}); all clean <= bound = "
      f"**{v['all_nondegen_within_bound']}**.")
    A(f"- Amendment 2 passes = **{v['passed']}**.\n")
    A(f"## VERDICT: **{v['amendment2']}**\n")
    if v["passed"]:
        A(f"**PASS** — {v['n_nondegenerate_scales']} sub-degenerate scales, and every "
          f"non-degenerate scale reproduces <= {m['gain_bound_pts']:.1f} pts of the "
          f"{m['E_native_pts']:.1f}-pt task-vector effect. No admissible scale of the "
          f"prompt-independent output-push control reproduces the input-dependent "
          f"antonym behavior; the void control cell does not unfairly favor Genuine. "
          f"**Task-vectors Genuine verdict stands.**")
    else:
        A(f"**FAIL** — {'; '.join(v['fail_reasons'])}. Amendment 2 not satisfied.")
    A("")
    A(f"Runtime: {out['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
