"""A1 anchor — control dose-response sensitivity (plan §11 Amendment 2).

A1 run 1 returned Genuine with the primary logit-bias control cell VOID
(control degenerated at the matched budget, bias scalar c=3.5408: rep 0.524,
"conversely conversely..." loops). Amendment 2 requires a dose-response
sensitivity: the primary control, evaluated at >=3 sub-degenerate scales (below
the degeneracy gate), must show effect <= 0.3 x E(steering) at every
non-degenerate scale. PASS => no admissible scale of the control reproduces the
FV effect, so the void control does not unfairly favor Genuine.

This driver RE-EVALUATES ONLY THE CONTROL CONDITION at bias scalars
c in {0.44, 0.89, 1.77, 2.66, 3.54} (3.54 = run-1 matched budget, for
continuity) on the SAME 150 eval prompts, SAME 8-token greedy generation, SAME
task-accuracy metric, SAME degeneracy gate (same baseline refs rep=0.008,
median_len=8.0, nll=1.006), bootstrap 10k CIs per scale.

It reconstructs the exact run-1 setup deterministically: same model
(EleutherAI/pythia-2.8b, MPS, bf16), same seed (20260706), same dataset split,
same edit_layer (11), FV loaded from the stage1 cache, and recomputes the
position-1 mean logit delta on the SAME calib split (identical prompts under the
fixed seed) so the control bias vector c * mean_delta[token_set] at c=3.54
reproduces run-1's control exactly.

Nothing else is re-run (no FV eval, no floor, no kappa/rho recompute). E(FV) and
the baseline are taken as fixed run-1 references (E_all = 39.3 pts, baseline
5.3%). Output: dose_response.md (append a "## Dose-response sensitivity
(Amendment 2)" section) + dose_response.json in the run-1 dir. Does not touch
report.md or any pre-existing file.

Usage:
  source .venv/bin/activate
  python run_a1_dose.py            # full (pythia-2.8b, MPS, bf16)
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
import battery as B          # noqa: E402
import fv_extract as FV      # noqa: E402
from run_a1 import FVMethod, answer_hit, fv_position1_logit_delta  # noqa: E402

DOSE_SCALES = [0.44, 0.89, 1.77, 2.66, 3.54]  # 3.54 ~= run-1 matched c=3.5408
E_FV_PTS = 39.3          # run-1 E_all effect (pts); Amendment 2 denominator
BASELINE_RATE = 0.053    # run-1 zero-shot baseline (5.3%)
AMEND2_RATIO = 0.3       # control effect must be <= 0.3 x E(FV) at every scale
GAIN_BOUND_PTS = AMEND2_RATIO * E_FV_PTS   # 11.79 pts


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "external", "function_vectors", "dataset_files", "abstractive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16", choices=["float32", "bf16"])
    ap.add_argument("--task", default="antonym")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260706)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-06-a1-anchor")
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    log(f"DOSE-RESPONSE (Amendment 2)  model={args.model} device={args.device} "
        f"scales={DOSE_SCALES}")

    # --- model (same as run 1) ---
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tok = B.load_model(args.model, device=args.device, dtype=_dtype)
    device = args.device
    cfg = FV.neox_config(model)
    n_layers = cfg["n_layers"]

    # --- dataset split (identical to run 1) ---
    pairs = FV.load_pairs(os.path.join(DATA_DIR, f"{args.task}.json"))
    train_pool, eval_pairs = FV.split_pairs(pairs, args.n_eval, seed=args.seed)
    eval_pairs = eval_pairs[:args.n_eval]
    rng = np.random.default_rng(args.seed + 1)
    eidx = rng.permutation(len(eval_pairs))
    calib_pairs = [eval_pairs[i] for i in eidx[:args.n_calib]]
    heldout_pairs = [eval_pairs[i] for i in eidx[args.n_calib:]]
    log(f"split: calib {len(calib_pairs)} / eval {len(heldout_pairs)}")

    def zs_pairs(prs):
        return ([FV.zero_shot_prompt(x) for (x, y) in prs],
                [y for (x, y) in prs])

    calib_prompts, calib_golds = zs_pairs(calib_pairs)
    eval_prompts, eval_golds = zs_pairs(heldout_pairs)

    # --- FV from stage1 cache (same edit_layer=11) ---
    s1_cache = os.path.join(
        outdir, f"stage1_{args.task}_{args.model.split('/')[-1]}.pt")
    if not os.path.exists(s1_cache):
        raise SystemExit(f"stage1 cache not found: {s1_cache}")
    log(f"loading stage1 cache {s1_cache}")
    blob = torch.load(s1_cache, weights_only=False)  # our own trusted cache
    fv_vec = blob["fv"]
    edit_layer = blob["edit_layer"]
    log(f"  FV norm={fv_vec.norm().item():.3f}  edit_layer={edit_layer}")
    fvm = FVMethod(model, tok, edit_layer, fv_vec, device=device,
                   max_new_tokens=args.max_new_tokens)

    # --- rebuild the control token set + mean_delta on the SAME calib split ---
    log("control: position-1 mean logit delta on calib (rebuild bias basis)...")
    mean_delta = fv_position1_logit_delta(fvm, calib_prompts)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    log(f"  token set size={len(ctrl_token_ids)} "
        f"top={[tok.decode([i]) for i in ctrl_token_ids[:8]]}")

    # sanity: token set matches run-1 saved set (deterministic reconstruction)
    run1 = json.load(open(os.path.join(outdir, "results_full.json")))
    run1_ids = run1["control_calibration"]["token_ids"]
    ids_match = (ctrl_token_ids == run1_ids)
    log(f"  token set matches run-1 saved set: {ids_match}")

    tid_t = torch.tensor(ctrl_token_ids)

    # baseline refs frozen from run 1 (Amendment 2 requires SAME gate refs)
    ev_rep = run1["eval_baseline_refs"]["rep"]      # 0.00778
    ev_med = run1["eval_baseline_refs"]["median_len"]  # 8.0
    ev_nll = run1["eval_baseline_refs"]["nll"]      # 1.00564
    log(f"gate refs (frozen, run-1): rep={ev_rep:.5f} med={ev_med} nll={ev_nll:.5f}")

    # =====================================================================
    # Dose sweep — control only
    # =====================================================================
    rows = []
    for c in DOSE_SCALES:
        log(f"--- scale c={c} ---")
        bias_vals = c * mean_delta[tid_t]
        processor = B.LogitBiasProcessor(ctrl_token_ids, bias_vals)
        ctrl_texts = [B.control_generate(model, tok, p, processor,
                                         args.max_new_tokens, device)
                      for p in eval_prompts]
        ctrl_hits = [int(answer_hit(t, g))
                     for t, g in zip(ctrl_texts, eval_golds)]
        r = B.bootstrap_rate_ci(ctrl_hits, args.n_boot, seed=7)
        gg = B.degeneracy_gate(ctrl_texts, eval_prompts, ev_rep, ev_med, ev_nll,
                               model, tok, device=device)
        gain_pts = (r[0] - BASELINE_RATE) * 100
        effect_ratio = gain_pts / E_FV_PTS
        rows.append({
            "scale": c,
            "accuracy": r[0], "ci_lo": r[1], "ci_hi": r[2],
            "gain_pts": gain_pts,
            "gate_tripped": bool(gg.tripped),
            "gate_rep": gg.rep_rate, "gate_median_len": gg.median_len,
            "gate_nll": gg.mean_nll, "gate_reasons": gg.reasons,
            "effect_ratio_vs_E_FV": effect_ratio,
            "samples": list(zip(eval_prompts[:6], [t.strip()[:50] for t in ctrl_texts[:6]],
                                eval_golds[:6])),
        })
        log(f"  acc={r[0]*100:.1f}% [{r[1]*100:.1f},{r[2]*100:.1f}]  "
            f"gain={gain_pts:+.1f}pts  ratio={effect_ratio:.3f}  "
            f"gate_tripped={gg.tripped} rep={gg.rep_rate:.3f} "
            f"nll={gg.mean_nll:.3f}  reasons={gg.reasons}")

    # =====================================================================
    # Amendment 2 verdict
    # =====================================================================
    # Non-degenerate (admissible) scales = gate not tripped.
    nondegen = [r for r in rows if not r["gate_tripped"]]
    n_nondegen = len(nondegen)
    # PASS = >=3 sub-degenerate scales AND every non-degenerate scale has
    #        gain <= 0.3 x E(FV) = 11.8 pts (equivalently effect_ratio <= 0.3).
    enough_scales = n_nondegen >= 3
    all_within = all(r["gain_pts"] <= GAIN_BOUND_PTS for r in nondegen)
    passed = enough_scales and all_within
    verdict = "PASS" if passed else "FAIL"
    fail_reasons = []
    if not enough_scales:
        fail_reasons.append(
            f"only {n_nondegen} non-degenerate scale(s) (< 3 required)")
    over = [r for r in nondegen if r["gain_pts"] > GAIN_BOUND_PTS]
    if over:
        fail_reasons.append(
            "scales exceeding bound: " +
            ", ".join(f"c={r['scale']} gain={r['gain_pts']:+.1f}pts" for r in over))

    log(f"AMENDMENT 2 VERDICT: {verdict}  "
        f"(non-degenerate scales={n_nondegen}, bound={GAIN_BOUND_PTS:.1f}pts, "
        f"all_within={all_within})")
    if fail_reasons:
        log("  fail reasons: " + "; ".join(fail_reasons))

    out = {
        "meta": {
            "purpose": "A1 control dose-response sensitivity (plan §11 Amendment 2)",
            "model": args.model, "device": device, "dtype": args.dtype,
            "task": args.task, "edit_layer": edit_layer,
            "n_eval": len(eval_prompts), "n_calib": len(calib_prompts),
            "max_new_tokens": args.max_new_tokens, "n_boot": args.n_boot,
            "seed": args.seed, "dose_scales": DOSE_SCALES,
            "run1_matched_c": run1["control_calibration"]["bias_scalar"],
            "token_set_size": len(ctrl_token_ids),
            "token_set_matches_run1": bool(ids_match),
            "gate_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
            "E_FV_pts": E_FV_PTS, "baseline_rate": BASELINE_RATE,
            "amend2_ratio": AMEND2_RATIO, "gain_bound_pts": GAIN_BOUND_PTS,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "scales": rows,
        "verdict": {
            "amendment2": verdict, "passed": bool(passed),
            "rule": f"PASS = >=3 non-degenerate scales AND every non-degenerate "
                    f"scale has gain <= {AMEND2_RATIO} x E(FV) = "
                    f"{GAIN_BOUND_PTS:.1f} pts",
            "n_nondegenerate_scales": n_nondegen,
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
    A("# A1 anchor — control dose-response (Amendment 2)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` (edit_layer {m['edit_layer']}), device "
      f"`{m['device']}` ({m['dtype']}).  ")
    A(f"**Eval:** {m['n_eval']} held-out prompts, {m['max_new_tokens']}-token "
      f"greedy, task accuracy, {m['n_boot']} bootstrap.  ")
    A(f"**Control token set:** {m['token_set_size']} tokens "
      f"(matches run-1 saved set: {m['token_set_matches_run1']}).  ")
    A(f"**Gate refs (frozen, run-1):** rep={m['gate_refs']['rep']:.3f}, "
      f"median_len={m['gate_refs']['median_len']:.1f}, "
      f"nll={m['gate_refs']['nll']:.3f}.\n")

    A("## Dose-response sensitivity (Amendment 2)\n")
    A(f"Re-evaluation of the primary calibrated logit-bias control ONLY, at "
      f"bias scalars c in {m['dose_scales']} (c={m['dose_scales'][-1]} = run-1 "
      f"matched budget, c={m['run1_matched_c']:.4f}, for continuity). Same 150 "
      f"eval prompts, same generation, same accuracy metric, same degeneracy "
      f"gate. Baseline = {m['baseline_rate']*100:.1f}% (run 1); "
      f"E(FV) = {m['E_FV_pts']:.1f} pts (run-1 E_all). Amendment 2 bound = "
      f"0.3 x E(FV) = {m['gain_bound_pts']:.1f} pts.\n")

    A("| scale c | accuracy [95% CI] | gain vs baseline | gate | gate reasons | "
      "effect ratio vs E(FV) |")
    A("|--------:|---|---:|:---:|---|---:|")
    for r in out["scales"]:
        reasons = "; ".join(r["gate_reasons"]) if r["gate_reasons"] else "—"
        gate = "VOID (degenerate)" if r["gate_tripped"] else "ok"
        A(f"| {r['scale']:.2f} | {r['accuracy']*100:.1f}% "
          f"[{r['ci_lo']*100:.1f}, {r['ci_hi']*100:.1f}] | "
          f"{r['gain_pts']:+.1f} pts | {gate} | {reasons} | "
          f"{r['effect_ratio_vs_E_FV']:.3f} |")
    A("")
    A(f"Gate details per scale (rep / median_len / nll):")
    for r in out["scales"]:
        A(f"- c={r['scale']:.2f}: rep={r['gate_rep']:.3f}, "
          f"median_len={r['gate_median_len']:.1f}, nll={r['gate_nll']:.3f}")
    A("")

    A(f"### Verdict: **{v['amendment2']}**\n")
    A(f"Rule: {v['rule']}.  ")
    A(f"Non-degenerate (sub-gate) scales: {v['n_nondegenerate_scales']}; "
      f"all within bound: {v['all_nondegen_within_bound']}.\n")
    if v["passed"]:
        A(f"**PASS** — {v['n_nondegenerate_scales']} sub-degenerate scales, and "
          f"every non-degenerate scale reproduces <= "
          f"{m['gain_bound_pts']:.1f} pts of the {m['E_FV_pts']:.1f}-pt FV "
          f"effect. No admissible scale of the prompt-independent output-push "
          f"control reproduces the function-vector effect; the void control "
          f"cell does not unfairly favor the Genuine verdict. A1 Genuine "
          f"verdict stands.")
    else:
        A(f"**FAIL** — {'; '.join(v['fail_reasons'])}. At least one admissible "
          f"(non-degenerate) control scale reproduces > "
          f"{m['gain_bound_pts']:.1f} pts of the FV effect (or too few "
          f"admissible scales exist); Amendment 2 is NOT satisfied.")
    A("")
    A(f"Runtime: {out['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
