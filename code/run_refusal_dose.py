"""Control-only dose-response RESOLUTION run for the refusal arm.

Purpose (plan §11 Amendment 2 adjudication): the primary logit-suppression
control degenerated at full budget (rep 0.968) in the 2026-07-07 refusal run,
so the Dissolved path is closed and the verdict is Mixed *pending a finer
control dose-response*. Amendment-2 Genuine applies iff >=3 clean sub-degenerate
scales ALL have effect <= 0.3 * E_native (=0.927 drop -> bound 0.278 pts, i.e.
effect ratio <= 0.3). This driver re-evaluates the SAME primary control (the
calibrated negative logit bias on the refusal-onset token set) at a finer scale
grid on the SAME 150 eval prompts, 128 tok greedy, applying NO model-side
intervention (logit-bias only).

No model-side interventions. Reuses run-1 machinery verbatim (refusal
classifier, token set S, base bias scalar 2.4783, eval prompts, degeneracy
refs). The chosen refusal direction is loaded from run-1's staged
chosen_direction.pt (weights_only=False; our own artifact) so the direction is
byte-identical; the calib/eval split, token set, and mean position-1 logit-delta
are reconstructed deterministically from the frozen splits and verified against
run-1 before any grid point is run (frac 0.50 must reproduce rate 0.70,
effect ratio 0.259).

Scale grid: fracs {0.10,0.20,0.30,0.40,0.50,0.60,0.65} x base scalar 2.4783.
Per scale: refusal rate + 10k bootstrap 95% CI, effect ratio vs E_native=0.927,
degeneracy gate (rep/length = TRUE degeneracy voids; nll-only = flagged, does
not void, per §4 chat-model recalibration).

Also: regenerate the TF-KL sensitivity control (bias scalar 2.3410) generations
on the same 150 eval prompts and record its degeneracy gate status (run-1
quoted its 70.7-pt effect with NO recorded gate status).

Verdict line: "Genuine (Amendment-2)" iff (#clean scales >= 3) AND (every clean
scale effect ratio <= 0.3); else "Mixed".

Staged to disk (dose_stage.json) so a timed-out call can resume. Foreground.

Usage:
  .venv/bin/python run_refusal_dose.py                 # full run, mps
  .venv/bin/python run_refusal_dose.py --smoke --n 6   # quick shape check
  .venv/bin/python run_refusal_dose.py --device cpu
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
import battery as B  # noqa: E402
import refusal_direction as R  # noqa: E402
import run_refusal as RR  # noqa: E402  (ablation_position1_logit_delta, refusal_rate_with_bias)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_REPO, "tools"))
from actlib import load_model  # noqa: E402
from actlib.models import get_model_info  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RUN_DIR = os.path.join(_REPO, "runs/steering-content-audit/2026-07-07-refusal-arm")

# Frozen from run-1 (results_full.json) — used for determinism assertions.
BASE_SCALAR = 2.478271484375        # calibrated control scalar c_full
E_NATIVE = 0.9266666666666666       # E_native refusal DROP (150-prompt eval)
TFKL_SCALAR = 2.34104048833251      # TF-per-step-KL sensitivity control scalar
RUN1_TOKEN_IDS = None               # loaded from results_full.json at runtime
RUN1_EVAL_REFS = None               # rep/median_len/nll baseline refs

# Amendment-2 rule
EFFECT_RATIO_BOUND = 0.3
MIN_CLEAN_SCALES = 3

FRACS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.65]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_split(name: str):
    with open(os.path.join(RR.SPLITS, f"{name}.json")) as f:
        return [d["instruction"] for d in json.load(f)]


def load_run1():
    with open(os.path.join(RUN_DIR, "results_full.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def stage_path(outdir):
    return os.path.join(outdir, "dose_stage.json")


def load_stage(outdir):
    p = stage_path(outdir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"grid": {}, "tfkl": None, "setup": None}


def save_stage(outdir, stage):
    with open(stage_path(outdir), "w") as f:
        json.dump(stage, f, indent=2)


# ---------------------------------------------------------------------------
# Gate helper (true-degeneracy adjudication, §4 chat recalibration)
# ---------------------------------------------------------------------------

def gate_status(texts, prompts_templated, ev_rep, ev_med, ev_nll, model,
                tokenizer, device):
    gate = B.degeneracy_gate(texts, prompts_templated, ev_rep, ev_med, ev_nll,
                             model, tokenizer, device=device)
    rep_trip = gate.rep_rate > 2 * ev_rep + 0.1
    len_trip = gate.median_len < 0.5 * ev_med
    degenerate = bool(rep_trip or len_trip)      # TRUE degeneracy
    nll_only = bool(gate.tripped and not degenerate)
    return {
        "raw_tripped": bool(gate.tripped),
        "degenerate": degenerate,        # voids the cell
        "nll_only": nll_only,            # flag only, does not void
        "rep_trip": bool(rep_trip),
        "len_trip": bool(len_trip),
        "rep": gate.rep_rate,
        "median_len": gate.median_len,
        "nll": gate.mean_nll,
        "reasons": gate.reasons,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=RUN_DIR)
    args = ap.parse_args()

    if args.smoke:
        args.n = args.n if args.n != 200 else 12
        args.n_boot = 200

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    global RUN1_TOKEN_IDS, RUN1_EVAL_REFS
    run1 = load_run1()
    RUN1_TOKEN_IDS = run1["control_calibration"]["token_ids"]
    RUN1_EVAL_REFS = run1["eval_baseline_refs"]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    log(f"loading {MODEL_ID} on {args.device} {args.dtype}")
    model, tokenizer = load_model(MODEL_ID, device=args.device, dtype=dtype)
    device = next(model.parameters()).device.type
    n_layers = get_model_info(model).n_layers
    all_layers = list(range(n_layers))

    # --- reconstruct frozen splits (must match run-1 exactly) ---
    harmful_test = load_split("harmful_test")
    n = min(args.n, len(harmful_test))
    prompts = harmful_test[:n]
    n_calib = min(args.n_calib, n // 2)
    calib = prompts[:n_calib]
    eval_ = prompts[n_calib:]
    log(f"splits: n={n} calib={len(calib)} eval={len(eval_)}")

    # --- load run-1 chosen direction (our own artifact) ---
    dd = torch.load(os.path.join(outdir, "chosen_direction.pt"),
                    weights_only=False)
    r_hat = dd["r_hat"]
    L = dd["layer"]
    log(f"loaded chosen_direction.pt: L={L} raw_norm={dd['diagnostics']['raw_norm']:.4f}")
    assert L == run1["meta"]["chosen_layer"], "layer mismatch vs run-1"

    stage = load_stage(outdir)

    # --- reconstruct token set + mean position-1 logit-delta on calib ---
    ev_rep = RUN1_EVAL_REFS["rep"]
    ev_med = RUN1_EVAL_REFS["median_len"]
    ev_nll = RUN1_EVAL_REFS["nll"]
    prompts_templated = [R.build_chat_prompt(tokenizer, p) for p in eval_]

    log("reconstructing position-1 logit-delta token set on calib")
    mean_delta = RR.ablation_position1_logit_delta(
        model, tokenizer, calib, r_hat, all_layers, device=device)
    ctrl_token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)

    # Determinism check A: token set matches run-1 (skip if smoke -> fewer calib).
    tokenset_match = (ctrl_token_ids == RUN1_TOKEN_IDS)
    if not args.smoke:
        if not tokenset_match:
            log("WARNING: token set does not match run-1 exactly")
            log(f"  reconstructed[:10]={ctrl_token_ids[:10]}")
            log(f"  run1[:10]        ={RUN1_TOKEN_IDS[:10]}")
        else:
            log(f"determinism A OK: token set matches run-1 (size {len(ctrl_token_ids)})")
    tid = torch.tensor(ctrl_token_ids)
    dvals = mean_delta[tid]

    # --- baseline on eval (for effect = base_rate - cond_rate) ---
    # Reuse run-1's eval baseline hits if present for determinism; else recompute.
    if stage.get("setup") and "base_hits" in stage["setup"]:
        base_hits = stage["setup"]["base_hits"]
        base_rate = float(np.mean(base_hits))
        log(f"reusing staged eval baseline: rate={base_rate:.3f}")
    else:
        log("generating eval baseline (unsteered, for effect denominator)")
        base_texts = [R.base_generate_chat(
            model, tokenizer, R.build_chat_prompt(tokenizer, p),
            max_new_tokens=args.tokens, device=device) for p in eval_]
        base_hits = [int(R.is_refusal(t)) for t in base_texts]
        base_rate = float(np.mean(base_hits))
        stage["setup"] = {
            "base_hits": base_hits, "base_rate": base_rate,
            "token_ids": ctrl_token_ids, "tokenset_match_run1": bool(tokenset_match),
            "n_calib": len(calib), "n_eval": len(eval_),
        }
        save_stage(outdir, stage)
    log(f"eval baseline refusal rate = {base_rate:.4f} "
        f"(run-1 was {run1['rates']['baseline']['rate']:.4f})")

    # --- determinism check B: frac 0.50 must reproduce run-1 (rate 0.70, ratio 0.259) ---
    def run_scale(frac):
        key = f"{frac:.2f}"
        if key in stage["grid"]:
            log(f"  frac {frac:.2f}: cached")
            return stage["grid"][key]
        scalar = frac * BASE_SCALAR
        bias = scalar * dvals
        rate, texts = RR.refusal_rate_with_bias(
            model, tokenizer, eval_, ctrl_token_ids, bias, args.tokens, device)
        hits = [int(R.is_refusal(t)) for t in texts]
        rate = float(np.mean(hits))
        effect = base_rate - rate
        ratio = effect / max(E_NATIVE, 1e-9)
        g = gate_status(texts, prompts_templated, ev_rep, ev_med, ev_nll,
                        model, tokenizer, device)
        rb, rlo, rhi = B.bootstrap_rate_ci(hits, args.n_boot)
        row = {
            "frac": frac, "bias_scalar": scalar, "rate": rate,
            "rate_ci_lo": rlo, "rate_ci_hi": rhi,
            "effect": effect, "effect_over_native": ratio,
            "gate": g, "hits": hits,
            "sample": texts[0][:200] if texts else "",
        }
        stage["grid"][key] = row
        save_stage(outdir, stage)
        log(f"  frac {frac:.2f} (scalar {scalar:.4f}): rate={rate:.3f} "
            f"ratio={ratio:.3f} degenerate={g['degenerate']} "
            f"(nll_only={g['nll_only']}) rep={g['rep']:.3f} med={g['median_len']:.1f}")
        return row

    log("determinism B: reproducing run-1 frac 0.50 (expect rate 0.70, ratio 0.259)")
    r050 = run_scale(0.50)
    det_b_ok = abs(r050["rate"] - 0.70) < 1e-6 and abs(r050["effect_over_native"] - 0.2589928057553957) < 1e-6
    if not args.smoke:
        if det_b_ok:
            log("determinism B OK: frac 0.50 reproduces rate 0.70, ratio 0.259")
        else:
            log(f"WARNING: determinism B mismatch: rate={r050['rate']:.4f} "
                f"ratio={r050['effect_over_native']:.4f} (expected 0.70 / 0.2590)")

    # --- run the full fine grid ---
    log(f"running fine grid fracs={FRACS}")
    for frac in FRACS:
        run_scale(frac)

    grid_rows = [stage["grid"][f"{frac:.2f}"] for frac in FRACS]

    # --- TF-KL sensitivity control: regenerate + gate ---
    if stage.get("tfkl") is None:
        log(f"regenerating TF-KL sensitivity control (scalar {TFKL_SCALAR:.4f})")
        tfkl_bias = TFKL_SCALAR * dvals
        tfkl_rate, tfkl_texts = RR.refusal_rate_with_bias(
            model, tokenizer, eval_, ctrl_token_ids, tfkl_bias, args.tokens, device)
        tfkl_hits = [int(R.is_refusal(t)) for t in tfkl_texts]
        tfkl_rate = float(np.mean(tfkl_hits))
        tfkl_g = gate_status(tfkl_texts, prompts_templated, ev_rep, ev_med,
                             ev_nll, model, tokenizer, device)
        trb, trlo, trhi = B.bootstrap_rate_ci(tfkl_hits, args.n_boot)
        stage["tfkl"] = {
            "bias_scalar": TFKL_SCALAR, "rate": tfkl_rate,
            "rate_ci_lo": trlo, "rate_ci_hi": trhi,
            "effect": base_rate - tfkl_rate,
            "effect_over_native": (base_rate - tfkl_rate) / max(E_NATIVE, 1e-9),
            "gate": tfkl_g, "sample": tfkl_texts[0][:200] if tfkl_texts else "",
            "run1_reported_effect": run1["tfkl_control"]["effect"],
            "run1_reported_rate": run1["tfkl_control"]["refusal_rate"],
        }
        save_stage(outdir, stage)
        log(f"TF-KL: rate={tfkl_rate:.3f} effect={base_rate - tfkl_rate:.3f} "
            f"degenerate={tfkl_g['degenerate']} (nll_only={tfkl_g['nll_only']}) "
            f"rep={tfkl_g['rep']:.3f} med={tfkl_g['median_len']:.1f}")
    tfkl = stage["tfkl"]

    # --- adjudicate (Amendment 2) ---
    clean_rows = [r for r in grid_rows if not r["gate"]["degenerate"]]
    n_clean = len(clean_rows)
    all_clean_le_bound = all(r["effect_over_native"] <= EFFECT_RATIO_BOUND
                             for r in clean_rows)
    passes = (n_clean >= MIN_CLEAN_SCALES) and all_clean_le_bound
    verdict = "Genuine (Amendment-2)" if passes else "Mixed"
    max_clean_ratio = max((r["effect_over_native"] for r in clean_rows), default=None)

    runtime = time.time() - t0
    result = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": "control-only dose-response resolution of refusal-arm "
                       "verdict (plan §11 Amendment 2)",
            "model": MODEL_ID, "n_layers": n_layers, "chosen_layer": L,
            "device": device, "dtype": args.dtype,
            "n_total": n, "n_calib": len(calib), "n_eval": len(eval_),
            "max_new_tokens": args.tokens, "n_boot": args.n_boot,
            "base_scalar": BASE_SCALAR, "E_native": E_NATIVE,
            "fracs": FRACS,
            "no_model_side_intervention": True,
        },
        "determinism_checks": {
            "token_set_matches_run1": bool(tokenset_match),
            "token_set_size": len(ctrl_token_ids),
            "eval_baseline_rate": base_rate,
            "run1_eval_baseline_rate": run1["rates"]["baseline"]["rate"],
            "frac_0.50_rate": r050["rate"],
            "frac_0.50_ratio": r050["effect_over_native"],
            "frac_0.50_matches_run1": bool(det_b_ok),
            "run1_frac_0.50_rate": 0.70,
            "run1_frac_0.50_ratio": 0.2589928057553957,
        },
        "eval_baseline_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
        "control_token_set": {
            "size": len(ctrl_token_ids), "token_ids": ctrl_token_ids,
            "top_tokens": [tokenizer.decode([i]) for i in ctrl_token_ids[:12]],
        },
        "grid": [{k: v for k, v in r.items() if k != "hits"} for r in grid_rows],
        "tfkl_control": {k: v for k, v in tfkl.items()},
        "adjudication": {
            "rule": "Genuine (Amendment-2) iff >=3 clean sub-degenerate scales "
                    "AND every clean scale effect_ratio <= 0.3 (=0.3*E_native, "
                    "bound 0.278 pts drop); else Mixed.",
            "effect_ratio_bound": EFFECT_RATIO_BOUND,
            "min_clean_scales": MIN_CLEAN_SCALES,
            "n_clean_scales": n_clean,
            "clean_fracs": [r["frac"] for r in clean_rows],
            "max_clean_effect_ratio": max_clean_ratio,
            "all_clean_le_bound": bool(all_clean_le_bound),
            "passes_amendment2": bool(passes),
            "verdict": verdict,
        },
        "runtime_sec": runtime,
    }

    json_path = os.path.join(outdir, "dose_response_fine.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    md_path = os.path.join(outdir, "dose_response_fine.md")
    write_md(result, md_path)
    log(f"done in {runtime:.0f}s; verdict={verdict}")
    log(f"wrote {json_path}")
    log(f"wrote {md_path}")


def write_md(r, path):
    m = r["meta"]; adj = r["adjudication"]; dc = r["determinism_checks"]
    tf = r["tfkl_control"]
    lines = []
    A = lines.append
    A("# Refusal arm — control-only dose-response resolution "
      "(Amendment-2 adjudication)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}`, L={m['chosen_layer']}, device `{m['device']}`, "
      f"{m['dtype']}. Control-only re-eval — **no model-side intervention** "
      f"(calibrated negative logit bias on the refusal-onset token set only).  ")
    A(f"**Eval:** SAME {m['n_eval']} eval prompts (harmful_test[{m['n_calib']}:"
      f"{m['n_total']}]), {m['max_new_tokens']} tok greedy, {m['n_boot']} "
      f"bootstrap resamples. Base control scalar c={m['base_scalar']:.4f}; "
      f"E_native={m['E_native']*100:.1f} pts drop.\n")

    A("## Purpose\n")
    A("Run-1's primary control degenerated at full budget (rep 0.968), closing "
      "the Dissolved path (a degenerate control can never reproduce a behavior). "
      "Per plan §3 + §11 Amendment 2 the verdict is **Mixed pending a finer "
      "control dose-response**: Amendment-2 **Genuine** applies iff **>=3 clean "
      "(sub-degenerate) scales** ALL have effect **<= 0.3 x E_native** "
      "(ratio <= 0.3). A single clean scale > 0.3 keeps **Mixed**. Run-1's "
      "coarse grid found only 2 clean scales (both 0.259) with degeneracy onset "
      "between frac 0.5 and 0.75 — the grid, not the phenomenon, failed. This "
      "run uses a finer grid to seek >=3 clean scales.\n")

    A("## Determinism vs run-1\n")
    A(f"- Refusal direction: loaded byte-identical from run-1 `chosen_direction.pt` "
      f"(L={m['chosen_layer']}).")
    A(f"- Control token set S matches run-1 exactly: "
      f"**{dc['token_set_matches_run1']}** (size {dc['token_set_size']}).")
    A(f"- Eval baseline refusal rate: {dc['eval_baseline_rate']*100:.1f}% "
      f"(run-1 {dc['run1_eval_baseline_rate']*100:.1f}%).")
    A(f"- frac 0.50 reproduction: rate {dc['frac_0.50_rate']*100:.1f}% "
      f"(run-1 {dc['run1_frac_0.50_rate']*100:.1f}%), ratio "
      f"{dc['frac_0.50_ratio']:.3f} (run-1 {dc['run1_frac_0.50_ratio']:.3f}) — "
      f"matches = **{dc['frac_0.50_matches_run1']}**.\n")

    A("## Fine dose-response grid\n")
    er = r["eval_baseline_refs"]
    A(f"Gate refs (run-1 eval baseline): rep={er['rep']:.3f}, "
      f"median_len={er['median_len']:.1f}, nll={er['nll']:.3f}. TRUE degeneracy "
      f"= rep > {2*er['rep']+0.1:.3f} OR median_len < {0.5*er['median_len']:.1f}; "
      f"an nll-only trip on coherent text is flagged, not voided (§4).\n")
    A("| frac | bias scalar | rate | rate 95% CI | effect ratio (/E_native) | gate |")
    A("|-----:|------------:|-----:|:-----------:|-------------------------:|:----:|")
    for row in r["grid"]:
        g = row["gate"]
        if g["degenerate"]:
            gate_s = "**DEGENERATE**"
        elif g["nll_only"]:
            gate_s = "clean (nll-only flag)"
        else:
            gate_s = "clean"
        A(f"| {row['frac']:.2f} | {row['bias_scalar']:.4f} | "
          f"{row['rate']*100:.1f}% | "
          f"[{row['rate_ci_lo']*100:.1f}, {row['rate_ci_hi']*100:.1f}] | "
          f"{row['effect_over_native']:.3f} | {gate_s} |")
    A("")
    A("Per-scale gate detail (rep / median_len / mean_nll):\n")
    A("| frac | rep | median_len | mean_nll | raw gate | rep_trip | len_trip | nll_only |")
    A("|-----:|----:|-----------:|---------:|:--------:|:--------:|:--------:|:--------:|")
    for row in r["grid"]:
        g = row["gate"]
        A(f"| {row['frac']:.2f} | {g['rep']:.3f} | {g['median_len']:.1f} | "
          f"{g['nll']:.3f} | {'trip' if g['raw_tripped'] else 'ok'} | "
          f"{'Y' if g['rep_trip'] else '-'} | {'Y' if g['len_trip'] else '-'} | "
          f"{'Y' if g['nll_only'] else '-'} |")
    A("")

    A("## TF-KL sensitivity control gate status\n")
    g = tf["gate"]
    A(f"Regenerated the TF-per-step-KL sensitivity control (bias scalar "
      f"{tf['bias_scalar']:.4f}) on the same {m['n_eval']} eval prompts to record "
      f"the gate status run-1 omitted.\n")
    A(f"- refusal rate = {tf['rate']*100:.1f}% "
      f"[{tf['rate_ci_lo']*100:.1f}, {tf['rate_ci_hi']*100:.1f}]; effect = "
      f"{tf['effect']*100:.1f} pts (ratio {tf['effect_over_native']:.3f}); "
      f"run-1 reported {tf['run1_reported_effect']*100:.1f} pts at rate "
      f"{tf['run1_reported_rate']*100:.1f}%.")
    A(f"- gate: rep={g['rep']:.3f}, median_len={g['median_len']:.1f}, "
      f"nll={g['nll']:.3f} -> **{'DEGENERATE' if g['degenerate'] else ('clean (nll-only flag)' if g['nll_only'] else 'clean')}** "
      f"(rep_trip={g['rep_trip']}, len_trip={g['len_trip']}).\n")

    A("## Adjudication (plan §11 Amendment 2)\n")
    A(f"- Rule: {adj['rule']}")
    A(f"- Clean (sub-degenerate) scales: **{adj['n_clean_scales']}** "
      f"(fracs {adj['clean_fracs']}); need >= {adj['min_clean_scales']}.")
    A(f"- Max effect ratio among clean scales: "
      f"**{adj['max_clean_effect_ratio']:.3f}** (bound {adj['effect_ratio_bound']}); "
      f"all clean <= bound = **{adj['all_clean_le_bound']}**.")
    A(f"- Amendment-2 passes = **{adj['passes_amendment2']}**.\n")
    A(f"## VERDICT: **{adj['verdict']}**\n")
    A(f"Runtime: {r['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
