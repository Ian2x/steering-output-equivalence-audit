"""Task-vec anchor — STEELMAN input-conditional null (plan §11 Amendment 7).

CONFIRMATION of the FV steelman (Amendment 6b SURVIVED). Task-vec is the second
input-dependent Genuine-pole exemplar AND a *stronger* test of HOLE 1: it is a
single-position REPLACE-patch (kappa_native=1.0, prompt-injection family, Hendel
2310.15916), so if even a one-token write's effect cannot be reproduced by a
matched-budget input-conditional output push, the null-class-mismatch rebuttal
generalizes beyond FV's every-position ADD.

This is a SIBLING of run_a1_steelman.py. The machinery (rank-k fit ladder,
budget bisection, per-prompt LogitBias generation, gate/bootstrap/decision, all
Amendment-4 validity guards) is IMPORTED verbatim from run_a1_steelman.py. The
ONLY substantive change is the **steered field** target Y: replace the FV ADD
patch (`fv_teacher_forced_delta_on_S`, add_vector at every continuation position)
with the task-vec **REPLACE** patch computed by mirroring
`taskvec.py::TaskVecMethod._forward_write_positions` EXACTLY (REPLACE theta at
L12 the final query token, teacher-forced on fixed unsteered continuations, per-
position logit delta on S). All frozen anchors are loaded from the task-vec run:
  runs/steering-content-audit/2026-07-07-taskvec-arm/results_full.json
  runs/steering-content-audit/2026-07-07-taskvec-arm/stage_antonym_full.json

Null form (IDENTICAL to Amendment 4/6/6b):
    b(p) = g * M_k * phi_tilde(p)   restricted to token set S
    phi(p)       = BASE last-layer resid_post @ final prompt token (no hooks)
    phi_tilde(p) = [ (phi - mu)/sigma ; 1 ]   (mu, sigma from CALIB only)
    M_k          = rank-k truncation of the ridge-fit map, written only to |S|
    g            = scalar budget knob, bisected AFTER fitting to hit B*.
Rank ladder k in {0, 4, 16, full}. k=0 == best-constant-bias == static-null
analogue. k=full is a labeled UPPER ANCHOR (never calls Dissolved).

DENOMINATOR NOTE (differs from FV steelman): the frozen task-vec rho used
E_native = E_first (the KV-baked single write), NOT E_all. This driver therefore
uses E_first as the bootstrap denominator (regen deterministically, greedy,
seed=13) so the k=0 rung reproduces the frozen static-null rho = -0.1408 EXACTLY.

Decision (plan Amendment 7; CI bounds ONLY; identical to Amendments 4/6/6b):
    SURVIVES  <=> rho_hi <= 0.3 for ALL non-degenerate k <= 16.
    ARTIFACT  <=> clean rho_lo >= 0.9 at SOME k <= 16.
    k=full = upper anchor, never calls Dissolved.

Mandatory harness gate (STOP condition, Amendment 7): the k=0 rung MUST
reproduce the frozen static-null rho = -0.1408 within bootstrap noise. If it does
NOT, the steered-field/patch config is wrong -> STOP and debug, do NOT certify.

Outputs steelman_dose_f*.{json,md} (or steelman.{json,md} at B*) APPEND-ONLY
into the taskvec-arm run dir. NEVER touches results_full.json / stage_*.json.

Usage:
  source .venv/bin/activate
  # full task-vec job (pythia-2.8b, MPS, bf16), single budget B*:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_a1_steelman_taskvec.py
  # a budget-frac point (writes steelman_dose_f0.10.json):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_a1_steelman_taskvec.py \
      --budget-kl 0.7380771484375 --dose-frac 0.10
  # smoke (pythia-160m, exercises the FULL code path):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_a1_steelman_taskvec.py --smoke
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
import taskvec as TV         # noqa: E402
from run_taskvec import answer_hit  # noqa: E402  (same \b word-boundary hit)

# Import the shared steelman machinery VERBATIM (no re-implementation).
import run_a1_steelman as ST  # noqa: E402
from run_a1_steelman import (  # noqa: E402
    fit_rank_ladder,
    prompt_bias_matrix,
    teacher_forced_stepkl_biased_perprompt,
    bisect_g_for_budget,
    per_prompt_kl_spread,
    BUDGET_TOL,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "external", "function_vectors", "dataset_files", "abstractive")

RUNDIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "runs", "steering-content-audit", "2026-07-07-taskvec-arm")


# ===========================================================================
# Fit target Y (calib): task-vec REPLACE-patched-minus-base teacher-forced
# per-position logit delta on token set S. Mirrors
# taskvec.TaskVecMethod._forward_write_positions / teacher_forced_stepkl EXACTLY
# (same L12 REPLACE at every continuation-predicting position, same fixed
# unsteered continuation, single forward), but KEEPS the per-position delta on S
# instead of collapsing to KL. This is the task-vec analogue of the FV
# fv_teacher_forced_delta_on_S (which used the FV ADD patch).
# ===========================================================================

def taskvec_teacher_forced_delta_on_S(
        meth: "TV.TaskVecMethod", tokenizer, prompts, continuation_ids,
        token_ids, device="cpu"):
    """Per-prompt mean (over continuation positions) of (steered - base) logits
    restricted to ``token_ids`` (== S). Returns Y ndarray [n_prompts, |S|].

    Steered == the task-vec E_all patch config that DEFINES B*: theta written by
    REPLACE (meth.op) at EVERY continuation-producing position via
    meth._forward_write_positions, teacher-forced on the fixed unsteered
    continuation — the exact configuration of meth.teacher_forced_stepkl (whose
    mean-over-(positions x prompts) KL is the frozen B* = 7.3808).
    """
    v = meth.theta.to(device)
    tid = torch.tensor(list(token_ids), device=device)
    rows = []
    for prompt, cont in zip(prompts, continuation_ids):
        if len(cont) == 0:
            rows.append(np.zeros(len(token_ids), dtype=np.float64))
            continue
        # Build the teacher-forced sequence EXACTLY as taskvec.teacher_forced_stepkl
        # does (prompt ids ++ continuation ids), and the same pred_positions.
        p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        P = p_ids.shape[1]
        c_ids = torch.tensor([list(cont)], device=device)
        full = torch.cat([p_ids, c_ids], dim=1)
        n = len(cont)
        pred_positions = list(range(P - 1, P - 1 + n))
        with torch.no_grad():
            base_logits = meth.model(full).logits[0]          # [seq, vocab]
        # REPLACE theta at every pred position — taskvec._forward_write_positions.
        steered_logits = meth._forward_write_positions(full, v, pred_positions)
        deltas = []
        for pos in pred_positions:
            d = (steered_logits[pos, tid] - base_logits[pos, tid])
            deltas.append(d.to("cpu").float())
        rows.append(torch.stack(deltas).mean(0).numpy().astype(np.float64))
    return np.stack(rows)                                      # [n, |S|]


def base_phi(model, tokenizer, prompts, phi_layer, device="cpu", batch_size=16):
    """BASE resid_post at ``phi_layer``, last token -> [n, resid]. No hooks.
    (Re-exported here for clarity; identical to run_a1_steelman.base_phi.)"""
    return ST.base_phi(model, tokenizer, prompts, phi_layer, device=device,
                       batch_size=batch_size)


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-2.8b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16", choices=["float32", "bf16"])
    ap.add_argument("--task", default="antonym")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--n-mean", type=int, default=100,
                    help="ICL demo contexts for theta (only used if no cache)")
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--layer", type=int, default=None,
                    help="task-vec inject layer L (default = frozen chosen_layer)")
    ap.add_argument("--op", default=None,
                    help="replace|add (default = frozen chosen_op)")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--phi-layer", type=int, default=None,
                    help="resid_post layer for phi (default = n_layers-1)")
    ap.add_argument("--ranks", default="0,4,16,full")
    ap.add_argument("--ridge-frac", type=float, default=1e-2)
    ap.add_argument("--budget-kl", type=float, default=None,
                    help="explicit budget-match target KL. Default: frozen B* "
                         "(2.8b full run) or, in --smoke, an auto reachable "
                         "target (budget-frac x the per-k KL ceiling).")
    ap.add_argument("--budget-frac", type=float, default=0.7,
                    help="smoke-only: fraction of the (saturating) achievable "
                         "bias-KL ceiling to target so the g-bisection converges "
                         "on the tiny model (full code-path exercise).")
    ap.add_argument("--dose-frac", type=float, default=None,
                    help="label only: f in steelman_dose_f{f}.json (the fraction "
                         "of B* this --budget-kl represents). If set, output "
                         "filename is steelman_dose_f{dose-frac}.{json,md}.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.smoke:
        # Smoke: pythia-160m, tiny n, full code path (spec §smoke path).
        args.model = "EleutherAI/pythia-160m"
        args.device = "cpu"
        args.dtype = "float32"
        args.n_eval = 20
        args.n_calib = 10
        args.n_mean = 20
        args.n_boot = 200
        args.ranks = "0,4,full"       # 16 ~= full at n_calib=10

    ranks = [("full" if r.strip() == "full" else int(r.strip()))
             for r in args.ranks.split(",") if r.strip()]

    outdir = args.outdir or RUNDIR_DEFAULT
    os.makedirs(outdir, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log(f"STEELMAN-TASKVEC (Amendment 7)  model={args.model} "
        f"device={args.device} dtype={args.dtype} ranks={ranks} tag={tag}")

    # =====================================================================
    # Load + assert frozen task-vec anchors (VERIFY at load; Amendment 7).
    # =====================================================================
    rf = os.path.join(outdir, "results_full.json")
    stage_path = os.path.join(outdir, f"stage_{args.task}_full.json")
    B_STAR = None
    stored_ids = None
    frozen_rho = None
    frozen_gate_refs = None
    frozen_effect = None
    chosen_layer_frozen = None
    chosen_op_frozen = None
    if os.path.exists(rf):
        RF = json.load(open(rf))
        cc = RF["control_calibration"]
        B_STAR = float(cc["B_star_target_kl"])
        stored_ids = cc["token_ids"]
        frozen_rho = RF["rho"]
        frozen_gate_refs = RF["eval_baseline_refs"]
        frozen_effect = RF["effect"]
        chosen_layer_frozen = RF["meta"]["chosen_layer"]
        chosen_op_frozen = RF["meta"]["chosen_op"]
        log(f"FROZEN anchors: B*={B_STAR}  |S|={len(stored_ids)}  "
            f"rho={frozen_rho['point']:.6f} "
            f"[{frozen_rho['ci_lo']:.6f},{frozen_rho['ci_hi']:.6f}]  "
            f"L={chosen_layer_frozen} op={chosen_op_frozen}")
        log(f"FROZEN gate refs: rep={frozen_gate_refs['rep']:.6f} "
            f"median_len={frozen_gate_refs['median_len']} "
            f"nll={frozen_gate_refs['nll']:.6f}")
        # Hard-assert the Amendment-7 anchor values (defend against silent drift).
        assert abs(B_STAR - 7.380771484375) < 1e-9, \
            f"frozen B* {B_STAR} != Amendment-7 7.380771484375"
        assert len(stored_ids) == 100, f"|S_frozen|={len(stored_ids)} != 100"
        assert abs(frozen_rho["point"] - (-0.1408450704225352)) < 1e-9, \
            f"frozen rho point {frozen_rho['point']} != -0.1408450704225352"
        assert chosen_layer_frozen == 12 and chosen_op_frozen == "replace", \
            f"frozen (L,op)=({chosen_layer_frozen},{chosen_op_frozen}) != (12,replace)"
    elif not args.smoke:
        raise SystemExit(f"missing frozen anchor file {rf} — cannot run FULL")

    # Static-null rho anchor for the k=0 STOP-condition determinism check.
    if frozen_rho is not None:
        STATIC_NULL_RHO_POINT = float(frozen_rho["point"])
        STATIC_NULL_RHO_LO = float(frozen_rho["ci_lo"])
        STATIC_NULL_RHO_HI = float(frozen_rho["ci_hi"])
    else:  # smoke without anchor file: use the frozen constants as a placeholder.
        STATIC_NULL_RHO_POINT = -0.1408450704225352
        STATIC_NULL_RHO_LO = -0.25
        STATIC_NULL_RHO_HI = -0.06097560975609757

    # --- model ---
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tok = B.load_model(args.model, device=args.device, dtype=_dtype)
    device = args.device
    cfg = FV.neox_config(model)
    n_layers = cfg["n_layers"]
    phi_layer = args.phi_layer if args.phi_layer is not None else n_layers - 1
    # inject layer: frozen L12 on the full model; on the tiny smoke model L12 is
    # out of range (12-layer 160m), so clamp to a valid mid layer and extract a
    # fresh (correctly-dimensioned) theta there — the frozen 2560-d cache does
    # not fit the 768-d smoke resid anyway.
    if args.layer is not None:
        inject_layer = args.layer
    elif args.smoke:
        inject_layer = min(6, n_layers - 2)
    elif chosen_layer_frozen is not None:
        inject_layer = chosen_layer_frozen
    else:
        inject_layer = 12
    inject_op = args.op if args.op is not None else (
        chosen_op_frozen if chosen_op_frozen is not None else "replace")
    log(f"config: layers={n_layers} heads={cfg['n_heads']} "
        f"resid={cfg['resid_dim']} phi_layer={phi_layer} "
        f"inject_layer={inject_layer} inject_op={inject_op}")

    # --- dataset split (IDENTICAL to run_taskvec / run_a1) ---
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

    # =====================================================================
    # theta: load frozen cache from stage_*.json if present, else extract.
    # (Full run MUST use the frozen theta so the steered field == the audited
    # arm's field; smoke extracts fresh on the tiny model.)
    # =====================================================================
    theta = None
    # Full run: load the frozen theta (audited task vector). Smoke: the frozen
    # cache is 2560-d (2.8b) and cannot fit the 768-d smoke resid, so skip it and
    # extract fresh below.
    if os.path.exists(stage_path) and not args.smoke:
        S1 = json.load(open(stage_path))
        key = str(inject_layer)
        if key in S1.get("theta", {}):
            theta = torch.tensor(S1["theta"][key], dtype=torch.float32)
            log(f"theta: loaded frozen L={inject_layer} from {os.path.basename(stage_path)} "
                f"(norm {theta.norm().item():.4f})")
    if theta is None:
        if not args.smoke:
            raise SystemExit(
                f"missing frozen theta[L={inject_layer}] in {stage_path} — "
                f"cannot run FULL without the audited task vector")
        log(f"theta: extracting fresh at L={inject_layer} ({args.n_mean} ctx, smoke)...")
        clean = FV.sample_icl_prompts(train_pool, args.n_mean, args.n_shots,
                                      seed=args.seed + 2, shuffle_labels=False)
        theta = TV.mean_task_vector(model, tok, clean, inject_layer,
                                    device=device, log=log)
    meth = TV.TaskVecMethod(model, tok, inject_layer, theta, op=inject_op,
                            device=device, max_new_tokens=args.max_new_tokens)
    log(f"  theta norm={meth.norm:.3f}  inject_layer={inject_layer} op={inject_op}")
    if not args.smoke:
        assert abs(meth.norm - 66.496337890625) < 1e-2, \
            f"theta norm {meth.norm} != frozen 66.4963 — wrong task vector"

    # =====================================================================
    # Token set S: reconstruct identically to run_taskvec (position-1 mean delta
    # on calib -> discover_token_set 0.90 cap 100). ASSERT == stored 100.
    # =====================================================================
    log("S: position-1 mean logit delta on calib (task-vec REPLACE @ final token)...")
    mean_delta = meth.position1_logit_delta(calib_prompts)
    S = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    log(f"  |S|={len(S)}  top={[tok.decode([i]) for i in S[:8]]}")
    s_matches_stored = None
    if stored_ids is not None:
        s_matches_stored = (S == stored_ids)
        log(f"  S matches stored 100-token set: {s_matches_stored}")
        if not args.smoke:
            assert s_matches_stored, (
                "token set S != frozen results_full.json control_calibration."
                "token_ids; split/theta reconstruction diverged — ABORT")

    # =====================================================================
    # phi (BASE resid_post @ last token) on calib + eval; standardize by CALIB.
    # (IDENTICAL to run_a1_steelman: null form phi = base last-layer resid_post.)
    # =====================================================================
    log("phi: base last-layer resid_post @ final token (calib + eval)...")
    phi_calib = base_phi(model, tok, calib_prompts, phi_layer, device=device)
    phi_eval = base_phi(model, tok, eval_prompts, phi_layer, device=device)
    mu = phi_calib.mean(axis=0)
    sigma = phi_calib.std(axis=0) + 1e-6
    zc = (phi_calib - mu) / sigma                          # [n_calib, d]
    ze = (phi_eval - mu) / sigma                           # [n_eval, d]
    phi_tilde_calib = np.concatenate(
        [zc, np.ones((zc.shape[0], 1))], axis=1)           # [n_calib, d+1]

    # =====================================================================
    # Fit target Y (calib): task-vec REPLACE teacher-forced per-position delta on S.
    # =====================================================================
    log("Y: unsteered calib continuations (teacher-forcing base)...")
    calib_cont_ids = [B.base_generate_ids(model, tok, p, args.max_new_tokens,
                                          device) for p in calib_prompts]
    log("Y: task-vec REPLACE teacher-forced (steered-base) per-position delta on S...")
    Y = taskvec_teacher_forced_delta_on_S(
        meth, tok, calib_prompts, calib_cont_ids, S, device=device)
    log(f"  Y shape={Y.shape}  |Y| mean={np.abs(Y).mean():.4f}")

    # --- structural no-leakage assert (guard c) — IDENTICAL to run_a1_steelman ---
    import inspect
    _fit_params = set(inspect.signature(fit_rank_ladder).parameters)
    _forbidden = {"model", "meth", "fvm", "fv_vec", "theta", "golds",
                  "eval_prompts", "eval_golds", "tokenizer", "tok"}
    leak = _fit_params & _forbidden
    assert not leak, f"LEAKAGE: fit_rank_ladder exposes forbidden params {leak}"
    log(f"  no-leakage: fit_rank_ladder params={sorted(_fit_params)} "
        f"(forbidden intersect empty: {not leak})")

    # =====================================================================
    # Fit rank ladder (pure, deterministic; shared code).
    # =====================================================================
    log(f"FIT: ridge + SVD ladder (ridge_frac={args.ridge_frac})...")
    fit = fit_rank_ladder(phi_tilde_calib, Y, ranks, args.ridge_frac)
    log(f"  lambda={fit['lam']:.4g}  eff_rank={fit['eff_rank']}  "
        f"top-singular={fit['singular'][:5]}")

    # =====================================================================
    # Budget-match target B*. FULL: frozen B_STAR (7.3808), recomputing the
    # task-vec's own teacher-forced per-step KL as a sanity check that the frozen
    # anchor still holds on this machine. SMOKE: reachable target on 160m.
    # =====================================================================
    log("BUDGET: recompute task-vec own teacher-forced per-step KL (anchor check)...")
    tv_own_kl = meth.teacher_forced_stepkl(calib_prompts, calib_cont_ids)
    log(f"  task-vec own TF per-step KL = {tv_own_kl:.5f}  "
        f"(frozen B*={B_STAR if B_STAR else float('nan'):.5f})")
    budget_note = ""
    if args.budget_kl is not None:
        budget_target = float(args.budget_kl)
        budget_note = f"explicit --budget-kl={budget_target}"
    elif args.smoke:
        sat = []
        for k in ranks:
            b0k, Wk = fit["maps"][k]
            unit_k = prompt_bias_matrix(zc, b0k, Wk)
            sat.append(teacher_forced_stepkl_biased_perprompt(
                model, tok, calib_prompts, calib_cont_ids, S, 1e6 * unit_k,
                device=device))
        ceil_kl = float(min(sat))
        budget_target = args.budget_frac * ceil_kl
        budget_note = (f"SMOKE auto: {args.budget_frac} x min-per-k KL ceiling "
                       f"{ceil_kl:.5f} (frozen B* {B_STAR} unreachable on 160m)")
        log(f"  smoke KL ceilings per-k={[round(x, 5) for x in sat]} -> "
            f"target={budget_target:.5f}")
    else:
        budget_target = B_STAR
        budget_note = f"frozen B* = {B_STAR}"
        anchor_rel = abs(tv_own_kl - B_STAR) / B_STAR
        log(f"  frozen-anchor check: |TV_KL - B*|/B* = {anchor_rel:.4f}")
        if anchor_rel > 0.02:
            log(f"  WARNING: task-vec own KL {tv_own_kl:.5f} deviates >2% from "
                f"frozen B* {B_STAR:.5f}; split/theta reconstruction may have drifted.")
    log(f"BUDGET target = {budget_target:.5f}  ({budget_note})")

    # =====================================================================
    # Baseline + E_first (native) bootstrap denominators. run-1 hit arrays are
    # NOT persisted as hits -> deterministically regen once (greedy), matching
    # run_taskvec's construction. DENOMINATOR = E_first (native single write),
    # NOT E_all — this is the frozen task-vec rho convention (verified: reproduces
    # rho=-0.1408 exactly). E_all is regenerated only as a diagnostic denom option.
    # =====================================================================
    log("DENOM: baseline hits on eval (greedy, regen for bootstrap denom)...")
    base_texts = [TV.base_generate(model, tok, p, args.max_new_tokens, device)
                  for p in eval_prompts]
    base_hits = [int(answer_hit(t, g)) for t, g in zip(base_texts, eval_golds)]
    log(f"  baseline acc={np.mean(base_hits)*100:.1f}%")
    log("DENOM: task-vec E_first (native single write, KV-baked) hits on eval...")
    first_texts = [meth.generate(p, "first") for p in eval_prompts]
    first_hits = [int(answer_hit(t, g)) for t, g in zip(first_texts, eval_golds)]
    log(f"  E_first (native) acc={np.mean(first_hits)*100:.1f}%  "
        f"(regen denom; frozen E_first was 54.0%)")

    # eval gate baseline refs from the freshly-regenerated baseline (matches
    # run_taskvec: rep/median_len/nll of the baseline eval texts).
    ev_rep = float(np.mean([B.three_gram_rep_rate(t, tok) for t in base_texts]))
    ev_med = B.median_len_tokens(base_texts, tok)
    ev_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                            for p, t in zip(eval_prompts, base_texts)]))
    log(f"  gate refs: rep={ev_rep:.5f} median_len={ev_med} nll={ev_nll:.5f}")

    # =====================================================================
    # Per-k: budget bisection (calib) -> eval generation -> rho, gate.
    # (Loop body IDENTICAL to run_a1_steelman EXCEPT the denominator is E_first,
    # via first_hits, and the calib within-split reference uses the task-vec
    # E_first, via meth.generate(p,'first').)
    # =====================================================================
    rung_results = []
    rho_by_k = {}
    for k in ranks:
        klabel = "full" if k == "full" else str(k)
        log(f"===== rung k={klabel} =====")
        b0, Wk = fit["maps"][k]

        unit_calib = prompt_bias_matrix(zc, b0, Wk)        # [n_calib, |S|]
        unit_eval = prompt_bias_matrix(ze, b0, Wk)         # [n_eval, |S|]

        g_k, achieved_kl = bisect_g_for_budget(
            model, tok, calib_prompts, calib_cont_ids, S, unit_calib,
            budget_target, device=device)
        budget_err = abs(achieved_kl - budget_target) / budget_target
        log(f"  g_k={g_k:.5f}  achieved_KL={achieved_kl:.5f}  "
            f"target={budget_target:.5f}  rel_err={budget_err:.4f} "
            f"(<{BUDGET_TOL}: {budget_err < BUDGET_TOL})")

        kl_spread = per_prompt_kl_spread(
            model, tok, calib_prompts, calib_cont_ids, S, g_k * unit_calib,
            device=device)
        kl_min = float(np.min(kl_spread)) if kl_spread else 0.0
        kl_med = float(np.median(kl_spread)) if kl_spread else 0.0
        kl_max = float(np.max(kl_spread)) if kl_spread else 0.0
        log(f"  per-prompt KL spread: min={kl_min:.4f} med={kl_med:.4f} "
            f"max={kl_max:.4f}")

        # ----- EVAL: per-prompt LogitBiasProcessor generation -----
        eval_bias = g_k * unit_eval                        # [n_eval, |S|]
        null_texts = []
        for i, p in enumerate(eval_prompts):
            proc = B.LogitBiasProcessor(
                S, torch.as_tensor(eval_bias[i], dtype=torch.float32))
            null_texts.append(B.control_generate(
                model, tok, p, proc, args.max_new_tokens, device))
        null_hits = [int(answer_hit(t, g))
                     for t, g in zip(null_texts, eval_golds)]
        eval_acc = float(np.mean(null_hits))

        # ----- rho on eval: denom = E_first (native), base = baseline, seed=13 --
        rho = B.bootstrap_ratio_ci(null_hits, first_hits, base_hits,
                                   args.n_boot, seed=13)
        rho_by_k[klabel] = rho

        # ----- overfitting sniff: calib-rho (guard e). Same estimator on calib:
        # null vs task-vec E_first ON CALIB (within-calib reference). -----
        calib_bias = g_k * unit_calib
        calib_null_texts = []
        for i, p in enumerate(calib_prompts):
            proc = B.LogitBiasProcessor(
                S, torch.as_tensor(calib_bias[i], dtype=torch.float32))
            calib_null_texts.append(B.control_generate(
                model, tok, p, proc, args.max_new_tokens, device))
        calib_null_hits = [int(answer_hit(t, g))
                           for t, g in zip(calib_null_texts, calib_golds)]
        calib_tv_texts = [meth.generate(p, "first") for p in calib_prompts]
        calib_tv_hits = [int(answer_hit(t, g))
                         for t, g in zip(calib_tv_texts, calib_golds)]
        calib_base_texts = [TV.base_generate(model, tok, p, args.max_new_tokens,
                                             device) for p in calib_prompts]
        calib_base_hits = [int(answer_hit(t, g))
                           for t, g in zip(calib_base_texts, calib_golds)]
        rho_calib = B.bootstrap_ratio_ci(calib_null_hits, calib_tv_hits,
                                         calib_base_hits, args.n_boot, seed=13)

        # ----- degeneracy gate (eval), same refs as run_taskvec -----
        gg = B.degeneracy_gate(null_texts, eval_prompts, ev_rep, ev_med, ev_nll,
                               model, tok, device=device)
        log(f"  eval_acc={eval_acc*100:.1f}%  rho={rho[0]:.3f} "
            f"[{rho[1]:.3f},{rho[2]:.3f}]  calib_rho={rho_calib[0]:.3f} "
            f"[{rho_calib[1]:.3f},{rho_calib[2]:.3f}]  "
            f"gate_tripped={gg.tripped}")

        rung_results.append({
            "k": klabel,
            "g": g_k,
            "achieved_kl": achieved_kl,
            "budget_rel_err": budget_err,
            "budget_ok": bool(budget_err < BUDGET_TOL),
            "kl_spread": {"min": kl_min, "median": kl_med, "max": kl_max,
                          "n": len(kl_spread)},
            "eval_acc": eval_acc,
            "rho": {"point": rho[0], "ci_lo": rho[1], "ci_hi": rho[2]},
            "rho_calib": {"point": rho_calib[0], "ci_lo": rho_calib[1],
                          "ci_hi": rho_calib[2]},
            "gate": {"tripped": bool(gg.tripped), "rep": gg.rep_rate,
                     "median_len": gg.median_len, "nll": gg.mean_nll,
                     "reasons": gg.reasons},
            "samples": list(zip(eval_prompts[:6],
                                [t.strip()[:50] for t in null_texts[:6]],
                                eval_golds[:6])),
        })

    # =====================================================================
    # Sanity checks (a, b) + Amendment-7 decision (CI bounds only).
    # =====================================================================
    rho0 = rho_by_k.get("0")
    rho_full = rho_by_k.get("full")
    upper_anchor_ok = None
    if rho0 is not None and rho_full is not None:
        upper_anchor_ok = bool(rho_full[0] > rho0[0])
    lower_anchor_in_band = None
    if rho0 is not None:
        lower_anchor_in_band = bool(
            (rho0[2] >= STATIC_NULL_RHO_LO) and (rho0[1] <= STATIC_NULL_RHO_HI))

    def _kval(kl):
        return 10**9 if kl == "full" else int(kl)
    bounded_rungs = [r for r in rung_results if _kval(r["k"]) <= 16]
    nondegen_bounded = [r for r in bounded_rungs if not r["gate"]["tripped"]]
    survives = (len(nondegen_bounded) > 0 and
                all(r["rho"]["ci_hi"] <= 0.3 for r in nondegen_bounded))
    artifact = any((not r["gate"]["tripped"]) and r["rho"]["ci_lo"] >= 0.9
                   for r in bounded_rungs)
    if artifact:
        decision = "STATIC-NULL-ARTIFACT"
    elif survives:
        decision = "SURVIVES-STEELMAN"
    else:
        decision = "INCONCLUSIVE"

    _rf_pt = f"{rho_full[0]:.3f}" if rho_full else "n/a"
    _r0_pt = f"{rho0[0]:.3f}" if rho0 else "n/a"
    _r0_ci = f"[{rho0[1]:.3f},{rho0[2]:.3f}]" if rho0 else "n/a"
    log(f"SANITY (a) upper anchor rho_full>rho_0: {upper_anchor_ok} "
        f"(rho_full={_rf_pt} vs rho_0={_r0_pt})")
    log(f"SANITY (b) lower anchor rho_0 in static-null band "
        f"[{STATIC_NULL_RHO_LO:.3f},{STATIC_NULL_RHO_HI:.3f}]: "
        f"{lower_anchor_in_band} (rho_0 CI={_r0_ci})")
    # STOP-condition determinism readout (Amendment 7): print the exact match.
    if rho0 is not None:
        k0_exact = (abs(rho0[0] - STATIC_NULL_RHO_POINT) < 1e-9
                    and abs(rho0[1] - STATIC_NULL_RHO_LO) < 1e-9
                    and abs(rho0[2] - STATIC_NULL_RHO_HI) < 1e-9)
        log(f"STOP-CHECK k=0 determinism: point {rho0[0]:+.6f} vs frozen "
            f"{STATIC_NULL_RHO_POINT:+.6f}  in-band={lower_anchor_in_band}  "
            f"EXACT-MATCH={k0_exact}")
    log(f"DECISION (Amendment 7, CI bounds): {decision}  "
        f"(survives={survives} artifact={artifact} "
        f"non-degen k<=16 rungs={len(nondegen_bounded)})")

    result = {
        "meta": {
            "purpose": "Task-vec steelman input-conditional null (plan §11 "
                       "Amendment 7); confirmation of the FV steelman (Am 6b).",
            "model": args.model, "device": device, "dtype": args.dtype,
            "task": args.task, "tag": tag, "inject_layer": inject_layer,
            "inject_op": inject_op, "phi_layer": phi_layer, "n_layers": n_layers,
            "resid_dim": cfg["resid_dim"], "theta_norm": meth.norm,
            "n_eval": len(eval_prompts), "n_calib": len(calib_prompts),
            "max_new_tokens": args.max_new_tokens, "n_boot": args.n_boot,
            "seed": args.seed, "ranks": [str(r) for r in ranks],
            "ridge_frac": args.ridge_frac,
            "B_star_frozen": B_STAR, "budget_target": budget_target,
            "budget_target_note": budget_note, "tv_own_tf_kl": tv_own_kl,
            "dose_frac": args.dose_frac, "budget_tol": BUDGET_TOL,
            "token_set_size": len(S),
            "token_set_matches_stored": s_matches_stored,
            "denom_note": "run-1 hit arrays not persisted; baseline + task-vec "
                          "E_first (native single write, KV-baked) "
                          "deterministically regenerated (greedy) as the "
                          "bootstrap denominator/base (E_first, NOT E_all — the "
                          "frozen task-vec rho convention).",
            "denom_baseline_acc": float(np.mean(base_hits)),
            "denom_e_first_acc": float(np.mean(first_hits)),
            "gate_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
            "frozen_static_null_rho": {"point": STATIC_NULL_RHO_POINT,
                                       "ci_lo": STATIC_NULL_RHO_LO,
                                       "ci_hi": STATIC_NULL_RHO_HI},
            "frozen_gate_refs": frozen_gate_refs,
            "frozen_effect": frozen_effect,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "fit": {
            "lambda": fit["lam"], "eff_rank": fit["eff_rank"],
            "singular_spectrum": fit["singular"],
            "Y_abs_mean": float(np.abs(Y).mean()),
        },
        "no_leakage": {
            "fit_params": sorted(_fit_params),
            "forbidden_intersect_empty": bool(not leak),
            "note": "fit_rank_ladder receives ONLY (phi_tilde, Y, ranks, "
                    "ridge_frac); theta/golds/eval-steered cannot be passed.",
        },
        "rungs": rung_results,
        "sanity": {
            "upper_anchor_rho_full_gt_rho0": upper_anchor_ok,
            "rho_full": {"point": rho_full[0], "ci_lo": rho_full[1],
                         "ci_hi": rho_full[2]} if rho_full else None,
            "rho_0": {"point": rho0[0], "ci_lo": rho0[1], "ci_hi": rho0[2]}
                     if rho0 else None,
            "lower_anchor_rho0_in_static_band": lower_anchor_in_band,
            "static_null_rho": {"point": STATIC_NULL_RHO_POINT,
                                "ci_lo": STATIC_NULL_RHO_LO,
                                "ci_hi": STATIC_NULL_RHO_HI},
        },
        "decision": {
            "class": decision,
            "rule": "SURVIVES <=> rho_hi<=0.3 for ALL non-degenerate k<=16; "
                    "ARTIFACT <=> clean rho_lo>=0.9 at SOME k<=16; "
                    "k=full = upper anchor (never Dissolved). CI bounds only.",
            "survives_steelman": bool(survives),
            "static_null_artifact": bool(artifact),
            "n_nondegen_bounded_rungs": len(nondegen_bounded),
        },
        "runtime_sec": time.time() - t0,
    }

    # Output filename: dose-frac labeled if provided, else steelman.{json,md}.
    if args.dose_frac is not None:
        stem = f"steelman_dose_f{args.dose_frac:.2f}"
    else:
        stem = "steelman_taskvec" if not args.smoke else "steelman_taskvec_smoke"
    jpath = os.path.join(outdir, f"{stem}.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"wrote {jpath}")
    write_md(result, os.path.join(outdir, f"{stem}.md"))
    log(f"wrote {os.path.join(outdir, stem + '.md')}")
    log(f"DONE in {result['runtime_sec']:.0f}s  DECISION={decision}")


def write_md(r, path):
    m = r["meta"]; d = r["decision"]; sa = r["sanity"]; fit = r["fit"]
    L = []
    A = L.append
    A("# Task-vec anchor — steelman input-conditional null (Amendment 7)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` (inject L{m['inject_layer']} op={m['inject_op']}, "
      f"phi_layer {m['phi_layer']}, resid {m['resid_dim']}), device "
      f"`{m['device']}` ({m['dtype']}).  ")
    A(f"**Eval:** {m['n_eval']} held-out prompts, {m['max_new_tokens']}-token "
      f"greedy, task accuracy, {m['n_boot']} bootstrap.  ")
    A(f"**Calib:** {m['n_calib']} prompts (fit + budget-match).  ")
    A(f"**Token set S:** {m['token_set_size']} tokens "
      f"(matches frozen stored set: {m['token_set_matches_stored']}).  ")
    A(f"**Budget:** frozen B* = "
      f"{m['B_star_frozen'] if m['B_star_frozen'] else float('nan'):.5f}; target "
      f"= {m['budget_target']:.5f} ({m['budget_target_note']}); per-k tol "
      f"|achieved-target|/target < {m['budget_tol']}.  ")
    A(f"**Denominator:** E_first (native single write). {m['denom_note']} "
      f"baseline acc {m['denom_baseline_acc']*100:.1f}%, E_first acc "
      f"{m['denom_e_first_acc']*100:.1f}%.\n")

    A(f"## DECISION: **{d['class']}**\n")
    A(f"Rule (plan Amendment 7, CI bounds only): {d['rule']}\n")
    A(f"- SURVIVES steelman = **{d['survives_steelman']}** "
      f"(non-degenerate k<=16 rungs: {d['n_nondegen_bounded_rungs']})")
    A(f"- static-null ARTIFACT = **{d['static_null_artifact']}**\n")

    A(f"## Validity guards (harness untrustworthy if any fails)\n")
    ua = sa["upper_anchor_rho_full_gt_rho0"]
    rf = sa["rho_full"]; r0 = sa["rho_0"]; sn = sa["static_null_rho"]
    if rf and r0:
        A(f"- **(a) upper anchor** rho_full > rho_0 = **{ua}**  "
          f"(rho_full = {rf['point']:.3f} [{rf['ci_lo']:.3f}, {rf['ci_hi']:.3f}]; "
          f"rho_0 = {r0['point']:.3f} [{r0['ci_lo']:.3f}, {r0['ci_hi']:.3f}]). "
          f"If FALSE the map class is not learning the steered field.")
    A(f"- **(b) lower anchor / STOP-condition** rho_0 in frozen static-null band "
      f"[{sn['ci_lo']:.3f}, {sn['ci_hi']:.3f}] = "
      f"**{sa['lower_anchor_rho0_in_static_band']}**  "
      f"(k=0 == best-constant-bias == static-null; Amendment-7 k=0 MUST "
      f"reproduce rho=-0.1408 or the patch config is wrong).")
    A(f"- **(c) no-leakage** fit_rank_ladder forbidden-param intersection empty "
      f"= **{r['no_leakage']['forbidden_intersect_empty']}** "
      f"(params: {r['no_leakage']['fit_params']}).")
    A(f"- **(d) budget** per-k |achieved-B*|/B* < {m['budget_tol']} (see table).")
    A(f"- **(e) overfitting** calib-rho vs eval-rho per k (see table).\n")

    A(f"## Fit\n")
    A(f"- Ridge lambda = {fit['lambda']:.4g}; effective rank of W = "
      f"{fit['eff_rank']} (calib-limited, <= n_calib); |Y| mean = "
      f"{fit['Y_abs_mean']:.4f}.")
    A(f"- Singular spectrum of W (top 10): "
      f"{[round(x, 3) for x in fit['singular_spectrum'][:10]]}\n")

    A(f"## Rank ladder (eval split)\n")
    A(f"rho = E(null_k)/E(task-vec E_first), bootstrap 95% CI. Genuine SURVIVES "
      f"iff rho_hi <= 0.30 for every non-degenerate bounded (k<=16) rung.\n")
    A("| k | g | achieved KL | budget err | eval acc | rho (eval) [95% CI] | "
      "calib rho [95% CI] | KL min/med/max | gate |")
    A("|---|---:|---:|---:|---:|---|---|---|:---:|")
    for r_ in r["rungs"]:
        ks = r_["kl_spread"]
        rho = r_["rho"]; rc = r_["rho_calib"]
        gate = "VOID" if r_["gate"]["tripped"] else "ok"
        anchor = " (upper anchor)" if r_["k"] == "full" else ""
        A(f"| {r_['k']}{anchor} | {r_['g']:.3f} | {r_['achieved_kl']:.4f} | "
          f"{r_['budget_rel_err']:.4f} | {r_['eval_acc']*100:.1f}% | "
          f"{rho['point']:.3f} [{rho['ci_lo']:.3f}, {rho['ci_hi']:.3f}] | "
          f"{rc['point']:.3f} [{rc['ci_lo']:.3f}, {rc['ci_hi']:.3f}] | "
          f"{ks['min']:.3f}/{ks['median']:.3f}/{ks['max']:.3f} | {gate} |")
    A("")
    A(f"Gate reasons (per tripped rung):")
    any_trip = False
    for r_ in r["rungs"]:
        if r_["gate"]["reasons"]:
            any_trip = True
            A(f"- k={r_['k']}: {'; '.join(r_['gate']['reasons'])}")
    if not any_trip:
        A("- (none tripped)")
    A("")
    A(f"Runtime: {r['runtime_sec']:.0f}s.\n")
    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
