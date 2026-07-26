"""A1 anchor — STEELMAN input-conditional null (plan §11 Amendment 4).

Closes HOLE 1. The frozen A1 primary control is a STATIC (input-independent)
logit-bias push; an input-conditional method (FV antonyms: the correct answer
depends on the prompt) is pre-ordained to under-reproduce under a static null
regardless of budget. This driver tests whether the FV "Genuine" verdict
survives a *steelman* input-conditional output interface: a capacity-indexed
(rank-k) linear map from the base model's last-layer resid_post at the final
prompt token -> a per-token logit bias on the SAME frozen token set S, fit on
the SAME 50 calib prompts, evaluated on the SAME 150 held-out eval prompts,
scaled to the SAME frozen budget B* (0.70353), scored by the SAME
metric/gate/bootstrap.

Null form (spec §Null form):
    b(p) = g * M_k * phi_tilde(p)   restricted to token set S
    phi(p)       = BASE last-layer resid_post @ final prompt token (no hooks)
    phi_tilde(p) = [ (phi - mu)/sigma ; 1 ]   (mu, sigma from CALIB only; the
                   appended constant column expresses the input-independent part)
    M_k          = rank-k truncation of the ridge-fit map, written only to |S|
    g            = scalar budget knob, bisected AFTER fitting to hit B*.

Rank ladder k in {0, 1, 4, 16, full(<=50, calib-limited)}. k=0 (W=0 => b0 only)
is the best-constant-bias == static-null analogue. k=full is a labeled UPPER
ANCHOR (never calls Dissolved).

Anti-cheat (spec §Info access): the pure fit function `fit_rank_ladder` receives
ONLY the calib design matrix Phi_tilde and the calib target Y. It is
structurally impossible to pass it the FV steering vector, gold labels, or any
eval-steered output. The target Y is the steered model's teacher-forced logit
delta on CALIB ONLY (allowed).

Decision (plan Amendment 4; CI bounds ONLY):
    SURVIVES  <=> rho_hi <= 0.3 for ALL non-degenerate k <= 16.
    ARTIFACT  <=> clean rho_lo >= 0.9 at SOME k <= 16.
    k=full = upper anchor, never calls Dissolved.

Mandatory sanity checks (inline, spec §Sanity checks / Amendment 4 guards):
    (a) upper anchor rho_full > rho_0 (map learns the field) — else STOP/debug Y.
    (b) lower anchor rho_0 ~= static-null rho (-0.12) within boot noise; k=0
        cell VOID like the static null.
    (c) no-leakage asserts (fv_vec / labels / eval-steered not passed to fit).
    (d) budget |achieved_KL - B*|/B* < 0.02 per k; per-prompt KL min/median/max.
    (e) overfitting: calib-rho vs eval-rho per k.

Outputs steelman.{json,md} APPEND-ONLY into the a1-anchor run dir. NEVER touches
report.md / results_full.json / any frozen input.

Usage:
  source .venv/bin/activate
  # full FV job (pythia-2.8b, MPS, bf16):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_a1_steelman.py
  # smoke (pythia-160m, exercises the FULL code path):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python run_a1_steelman.py --smoke
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

# Frozen anchors (VERIFIED against results_full.json; see spec §Frozen anchors).
B_STAR = 0.7035311031341552          # control_calibration.B_star_target_kl
STATIC_NULL_RHO_POINT = -0.11864406779661019
STATIC_NULL_RHO_LO = -0.23076923076923075
STATIC_NULL_RHO_HI = -0.0392156862745098
BUDGET_TOL = 0.02                    # |achieved-B*|/B* must be < this (guard d)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "data", "external", "function_vectors", "dataset_files", "abstractive")


# ===========================================================================
# Fit target Y (calib): steered-minus-base teacher-forced per-position logit
# delta on token set S. Mirrors battery.teacher_forced_stepkl_steered EXACTLY
# (same FV E_all patch, same pred_positions, single forward), but KEEPS the
# per-position delta on S instead of collapsing to KL.
# ===========================================================================

def fv_teacher_forced_delta_on_S(
        fvm: "FVMethod", tokenizer, prompts, continuation_ids, token_ids,
        device="cpu"):
    """Per-prompt mean (over continuation positions) of (steered - base) logits
    restricted to ``token_ids`` (== S). Returns Y ndarray [n_prompts, |S|].

    Steered == the FV E_all patch (fvm.fv add_vector at every continuation-
    producing position), teacher-forced on the fixed unsteered continuation —
    the exact configuration of teacher_forced_stepkl_steered.
    """
    from actlib.patching import _dynamic_patch_hook
    meth = fvm.meth
    model = meth.model
    layer = meth.layer
    hidden = meth.u.shape[0]
    addv = meth.add_vector(1.0)               # == fvm.fv (alpha=1)
    rb = torch.zeros(0, hidden, device=device)
    tid = torch.tensor(list(token_ids), device=device)
    rows = []
    for prompt, cont in zip(prompts, continuation_ids):
        if len(cont) == 0:
            rows.append(np.zeros(len(token_ids), dtype=np.float64))
            continue
        full, P = B._teacher_forced_seq(tokenizer, prompt, cont, device)
        n = len(cont)
        pred_positions = list(range(P - 1, P - 1 + n))
        with torch.no_grad():
            base_logits = model(full).logits[0]          # [seq, vocab]
        state = {"cache_offset": 0, "targets": pred_positions}
        with torch.no_grad(), _dynamic_patch_hook(
                model, "resid_post", layer, None, None, "subspace_transplant",
                state, remove_subspace=rb, add_vector=addv):
            steered_logits = model(full).logits[0]        # [seq, vocab]
        deltas = []
        for pos in pred_positions:
            d = (steered_logits[pos, tid] - base_logits[pos, tid])
            deltas.append(d.to("cpu").float())
        rows.append(torch.stack(deltas).mean(0).numpy().astype(np.float64))
    return np.stack(rows)                                  # [n, |S|]


# ===========================================================================
# phi(p): BASE last-layer resid_post at final prompt token (no hooks).
# ===========================================================================

def base_phi(model, tokenizer, prompts, phi_layer, device="cpu",
             batch_size=16):
    """Capture BASE resid_post at ``phi_layer``, last token -> ndarray [n, resid].
    No steering hooks; the plain forward pass."""
    acts = B.capture_activations(
        model, tokenizer, prompts, "resid_post", phi_layer,
        positions="last", batch_size=batch_size, device=device)
    return acts[(phi_layer, "resid_post")].float().numpy().astype(np.float64)


# ===========================================================================
# PURE FIT (anti-cheat surface). Receives ONLY the calib design matrix
# Phi_tilde [n, d+1] (last column == the constant 1) and the calib target
# Y [n, |S|]. Cannot see fv_vec / golds / eval-steered outputs — by signature.
# ===========================================================================

def fit_rank_ladder(phi_tilde, Y, ranks, ridge_frac):
    """Closed-form ridge + SVD rank ladder. Deterministic, CPU/numpy.

    phi_tilde : [n, d+1] design rows, FINAL column all-ones (constant).
    Y         : [n, m] targets (m == |S|).
    ranks     : list with ints and/or the string 'full'.
    ridge_frac: lambda = ridge_frac * mean(diag(Phi^T Phi)).

    Returns dict:
      'lam', 'M_full' [d+1, m], 'b0' [m] (const row), 'W' [d, m] (input block),
      'singular' (list, W spectrum), 'eff_rank', 'maps': {k: (b0[m], Wk[d,m])}.
    Reconstruction: bias(phi_z) = b0 + Wk^T @ phi_z, phi_z = (phi-mu)/sigma.
    """
    Phi = np.asarray(phi_tilde, dtype=np.float64)          # [n, d+1]
    Yv = np.asarray(Y, dtype=np.float64)                   # [n, m]
    d1 = Phi.shape[1]
    G = Phi.T @ Phi                                         # [d+1, d+1]
    lam = float(ridge_frac) * float(np.mean(np.diag(G)))
    M_full = np.linalg.solve(G + lam * np.eye(d1), Phi.T @ Yv)  # [d+1, m]
    b0 = M_full[-1, :].copy()                              # constant row -> [m]
    W = M_full[:-1, :].copy()                              # [d, m] input block
    # SVD of the input-conditional block (W = U S Vt).
    U, S, Vt = np.linalg.svd(W, full_matrices=False)       # U[d,r] S[r] Vt[r,m]
    eff_rank = int(np.sum(S > 1e-9 * (S[0] if S.size else 1.0)))
    maps = {}
    for k in ranks:
        if k == "full":
            Wk = W.copy()
        else:
            kk = int(k)
            if kk <= 0:
                Wk = np.zeros_like(W)
            else:
                kk = min(kk, S.shape[0])
                Wk = (U[:, :kk] * S[:kk]) @ Vt[:kk, :]     # [d, m]
        maps[k] = (b0.copy(), Wk)
    return {
        "lam": lam, "M_full": M_full, "b0": b0, "W": W,
        "singular": [float(x) for x in S], "eff_rank": eff_rank, "maps": maps,
    }


# ===========================================================================
# Per-prompt bias on S. bias_k(p) = g * (b0 + Wk^T @ phi_z(p)) restricted to S.
# ===========================================================================

def prompt_bias_matrix(phi_z, b0, Wk):
    """Return [n, m] bias rows for g=1: b0 + Wk^T @ phi_z(p) for every prompt.
    phi_z : [n, d] standardized phi. Wk : [d, m]. b0 : [m]."""
    return phi_z @ Wk + b0[None, :]                        # [n, m]


# ===========================================================================
# Per-prompt teacher-forced KL(biased || base), generalizing
# battery.teacher_forced_stepkl_biased from SHARED bias to a PER-PROMPT bias.
# bias_rows[i] is added to token set S at every continuation position of prompt i.
# ===========================================================================

def teacher_forced_stepkl_biased_perprompt(
        model, tokenizer, prompts, continuation_ids, token_ids, bias_rows,
        device="cpu"):
    """Mean teacher-forced per-step KL(biased || base) with a PER-PROMPT bias.

    bias_rows : [n_prompts, |S|] — prompt i's bias vector on token set S, added
    at EVERY continuation position of prompt i (teacher-forced on the fixed
    unsteered continuation), matching the steered-quantity averaging (all
    positions x prompts). Reuses the base_logits path of the shared version."""
    tid = torch.tensor(list(token_ids), device=device)
    all_kls = []
    for i, (prompt, cont) in enumerate(zip(prompts, continuation_ids)):
        if len(cont) == 0:
            continue
        full, P = B._teacher_forced_seq(tokenizer, prompt, cont, device)
        n = len(cont)
        pred_positions = list(range(P - 1, P - 1 + n))
        bv = torch.as_tensor(bias_rows[i], dtype=torch.float32, device=device)
        with torch.no_grad():
            base_logits = model(full).logits[0]           # [seq, vocab]
        for pos in pred_positions:
            base = base_logits[pos].float()
            biased = base.clone()
            biased[tid] = biased[tid] + bv
            p = torch.log_softmax(biased, dim=-1)
            q = torch.log_softmax(base, dim=-1)
            all_kls.append((p.exp() * (p - q)).sum().item())
    return float(np.mean(all_kls)) if all_kls else 0.0


def bisect_g_for_budget(
        model, tokenizer, prompts, continuation_ids, token_ids, unit_bias_rows,
        target_kl, device="cpu", lo=0.0, hi=8.0, iters=30):
    """Bisect scalar g so mean teacher-forced per-step KL(g*bias || base) == B*.

    unit_bias_rows : [n, |S|] the per-prompt bias at g=1 (b0 + Wk^T phi_z). KL
    rises monotonically from 0 at g=0, so bisection is valid. Mirrors
    battery.calibrate_bias_scalar_stepkl (same bracket-expand + 30 iters).
    Returns (g, achieved_kl). If unit bias is ~0 (k=0 with b0~0 can still be
    nonzero), the bracket expands up to 12x."""
    def kl_at(g):
        return teacher_forced_stepkl_biased_perprompt(
            model, tokenizer, prompts, continuation_ids, token_ids,
            g * unit_bias_rows, device=device)

    khi = kl_at(hi)
    tries = 0
    while khi < target_kl and tries < 12:
        hi *= 1.5
        khi = kl_at(hi)
        tries += 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        k = kl_at(mid)
        if k < target_kl:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    return g, kl_at(g)


def per_prompt_kl_spread(
        model, tokenizer, prompts, continuation_ids, token_ids, bias_rows,
        device="cpu"):
    """Per-prompt (mean-over-positions) teacher-forced KL(biased||base), so we
    can report min/median/max of the budget distribution (guard c: the steelman
    must not concentrate B* on a few prompts). Returns list[float] length<=n."""
    tid = torch.tensor(list(token_ids), device=device)
    out = []
    for i, (prompt, cont) in enumerate(zip(prompts, continuation_ids)):
        if len(cont) == 0:
            continue
        full, P = B._teacher_forced_seq(tokenizer, prompt, cont, device)
        n = len(cont)
        pred_positions = list(range(P - 1, P - 1 + n))
        bv = torch.as_tensor(bias_rows[i], dtype=torch.float32, device=device)
        kls = []
        with torch.no_grad():
            base_logits = model(full).logits[0]
        for pos in pred_positions:
            base = base_logits[pos].float()
            biased = base.clone()
            biased[tid] = biased[tid] + bv
            p = torch.log_softmax(biased, dim=-1)
            q = torch.log_softmax(base, dim=-1)
            kls.append((p.exp() * (p - q)).sum().item())
        out.append(float(np.mean(kls)))
    return out


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
    ap.add_argument("--n-mean", type=int, default=100)
    ap.add_argument("--n-cie", type=int, default=32)
    ap.add_argument("--n-shots", type=int, default=10)
    ap.add_argument("--cie-lo", type=int, default=3)
    ap.add_argument("--cie-hi", type=int, default=24)
    ap.add_argument("--no-cie-band", action="store_true")
    ap.add_argument("--n-top-heads", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--phi-layer", type=int, default=None,
                    help="resid_post layer for phi (default = n_layers-1)")
    ap.add_argument("--ranks", default="0,1,4,16,full")
    ap.add_argument("--ridge-frac", type=float, default=1e-2)
    ap.add_argument("--budget-kl", type=float, default=None,
                    help="explicit budget-match target KL. Default: frozen B* "
                         "(2.8b full run) or, in --smoke, an auto reachable "
                         "target (budget-frac x the per-k KL ceiling).")
    ap.add_argument("--budget-frac", type=float, default=0.7,
                    help="smoke-only: fraction of the (saturating) achievable "
                         "bias-KL ceiling to target so the g-bisection actually "
                         "converges on the tiny model (full code-path exercise).")
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
        args.n_boot = 200
        args.ranks = "0,1,4,full"     # 16 ~= full at n_calib=10

    ranks = [("full" if r.strip() == "full" else int(r.strip()))
             for r in args.ranks.split(",") if r.strip()]

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    outdir = args.outdir or os.path.join(
        repo, "runs", "steering-content-audit", "2026-07-06-a1-anchor")
    os.makedirs(outdir, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    log(f"STEELMAN (Amendment 4)  model={args.model} device={args.device} "
        f"dtype={args.dtype} ranks={ranks} tag={tag}")

    # --- model ---
    _dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model, tok = B.load_model(args.model, device=args.device, dtype=_dtype)
    device = args.device
    cfg = FV.neox_config(model)
    n_layers = cfg["n_layers"]
    phi_layer = args.phi_layer if args.phi_layer is not None else n_layers - 1
    log(f"config: layers={n_layers} heads={cfg['n_heads']} "
        f"resid={cfg['resid_dim']} phi_layer={phi_layer}")

    # --- dataset split (IDENTICAL to run_a1 / run_a1_dose) ---
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
    # FV: load stage1 cache if present, else extract (same as run_a1).
    # =====================================================================
    s1_cache = os.path.join(
        outdir, f"stage1_{args.task}_{args.model.split('/')[-1]}.pt")
    if os.path.exists(s1_cache):
        log(f"FV: loading stage1 cache {s1_cache}")
        blob = torch.load(s1_cache, weights_only=False)  # our own trusted cache
        fv_vec = blob["fv"]
        edit_layer = blob["edit_layer"]
    else:
        log("FV: no cache -> extracting Stage 1 (this reproduces run_a1)...")
        edit_layer = round(n_layers / 3)
        cie_layers = None if args.no_cie_band else list(
            range(args.cie_lo, min(args.cie_hi, n_layers - 1) + 1))
        clean = FV.sample_icl_prompts(train_pool, args.n_mean, args.n_shots,
                                      seed=args.seed + 2, shuffle_labels=False)
        mean_acts = FV.mean_head_activations(model, tok, clean, cfg,
                                             device=device, log=log)
        shuffled = FV.sample_icl_prompts(train_pool, args.n_cie, args.n_shots,
                                         seed=args.seed + 3, shuffle_labels=True)
        cie = FV.compute_indirect_effect(model, tok, shuffled, mean_acts, cfg,
                                         layers=cie_layers, device=device, log=log)
        top = FV.top_heads(cie, args.n_top_heads)
        fv_vec = FV.build_function_vector(model, mean_acts, top, cfg, device=device)
        torch.save({"fv": fv_vec, "top_heads": top, "mean_acts": mean_acts,
                    "cie": cie, "cie_layers": cie_layers,
                    "edit_layer": edit_layer, "cfg": cfg}, s1_cache)
        log(f"  cached Stage 1 -> {s1_cache}")
    fvm = FVMethod(model, tok, edit_layer, fv_vec, device=device,
                   max_new_tokens=args.max_new_tokens)
    log(f"  FV norm={fvm.norm:.3f}  edit_layer={edit_layer}")

    # =====================================================================
    # Token set S: reconstruct identically to run_a1 (position-1 mean delta on
    # calib -> discover_token_set 0.90 cap 100). ASSERT == stored 100 (full run).
    # =====================================================================
    log("S: position-1 mean logit delta on calib (reconstruct token set)...")
    mean_delta = fv_position1_logit_delta(fvm, calib_prompts)
    S = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    log(f"  |S|={len(S)}  top={[tok.decode([i]) for i in S[:8]]}")
    stored_ids = None
    s_matches_stored = None
    rf = os.path.join(outdir, "results_full.json")
    if os.path.exists(rf):
        stored_ids = json.load(open(rf))["control_calibration"]["token_ids"]
        s_matches_stored = (S == stored_ids)
        log(f"  S matches stored 100-token set (full run only): {s_matches_stored}")
        if not args.smoke:
            assert s_matches_stored, (
                "token set S does not match the frozen results_full.json set; "
                "split/FV reconstruction diverged — ABORT (frozen-input mismatch)")

    # =====================================================================
    # phi (BASE resid_post @ last token) on calib + eval; standardize by CALIB.
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
    # Fit target Y (calib): FV teacher-forced per-position delta on S.
    # =====================================================================
    log("Y: unsteered calib continuations (teacher-forcing base)...")
    calib_cont_ids = [B.base_generate_ids(model, tok, p, args.max_new_tokens,
                                          device) for p in calib_prompts]
    log("Y: FV teacher-forced (steered-base) per-position delta on S (calib)...")
    Y = fv_teacher_forced_delta_on_S(
        fvm, tok, calib_prompts, calib_cont_ids, S, device=device)
    log(f"  Y shape={Y.shape}  |Y| mean={np.abs(Y).mean():.4f}")

    # --- structural no-leakage assert (guard c) ---
    # The pure fit sees ONLY (phi_tilde_calib, Y). Prove by signature + a runtime
    # guard: fit_rank_ladder's parameters do not include model/fvm/golds/eval.
    import inspect
    _fit_params = set(inspect.signature(fit_rank_ladder).parameters)
    _forbidden = {"model", "fvm", "fv_vec", "golds", "eval_prompts",
                  "eval_golds", "tokenizer", "tok"}
    leak = _fit_params & _forbidden
    assert not leak, f"LEAKAGE: fit_rank_ladder exposes forbidden params {leak}"
    log(f"  no-leakage: fit_rank_ladder params={sorted(_fit_params)} "
        f"(forbidden intersect empty: {not leak})")

    # =====================================================================
    # Fit rank ladder (pure, deterministic).
    # =====================================================================
    log(f"FIT: ridge + SVD ladder (ridge_frac={args.ridge_frac})...")
    fit = fit_rank_ladder(phi_tilde_calib, Y, ranks, args.ridge_frac)
    log(f"  lambda={fit['lam']:.4g}  eff_rank={fit['eff_rank']}  "
        f"top-singular={fit['singular'][:5]}")

    # =====================================================================
    # Budget-match target B*. FULL RUN: the FROZEN B_STAR (0.70353), matched
    # exactly per spec; we recompute the FV's own teacher-forced per-step KL as
    # a sanity check that the frozen anchor still holds on this machine.
    # SMOKE: on pythia-160m the teacher-forced biased-KL on S SATURATES far
    # below B_STAR (mean_delta[S] is too flat to concentrate mass — verified;
    # this does NOT reproduce on 2.8b where run_a1 hit 0.7033 at c=3.54). So the
    # smoke targets a reachable value (budget_frac x the per-k KL ceiling) so the
    # g-bisection genuinely CONVERGES and the differentiated ladder + per-prompt
    # generation + gate + rho are all exercised on non-degenerate biases.
    # =====================================================================
    log("BUDGET: recompute FV own teacher-forced per-step KL (anchor check)...")
    fv_own_kl = B.teacher_forced_stepkl_steered(
        fvm.meth, tok, calib_prompts, calib_cont_ids, 1.0)
    log(f"  FV own TF per-step KL = {fv_own_kl:.5f}  (frozen B*={B_STAR:.5f})")
    budget_note = ""
    if args.budget_kl is not None:
        budget_target = float(args.budget_kl)
        budget_note = f"explicit --budget-kl={budget_target}"
    elif args.smoke:
        # Probe the saturating KL ceiling of each rung's unit bias (huge g),
        # take the MIN across k so the target is bracketable for EVERY rung.
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
        # sanity: the frozen anchor should still reproduce on this machine.
        anchor_rel = abs(fv_own_kl - B_STAR) / B_STAR
        log(f"  frozen-anchor check: |FV_KL - B*|/B* = {anchor_rel:.4f}")
        if anchor_rel > 0.02:
            log(f"  WARNING: FV own KL {fv_own_kl:.5f} deviates >2% from frozen "
                f"B* {B_STAR:.5f}; split/FV reconstruction may have drifted.")
    log(f"BUDGET target = {budget_target:.5f}  ({budget_note})")

    # =====================================================================
    # Baseline + FV E_all bootstrap denominators. run-1 hit arrays are NOT
    # persisted -> deterministically regen once (greedy), matching run_a1.
    # =====================================================================
    log("DENOM: baseline hits on eval (greedy, regen for bootstrap denom)...")
    base_texts = [B.base_generate(model, tok, p, args.max_new_tokens, device)
                  for p in eval_prompts]
    base_hits = [int(answer_hit(t, g)) for t, g in zip(base_texts, eval_golds)]
    log(f"  baseline acc={np.mean(base_hits)*100:.1f}%")
    log("DENOM: FV E_all hits on eval (greedy)...")
    all_texts = [fvm.generate(p, "all") for p in eval_prompts]
    all_hits = [int(answer_hit(t, g)) for t, g in zip(all_texts, eval_golds)]
    log(f"  FV E_all acc={np.mean(all_hits)*100:.1f}%  "
        f"(regen denom; run-1 E_all was 44.7%)")

    # eval gate baseline refs from the freshly-regenerated FV baseline (matches
    # run_a1 construction: rep/median_len/nll of the baseline eval texts).
    ev_rep = float(np.mean([B.three_gram_rep_rate(t, tok) for t in base_texts]))
    ev_med = B.median_len_tokens(base_texts, tok)
    ev_nll = float(np.mean([B.mean_nll_under_model(model, tok, p, t, device)
                            for p, t in zip(eval_prompts, base_texts)]))
    log(f"  gate refs: rep={ev_rep:.5f} median_len={ev_med} nll={ev_nll:.5f}")

    # =====================================================================
    # Per-k: budget bisection (calib) -> eval generation -> rho, gate.
    # =====================================================================
    rung_results = []
    rho_by_k = {}
    for k in ranks:
        klabel = "full" if k == "full" else str(k)
        log(f"===== rung k={klabel} =====")
        b0, Wk = fit["maps"][k]

        # unit (g=1) per-prompt bias rows on S, calib + eval.
        unit_calib = prompt_bias_matrix(zc, b0, Wk)        # [n_calib, |S|]
        unit_eval = prompt_bias_matrix(ze, b0, Wk)         # [n_eval, |S|]

        # budget bisection on calib to hit the budget target (guard d).
        g_k, achieved_kl = bisect_g_for_budget(
            model, tok, calib_prompts, calib_cont_ids, S, unit_calib,
            budget_target, device=device)
        budget_err = abs(achieved_kl - budget_target) / budget_target
        log(f"  g_k={g_k:.5f}  achieved_KL={achieved_kl:.5f}  "
            f"target={budget_target:.5f}  rel_err={budget_err:.4f} "
            f"(<{BUDGET_TOL}: {budget_err < BUDGET_TOL})")

        # per-prompt KL spread at the budget-matched g (guard c).
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

        # ----- rho on eval (frozen bootstrap: denom = FV E_all, base = baseline) -----
        rho = B.bootstrap_ratio_ci(null_hits, all_hits, base_hits,
                                   args.n_boot, seed=13)
        rho_by_k[klabel] = rho

        # ----- overfitting sniff: calib-rho (guard e). Same estimator on calib:
        # generate on the SAME calib prompts with the same map/budget, score,
        # bootstrap vs FV-E_all/base ON CALIB (a within-calib reference). -----
        calib_bias = g_k * unit_calib
        calib_null_texts = []
        for i, p in enumerate(calib_prompts):
            proc = B.LogitBiasProcessor(
                S, torch.as_tensor(calib_bias[i], dtype=torch.float32))
            calib_null_texts.append(B.control_generate(
                model, tok, p, proc, args.max_new_tokens, device))
        calib_null_hits = [int(answer_hit(t, g))
                           for t, g in zip(calib_null_texts, calib_golds)]
        calib_fv_texts = [fvm.generate(p, "all") for p in calib_prompts]
        calib_fv_hits = [int(answer_hit(t, g))
                         for t, g in zip(calib_fv_texts, calib_golds)]
        calib_base_texts = [B.base_generate(model, tok, p, args.max_new_tokens,
                                            device) for p in calib_prompts]
        calib_base_hits = [int(answer_hit(t, g))
                           for t, g in zip(calib_base_texts, calib_golds)]
        rho_calib = B.bootstrap_ratio_ci(calib_null_hits, calib_fv_hits,
                                         calib_base_hits, args.n_boot, seed=13)

        # ----- degeneracy gate (eval), same refs as run_a1 -----
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
    # Sanity checks (a, b) + Amendment-4 decision (CI bounds only).
    # =====================================================================
    rho0 = rho_by_k.get("0")
    rho_full = rho_by_k.get("full")
    # (a) upper anchor: rho_full point > rho_0 point (map learns the field).
    upper_anchor_ok = None
    if rho0 is not None and rho_full is not None:
        upper_anchor_ok = bool(rho_full[0] > rho0[0])
    # (b) lower anchor: rho_0 within static-null boot noise band.
    lower_anchor_in_band = None
    if rho0 is not None:
        lower_anchor_in_band = bool(
            (rho0[2] >= STATIC_NULL_RHO_LO) and (rho0[1] <= STATIC_NULL_RHO_HI))

    # Amendment 4 decision on CI bounds. Non-degenerate k<=16 rungs only.
    def _kval(kl):
        return 10**9 if kl == "full" else int(kl)
    bounded_rungs = [r for r in rung_results if _kval(r["k"]) <= 16]
    nondegen_bounded = [r for r in bounded_rungs if not r["gate"]["tripped"]]
    # SURVIVES <=> rho_hi <= 0.3 for every non-degenerate k<=16.
    survives = (len(nondegen_bounded) > 0 and
                all(r["rho"]["ci_hi"] <= 0.3 for r in nondegen_bounded))
    # ARTIFACT <=> clean rho_lo >= 0.9 at SOME k<=16.
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
    log(f"DECISION (Amendment 4, CI bounds): {decision}  "
        f"(survives={survives} artifact={artifact} "
        f"non-degen k<=16 rungs={len(nondegen_bounded)})")

    result = {
        "meta": {
            "purpose": "A1 steelman input-conditional null (plan §11 Amendment 4)",
            "model": args.model, "device": device, "dtype": args.dtype,
            "task": args.task, "tag": tag, "edit_layer": edit_layer,
            "phi_layer": phi_layer, "n_layers": n_layers,
            "resid_dim": cfg["resid_dim"], "fv_norm": fvm.norm,
            "n_eval": len(eval_prompts), "n_calib": len(calib_prompts),
            "max_new_tokens": args.max_new_tokens, "n_boot": args.n_boot,
            "seed": args.seed, "ranks": [str(r) for r in ranks],
            "ridge_frac": args.ridge_frac,
            "B_star_frozen": B_STAR, "budget_target": budget_target,
            "budget_target_note": budget_note, "fv_own_tf_kl": fv_own_kl,
            "budget_tol": BUDGET_TOL,
            "token_set_size": len(S),
            "token_set_matches_stored": s_matches_stored,
            "denom_note": "run-1 hit arrays not persisted; FV E_all + baseline "
                          "deterministically regenerated (greedy) as the "
                          "bootstrap denominator/base.",
            "denom_baseline_acc": float(np.mean(base_hits)),
            "denom_fv_eall_acc": float(np.mean(all_hits)),
            "gate_refs": {"rep": ev_rep, "median_len": ev_med, "nll": ev_nll},
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
                    "ridge_frac); fv_vec/golds/eval-steered cannot be passed.",
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

    jpath = os.path.join(outdir, "steelman.json")
    with open(jpath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log(f"wrote {jpath}")
    write_md(result, os.path.join(outdir, "steelman.md"))
    log(f"wrote {os.path.join(outdir, 'steelman.md')}")
    log(f"DONE in {result['runtime_sec']:.0f}s  DECISION={decision}")


def write_md(r, path):
    m = r["meta"]; d = r["decision"]; sa = r["sanity"]; fit = r["fit"]
    L = []
    A = L.append
    A("# A1 anchor — steelman input-conditional null (Amendment 4)\n")
    A(f"**Run:** {m['timestamp']}  ")
    A(f"**Model:** `{m['model']}` (edit_layer {m['edit_layer']}, phi_layer "
      f"{m['phi_layer']}, resid {m['resid_dim']}), device `{m['device']}` "
      f"({m['dtype']}).  ")
    A(f"**Eval:** {m['n_eval']} held-out prompts, {m['max_new_tokens']}-token "
      f"greedy, task accuracy, {m['n_boot']} bootstrap.  ")
    A(f"**Calib:** {m['n_calib']} prompts (fit + budget-match).  ")
    A(f"**Token set S:** {m['token_set_size']} tokens "
      f"(matches frozen stored set: {m['token_set_matches_stored']}).  ")
    A(f"**Budget:** frozen B* = {m['B_star_frozen']:.5f}; target = "
      f"{m['budget_target']:.5f} ({m['budget_target_note']}); per-k tol "
      f"|achieved-target|/target < {m['budget_tol']}.  ")
    A(f"**Denominator:** {m['denom_note']} baseline acc "
      f"{m['denom_baseline_acc']*100:.1f}%, FV E_all acc "
      f"{m['denom_fv_eall_acc']*100:.1f}%.\n")

    A(f"## DECISION: **{d['class']}**\n")
    A(f"Rule (plan Amendment 4, CI bounds only): {d['rule']}\n")
    A(f"- SURVIVES steelman = **{d['survives_steelman']}** "
      f"(non-degenerate k<=16 rungs: {d['n_nondegen_bounded_rungs']})")
    A(f"- static-null ARTIFACT = **{d['static_null_artifact']}**\n")

    A(f"## Validity guards (harness untrustworthy if any fails)\n")
    ua = sa["upper_anchor_rho_full_gt_rho0"]
    rf = sa["rho_full"]; r0 = sa["rho_0"]; sn = sa["static_null_rho"]
    A(f"- **(a) upper anchor** rho_full > rho_0 = **{ua}**  "
      f"(rho_full = {rf['point']:.3f} [{rf['ci_lo']:.3f}, {rf['ci_hi']:.3f}]; "
      f"rho_0 = {r0['point']:.3f} [{r0['ci_lo']:.3f}, {r0['ci_hi']:.3f}]). "
      f"If FALSE the map class is not learning the steered field -> the "
      f"'bounded interfaces fail' conclusion would be vacuous (debug target Y).")
    A(f"- **(b) lower anchor** rho_0 in static-null boot band "
      f"[{sn['ci_lo']:.3f}, {sn['ci_hi']:.3f}] = "
      f"**{sa['lower_anchor_rho0_in_static_band']}**  "
      f"(k=0 == best-constant-bias == static-null analogue; determinism / "
      f"budget-match check).")
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
    A(f"rho = E(null_k)/E(FV), bootstrap 95% CI. Genuine SURVIVES iff rho_hi "
      f"<= 0.30 for every non-degenerate bounded (k<=16) rung.\n")
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
