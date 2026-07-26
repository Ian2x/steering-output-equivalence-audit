"""Amendment 19: per-step full-vocabulary output-footprint distillation.

This is an append-only stress test for the two low-rho Pythia-2.8B cells.  The
student is fitted on calibration-only, teacher-forced native-minus-base logits
and is evaluated without the native intervention in closed-loop generation.

Run order:
  1. synthetic history-dependent output-policy anchor;
  2. function vectors;
  3. task vectors.

Exact same-prefix oracle reconstruction is a harness assertion only.  It is
algebraically guaranteed to reproduce native logits and is never a rho result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery as B  # noqa: E402
import fv_extract as FV  # noqa: E402
import run_a1 as A1  # noqa: E402
import taskvec as TV  # noqa: E402
from actlib.capture import get_blocks  # noqa: E402
from actlib.patching import _dynamic_patch_hook  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA_DIR = os.path.join(
    REPO, "data", "external", "function_vectors", "dataset_files",
    "abstractive")
FV_RUN = os.path.join(REPO, "runs", "steering-content-audit",
                      "2026-07-06-a1-anchor")
TASK_RUN = os.path.join(REPO, "runs", "steering-content-audit",
                        "2026-07-07-taskvec-arm")
DEFAULT_OUT = os.path.join(REPO, "runs", "steering-content-audit",
                           "2026-07-10-output-footprint-distill")
KL_GRID = [0.03, 0.07, 0.11, 0.15, 0.18]


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(json.dumps(list(a.shape)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    raise TypeError(type(x).__name__)


def gate_dict(g: B.GateResult) -> dict:
    return {
        "tripped": bool(g.tripped),
        "rep": float(g.rep_rate),
        "median_len": float(g.median_len),
        "nll": float(g.mean_nll),
        "reasons": list(g.reasons),
    }


def rate_dict(hits: Sequence[int], seed: int, n_boot: int) -> dict:
    p, lo, hi = B.bootstrap_rate_ci(hits, n_boot=n_boot, seed=seed)
    return {"rate": p, "ci_lo": lo, "ci_hi": hi,
            "count": int(sum(hits)), "n": len(hits)}


def rho_dict(control: Sequence[int], native: Sequence[int],
             base: Sequence[int], seed: int, n_boot: int) -> dict:
    p, lo, hi = B.bootstrap_ratio_ci(
        control, native, base, n_boot=n_boot, seed=seed)
    return {"point": p, "ci_lo": lo, "ci_hi": hi}


def block_hidden(output) -> torch.Tensor:
    hs = output[0] if isinstance(output, tuple) else output
    return hs


def forward_with_final_resid(model, input_ids: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return logits and final-block resid_post for one unpadded sequence."""
    store: Dict[str, torch.Tensor] = {}

    def hook(_mod, _args, output):
        store["hs"] = block_hidden(output).detach()

    handle = get_blocks(model)[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids, use_cache=False)
    finally:
        handle.remove()
    return out.logits[0].float(), store["hs"][0].float()


def prompt_and_cont_ids(tokenizer, prompt: str, cont: Sequence[int],
                        device: str) -> Tuple[torch.Tensor, int, List[int]]:
    p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    P = p_ids.shape[1]
    if cont:
        c_ids = torch.tensor([list(cont)], device=device)
        full = torch.cat([p_ids, c_ids], dim=1)
    else:
        full = p_ids
    pred = list(range(P - 1, P - 1 + len(cont)))
    return full, P, pred


def center_logits_delta(delta: torch.Tensor) -> torch.Tensor:
    return delta - delta.mean(dim=-1, keepdim=True)


@dataclass
class Arm:
    name: str
    model: object
    tokenizer: object
    device: str
    max_new_tokens: int
    calib_prompts: List[str]
    eval_prompts: List[str]
    calib_golds: Optional[List[str]]
    eval_golds: Optional[List[str]]
    native_generate: Callable[[str], str]
    native_logits: Callable[[torch.Tensor, int, Sequence[int]], torch.Tensor]
    hit: Callable[[str, Optional[str]], bool]
    frozen: Optional[dict]
    metadata: dict


class RidgeFootprint:
    """Calibration-only ridge map phi_t -> centered full-vocabulary delta."""

    def __init__(self, mu, sigma, intercept, V, shrink, C, device, meta):
        self.device = device
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)
        self.intercept = torch.as_tensor(
            intercept, dtype=torch.float32, device=device)
        self.V = torch.as_tensor(V, dtype=torch.float32, device=device)
        self.shrink = torch.as_tensor(
            shrink, dtype=torch.float32, device=device)
        self.C = C.to(device=device, dtype=torch.float32)
        self.meta = meta

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, device: str,
            ridge_frac: float = 1e-2) -> "RidgeFootprint":
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float32)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0) + 1e-6
        Z = (X - mu) / sigma
        intercept = Y.mean(axis=0, dtype=np.float64).astype(np.float32)
        Yc = Y - intercept[None, :]

        log(f"fit: SVD X {Z.shape}, full-vocab Y {Y.shape}")
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        tol = (S[0] if S.size else 1.0) * 1e-8
        rank = int(np.sum(S > tol))
        U = U[:, :rank]
        S = S[:rank]
        V = Vt[:rank, :].T
        lam = ridge_frac * float(np.mean(S * S))
        shrink = S / (S * S + lam)

        # C = U^T Yc.  Use the accelerator for the only large fit multiply.
        Ut = torch.from_numpy(U.T.astype(np.float32)).to(device)
        Yt = torch.from_numpy(Yc).to(device)
        C = Ut @ Yt
        del Ut, Yt

        meta = {
            "n_rows": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "vocab_size": int(Y.shape[1]),
            "effective_rank": rank,
            "ridge_frac": ridge_frac,
            "lambda": lam,
            "X_sha256": sha_array(X.astype(np.float32)),
            "Y_sha256": sha_array(Y),
            "mu_sha256": sha_array(mu.astype(np.float32)),
            "sigma_sha256": sha_array(sigma.astype(np.float32)),
            "singular_values": [float(v) for v in S],
        }
        return cls(mu, sigma, intercept, V.astype(np.float32),
                   shrink.astype(np.float32), C, device, meta)

    def predict(self, phi: torch.Tensor) -> torch.Tensor:
        one = phi.ndim == 1
        x = phi[None, :] if one else phi
        z = (x.float() - self.mu) / self.sigma
        coeff = (z @ self.V) * self.shrink
        out = self.intercept + coeff @ self.C
        return out[0] if one else out


def collect_base_rows(model, tokenizer, prompts: Sequence[str],
                      continuations: Sequence[Sequence[int]], device: str
                      ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    xs, logits, spans = [], [], []
    offset = 0
    for i, (prompt, cont) in enumerate(zip(prompts, continuations)):
        full, _P, pred = prompt_and_cont_ids(tokenizer, prompt, cont, device)
        base_logits, resid = forward_with_final_resid(model, full)
        if pred:
            xs.append(resid[pred].cpu().numpy().astype(np.float32))
            logits.append(base_logits[pred].cpu().numpy().astype(np.float32))
        spans.append((offset, offset + len(pred)))
        offset += len(pred)
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            log(f"base rows: {i+1}/{len(prompts)} prompts, {offset} steps")
    return np.concatenate(xs), np.concatenate(logits), spans


def collect_native_targets(arm: Arm, prompts: Sequence[str],
                           continuations: Sequence[Sequence[int]],
                           base_logits_rows: np.ndarray
                           ) -> np.ndarray:
    ys = []
    offset = 0
    for i, (prompt, cont) in enumerate(zip(prompts, continuations)):
        full, P, pred = prompt_and_cont_ids(
            arm.tokenizer, prompt, cont, arm.device)
        native_logits = arm.native_logits(full, P, pred)
        if pred:
            base = torch.from_numpy(
                base_logits_rows[offset:offset + len(pred)]).to(arm.device)
            delta = center_logits_delta(native_logits[pred].float() - base)
            ys.append(delta.cpu().numpy().astype(np.float32))
        offset += len(pred)
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            log(f"native targets: {i+1}/{len(prompts)} prompts")
    return np.concatenate(ys)


def mean_kl_rows(base: np.ndarray, delta: torch.Tensor, scalar: float,
                 device: str, return_rows: bool = False) -> Tuple[float, list]:
    vals: List[float] = []
    batch = 32
    for i in range(0, len(base), batch):
        b = torch.from_numpy(base[i:i + batch]).to(device).float()
        d = delta[i:i + batch].to(device).float() * float(scalar)
        lp = torch.log_softmax(b + d, dim=-1)
        lq = torch.log_softmax(b, dim=-1)
        k = (lp.exp() * (lp - lq)).sum(dim=-1)
        vals.extend(k.detach().cpu().tolist())
    return float(np.mean(vals)), vals if return_rows else []


def bisect_scalar(base: np.ndarray, unit_delta: torch.Tensor, target: float,
                  device: str) -> Tuple[float, float, list]:
    lo, hi = 0.0, 1.0
    achieved, _ = mean_kl_rows(base, unit_delta, hi, device)
    tries = 0
    while achieved < target and tries < 16:
        hi *= 2.0
        achieved, _ = mean_kl_rows(base, unit_delta, hi, device)
        tries += 1
    if achieved < target:
        raise RuntimeError(
            f"KL target {target} unreachable: hi={hi}, achieved={achieved}")
    for _ in range(30):
        mid = (lo + hi) / 2.0
        value, _ = mean_kl_rows(base, unit_delta, mid, device)
        if value < target:
            lo = mid
        else:
            hi = mid
    scalar = (lo + hi) / 2.0
    achieved, rows = mean_kl_rows(
        base, unit_delta, scalar, device, return_rows=True)
    return scalar, achieved, rows


def decode_ids(tokenizer, ids: Sequence[int]) -> str:
    return tokenizer.decode(torch.tensor(list(ids)), skip_special_tokens=True)


def base_generate_ids_text(model, tokenizer, prompt: str, max_new_tokens: int,
                           device: str) -> Tuple[List[int], str]:
    ids = B.base_generate_ids(model, tokenizer, prompt, max_new_tokens, device)
    return ids, decode_ids(tokenizer, ids)


def generate_with_policy(model, tokenizer, prompt: str, max_new_tokens: int,
                         device: str,
                         policy: Callable[[torch.Tensor], torch.Tensor]
                         ) -> Tuple[List[int], str]:
    """Closed-loop base-model generation with a full-vocab output policy."""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated: List[int] = []
    store: Dict[str, torch.Tensor] = {}

    def hook(_mod, _args, output):
        store["phi"] = block_hidden(output)[:, -1, :].detach()

    handle = get_blocks(model)[-1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        past = out.past_key_values
        for step in range(max_new_tokens):
            logits = out.logits[0, -1].float()
            delta = policy(store["phi"][0].float()).to(device).float()
            token = int((logits + delta).argmax())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
            cur = torch.tensor([[token]], device=device)
            with torch.no_grad():
                out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
    finally:
        handle.remove()
    return generated, decode_ids(tokenizer, generated)


def exact_oracle_check(arm: Arm, prompts: Sequence[str], n: int = 8) -> dict:
    rows = []
    for prompt in list(prompts)[:n]:
        p_ids = arm.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            arm.device)
        P = p_ids.shape[1]
        full = p_ids
        gen: List[int] = []
        native_gen: List[int] = []
        native_margins: List[float] = []
        same_path_tokens_match = True
        max_err = 0.0
        for _ in range(arm.max_new_tokens):
            base_logits, _resid = forward_with_final_resid(arm.model, full)
            pred = list(range(P - 1, full.shape[1]))
            native_logits = arm.native_logits(full, P, pred)
            delta = native_logits[-1].float() - base_logits[-1].float()
            recon = base_logits[-1].float() + delta
            max_err = max(max_err, float((recon - native_logits[-1]).abs().max()))
            token = int(recon.argmax())
            native_token = int(native_logits[-1].argmax())
            top2 = torch.topk(native_logits[-1].float(), k=2).values
            native_margins.append(float(top2[0] - top2[1]))
            native_gen.append(native_token)
            same_path_tokens_match = same_path_tokens_match and token == native_token
            gen.append(token)
            if token == arm.tokenizer.eos_token_id:
                break
            full = torch.cat([full, torch.tensor([[token]], device=arm.device)],
                             dim=1)
        oracle_text = decode_ids(arm.tokenizer, gen)
        same_path_native_text = decode_ids(arm.tokenizer, native_gen)
        cached_native_text = arm.native_generate(prompt)
        full_tokens = arm.tokenizer(
            same_path_native_text, add_special_tokens=False)["input_ids"]
        cached_tokens = arm.tokenizer(
            cached_native_text, add_special_tokens=False)["input_ids"]
        mismatch = None
        for j in range(max(len(full_tokens), len(cached_tokens))):
            a = full_tokens[j] if j < len(full_tokens) else None
            b = cached_tokens[j] if j < len(cached_tokens) else None
            if a != b:
                mismatch = j
                break
        rows.append({
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "oracle_text": oracle_text,
            "same_path_native_text": same_path_native_text,
            "same_path_token_match": same_path_tokens_match,
            "same_path_text_match": (
                oracle_text.strip() == same_path_native_text.strip()),
            "max_logit_error": max_err,
            "cached_native_text": cached_native_text,
            "cached_vs_recompute_text_match": (
                cached_native_text.strip() == same_path_native_text.strip()),
            "cached_vs_recompute_first_mismatch_token": mismatch,
            "same_path_native_top2_margin_at_mismatch": (
                native_margins[mismatch]
                if mismatch is not None and mismatch < len(native_margins)
                else None),
        })
    same_path_ok = all(r["same_path_token_match"] and
                       r["same_path_text_match"] for r in rows)
    max_error = max((r["max_logit_error"] for r in rows), default=0.0)
    return {
        "n": len(rows),
        "same_path_all_token_match": same_path_ok,
        "same_path_all_text_match": all(
            r["same_path_text_match"] for r in rows),
        "max_logit_error": max_error,
        "pass": same_path_ok and max_error <= 1e-5,
        "cached_vs_recompute_text_matches": int(sum(
            r["cached_vs_recompute_text_match"] for r in rows)),
        "cached_vs_recompute_all_text_match": all(
            r["cached_vs_recompute_text_match"] for r in rows),
        "rows": rows,
    }


def build_antonym_arm(kind: str, args) -> Arm:
    smoke = args.smoke
    model_name = "EleutherAI/pythia-160m" if smoke else "EleutherAI/pythia-2.8b"
    dtype = torch.float32 if smoke else torch.bfloat16
    n_eval = 30 if smoke else 200
    n_calib = 10 if smoke else 50
    max_tokens = 4 if smoke else 8
    seed = 20260706 if kind == "fv" else 20260707

    model, tok = B.load_model(model_name, device=args.device, dtype=dtype)
    pairs = FV.load_pairs(os.path.join(DATA_DIR, "antonym.json"))
    train_pool, eval_pairs = FV.split_pairs(pairs, n_eval, seed=seed)
    eval_pairs = eval_pairs[:n_eval]
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(eval_pairs))
    calib_pairs = [eval_pairs[i] for i in order[:n_calib]]
    heldout_pairs = [eval_pairs[i] for i in order[n_calib:]]
    calib_prompts = [FV.zero_shot_prompt(x) for x, _y in calib_pairs]
    eval_prompts = [FV.zero_shot_prompt(x) for x, _y in heldout_pairs]
    calib_golds = [y for _x, y in calib_pairs]
    eval_golds = [y for _x, y in heldout_pairs]

    frozen = None
    if kind == "fv":
        stage = os.path.join(
            FV_RUN, f"stage1_antonym_{model_name.split('/')[-1]}.pt")
        # Repo-local artifact produced by the frozen A1 driver. PyTorch 2.6
        # defaults weights_only=True, but this cache also stores NumPy metadata.
        blob = torch.load(stage, map_location="cpu", weights_only=False)
        vec = blob["fv"].float()
        layer = int(blob.get("edit_layer", 5 if smoke else 11))
        meth = A1.FVMethod(model, tok, layer, vec, device=args.device,
                           max_new_tokens=max_tokens)

        def native_logits(full, _P, pred):
            rb = torch.zeros(0, vec.numel(), device=args.device)
            state = {"cache_offset": 0, "targets": list(pred)}
            with torch.no_grad(), _dynamic_patch_hook(
                    model, "resid_post", layer, None, None,
                    "subspace_transplant", state, remove_subspace=rb,
                    add_vector=vec.to(args.device)):
                return model(full).logits[0].float()

        native_generate = lambda p: meth.generate(p, "all")
        if not smoke:
            frozen = json.load(open(os.path.join(FV_RUN, "results_full.json")))
        metadata = {"model": model_name, "layer": layer, "op": "add",
                    "native_regime": "all_generated", "seed": seed,
                    "vector_sha256": sha_array(vec.numpy())}
    else:
        if smoke:
            clean = FV.sample_icl_prompts(
                train_pool, 20, 10, seed=seed + 2, shuffle_labels=False)
            layer = 5
            vec = TV.mean_task_vector(
                model, tok, clean, layer, device=args.device, log=log).float()
        else:
            result = json.load(open(os.path.join(TASK_RUN, "results_full.json")))
            layer = int(result["meta"]["chosen_layer"])
            stage = json.load(open(os.path.join(
                TASK_RUN, "stage_antonym_full.json")))
            vec = torch.tensor(stage["theta"][str(layer)], dtype=torch.float32)
            frozen = result
        meth = TV.TaskVecMethod(model, tok, layer, vec, op="replace",
                                device=args.device,
                                max_new_tokens=max_tokens)

        def native_logits(full, P, _pred):
            return meth._forward_write_positions(
                full, vec.to(args.device), [P - 1]).float()

        native_generate = lambda p: meth.generate(p, "first")
        metadata = {"model": model_name, "layer": layer, "op": "replace",
                    "native_regime": "last_prompt_only", "seed": seed,
                    "vector_sha256": sha_array(vec.numpy())}

    return Arm(
        name=kind, model=model, tokenizer=tok, device=args.device,
        max_new_tokens=max_tokens, calib_prompts=calib_prompts,
        eval_prompts=eval_prompts, calib_golds=calib_golds,
        eval_golds=eval_golds, native_generate=native_generate,
        native_logits=native_logits,
        hit=lambda text, gold: A1.answer_hit(text, gold or ""),
        frozen=frozen, metadata=metadata)


def run_antonym_arm(kind: str, args) -> dict:
    arm = build_antonym_arm(kind, args)
    seed = int(arm.metadata["seed"])
    log(f"arm={kind}: base calibration continuations")
    calib_ids, calib_base_texts = [], []
    for i, p in enumerate(arm.calib_prompts):
        ids, text = base_generate_ids_text(
            arm.model, arm.tokenizer, p, arm.max_new_tokens, arm.device)
        calib_ids.append(ids)
        calib_base_texts.append(text)
        if (i + 1) % 10 == 0:
            log(f"calib base generate {i+1}/{len(arm.calib_prompts)}")

    X, base_rows, _spans = collect_base_rows(
        arm.model, arm.tokenizer, arm.calib_prompts, calib_ids, arm.device)
    Y = collect_native_targets(arm, arm.calib_prompts, calib_ids, base_rows)
    fit = RidgeFootprint.fit(X, Y, arm.device, ridge_frac=1e-2)
    with torch.no_grad():
        predicted = fit.predict(torch.from_numpy(X).to(arm.device))
    native_kl, native_kl_rows = mean_kl_rows(
        base_rows, torch.from_numpy(Y).to(arm.device), 1.0, arm.device, True)

    oracle = exact_oracle_check(arm, arm.calib_prompts, n=8)
    if not oracle["pass"]:
        raise RuntimeError(f"exact oracle guard failed: {oracle}")

    budgets = []
    grid = [0.03, 0.07] if args.smoke else KL_GRID
    for target in grid:
        scalar, achieved, rows = bisect_scalar(
            base_rows, predicted, target, arm.device)
        budgets.append({
            "target_kl": target,
            "scalar": scalar,
            "achieved_kl": achieved,
            "rel_error": abs(achieved - target) / target,
            "per_step_kl": {
                "min": float(np.min(rows)),
                "median": float(np.median(rows)),
                "p90": float(np.percentile(rows, 90)),
                "max": float(np.max(rows)),
            },
        })

    # Freeze all behavior generations before reading held-out native deltas.
    base_ids, base_texts, native_texts = [], [], []
    for i, p in enumerate(arm.eval_prompts):
        ids, text = base_generate_ids_text(
            arm.model, arm.tokenizer, p, arm.max_new_tokens, arm.device)
        base_ids.append(ids)
        base_texts.append(text)
        native_texts.append(arm.native_generate(p))
        if (i + 1) % 10 == 0:
            log(f"eval base/native {i+1}/{len(arm.eval_prompts)}")

    base_hits = [int(arm.hit(t, g)) for t, g in zip(base_texts, arm.eval_golds)]
    native_hits = [int(arm.hit(t, g))
                   for t, g in zip(native_texts, arm.eval_golds)]
    frozen_guard = {"available": arm.frozen is not None}
    if arm.frozen is not None:
        if kind == "fv":
            expected_base = int(round(
                arm.frozen["rates"]["baseline"]["rate"] * len(base_hits)))
            expected_native = int(round(
                arm.frozen["rates"]["E_all"]["rate"] * len(native_hits)))
        else:
            expected_base = int(round(
                arm.frozen["rates"]["baseline"]["rate"] * len(base_hits)))
            expected_native = int(round(
                arm.frozen["rates"]["E_native_first"]["rate"] * len(native_hits)))
        frozen_guard.update({
            "expected_base_count": expected_base,
            "observed_base_count": int(sum(base_hits)),
            "expected_native_count": expected_native,
            "observed_native_count": int(sum(native_hits)),
            "pass": (sum(base_hits) == expected_base and
                     sum(native_hits) == expected_native),
        })
        if not frozen_guard["pass"]:
            raise RuntimeError(f"frozen count guard failed: {frozen_guard}")

    if arm.frozen is not None:
        refs = arm.frozen["eval_baseline_refs"]
    else:
        refs = {
            "rep": float(np.mean([
                B.three_gram_rep_rate(t, arm.tokenizer) for t in base_texts])),
            "median_len": B.median_len_tokens(base_texts, arm.tokenizer),
            "nll": float(np.mean([
                B.mean_nll_under_model(
                    arm.model, arm.tokenizer, p, t, device=arm.device)
                for p, t in zip(arm.eval_prompts, base_texts)])),
        }

    for b in budgets:
        scalar = b["scalar"]
        texts = []
        for i, p in enumerate(arm.eval_prompts):
            _ids, text = generate_with_policy(
                arm.model, arm.tokenizer, p, arm.max_new_tokens, arm.device,
                lambda phi, s=scalar: s * fit.predict(phi))
            texts.append(text)
            if (i + 1) % 10 == 0:
                log(f"control KL={b['target_kl']:.2f}: "
                    f"{i+1}/{len(arm.eval_prompts)}")
        hits = [int(arm.hit(t, g)) for t, g in zip(texts, arm.eval_golds)]
        gate = B.degeneracy_gate(
            texts, arm.eval_prompts, refs["rep"], refs["median_len"],
            refs["nll"], arm.model, arm.tokenizer, device=arm.device)
        b["texts"] = texts
        b["hits"] = hits
        b["rate"] = rate_dict(hits, seed + 101, args.n_boot)
        b["rho"] = rho_dict(
            hits, native_hits, base_hits, seed + 102, args.n_boot)
        b["gate"] = gate_dict(gate)

    # Held-out teacher-footprint fidelity, diagnostic-only after behavior freeze.
    total_native_kl = 0.0
    total_resid_kl = 0.0
    total_steps = 0
    all_agree = 0
    flip_total = 0
    flip_recovered = 0
    for i, (prompt, cont) in enumerate(zip(arm.eval_prompts, base_ids)):
        Xh, Bh, _ = collect_base_rows(
            arm.model, arm.tokenizer, [prompt], [cont], arm.device)
        Yh = collect_native_targets(arm, [prompt], [cont], Bh)
        pred = fit.predict(torch.from_numpy(Xh).to(arm.device))
        b = torch.from_numpy(Bh).to(arm.device).float()
        y = torch.from_numpy(Yh).to(arm.device).float()
        native_lp = torch.log_softmax(b + y, dim=-1)
        base_lp = torch.log_softmax(b, dim=-1)
        pred_lp = torch.log_softmax(b + pred, dim=-1)
        nk = (native_lp.exp() * (native_lp - base_lp)).sum(dim=-1)
        rk = (native_lp.exp() * (native_lp - pred_lp)).sum(dim=-1)
        total_native_kl += float(nk.sum())
        total_resid_kl += float(rk.sum())
        narg = (b + y).argmax(dim=-1)
        barg = b.argmax(dim=-1)
        parg = (b + pred).argmax(dim=-1)
        all_agree += int((narg == parg).sum())
        flips = narg != barg
        flip_total += int(flips.sum())
        flip_recovered += int(((narg == parg) & flips).sum())
        total_steps += len(Xh)
        if (i + 1) % 10 == 0:
            log(f"heldout footprint diagnostic {i+1}/{len(arm.eval_prompts)}")

    coverage = 1.0 - total_resid_kl / max(total_native_kl, 1e-12)
    flip_recovery = flip_recovered / max(flip_total, 1)
    if coverage >= 0.80 and flip_recovery >= 0.80:
        fidelity_class = "strong"
    elif coverage >= 0.50 and flip_recovery >= 0.50:
        fidelity_class = "nontrivial"
    else:
        fidelity_class = "poor"

    clean = [b for b in budgets if not b["gate"]["tripped"]]
    artifact = any((b["rho"]["ci_lo"] is not None and
                    b["rho"]["ci_lo"] >= 0.9) for b in clean)
    all_low = bool(clean) and all(
        b["rho"]["ci_hi"] is not None and b["rho"]["ci_hi"] <= 0.3
        for b in clean)
    if artifact:
        verdict = "FOOTPRINT-DISTILLABLE"
    elif len(clean) >= 3 and all_low and fidelity_class == "strong":
        verdict = "FOOTPRINT-SURVIVES-STRONG"
    elif len(clean) >= 3 and all_low and fidelity_class == "nontrivial":
        verdict = "SURVIVES-BEHAVIORALLY-FIDELITY-LIMITED"
    elif len(clean) >= 3 and all_low and fidelity_class == "poor":
        verdict = "INCONCLUSIVE-CONTROLLER-FIT"
    else:
        verdict = "MIXED-OUTPUT-FOOTPRINT-WALL"

    result = {
        "meta": {
            "amendment": "19", "arm": kind, "smoke": args.smoke,
            "device": arm.device, "max_new_tokens": arm.max_new_tokens,
            "n_calib": len(arm.calib_prompts),
            "n_eval": len(arm.eval_prompts), "n_boot": args.n_boot,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **arm.metadata,
        },
        "fit": fit.meta,
        "oracle_guard": oracle,
        "frozen_count_guard": frozen_guard,
        "calibration": {
            "native_teacher_forced_kl": native_kl,
            "native_per_step_kl": {
                "min": float(np.min(native_kl_rows)),
                "median": float(np.median(native_kl_rows)),
                "p90": float(np.percentile(native_kl_rows, 90)),
                "max": float(np.max(native_kl_rows)),
            },
            "base_refs": refs,
        },
        "rates": {
            "base": rate_dict(base_hits, seed + 1, args.n_boot),
            "native": rate_dict(native_hits, seed + 2, args.n_boot),
        },
        "base_hits": base_hits,
        "native_hits": native_hits,
        "base_texts": base_texts,
        "native_texts": native_texts,
        "frontier": budgets,
        "heldout_footprint_fidelity": {
            "coverage": coverage,
            "mean_native_kl": total_native_kl / max(total_steps, 1),
            "mean_residual_kl": total_resid_kl / max(total_steps, 1),
            "all_top1_agreement": all_agree / max(total_steps, 1),
            "native_flip_positions": flip_total,
            "native_flip_recovered": flip_recovered,
            "changed_top1_recovery": flip_recovery,
            "class": fidelity_class,
        },
        "decision": {
            "verdict": verdict,
            "clean_targets": [b["target_kl"] for b in clean],
            "n_clean_targets": len(clean),
            "artifact_hit": artifact,
            "all_clean_rho_hi_le_0_3": all_low,
            "note": "Anchor pass is checked separately before paper interpretation.",
        },
    }
    return result


def run_synthetic(args) -> dict:
    """Matched-capacity affine history-dependent output-policy anchor."""
    smoke = args.smoke
    model_name = "gpt2"
    max_tokens = 8 if smoke else 32
    n_total = 24 if smoke else 200
    n_calib = 8 if smoke else 50
    seed = 20260710
    model, tok = B.load_model(model_name, device=args.device,
                              dtype=torch.float32)
    prompts = B.build_neutral_prompts(200)[:n_total]
    calib_prompts, eval_prompts = B.split_prompts(prompts, n_calib)

    calib_ids, calib_texts = [], []
    for p in calib_prompts:
        ids, text = base_generate_ids_text(
            model, tok, p, max_tokens, args.device)
        calib_ids.append(ids)
        calib_texts.append(text)
    X, base_rows, _ = collect_base_rows(
        model, tok, calib_prompts, calib_ids, args.device)

    rng = np.random.default_rng(seed)
    w = rng.normal(size=X.shape[1]).astype(np.float32)
    w /= np.linalg.norm(w) + 1e-12
    q = X @ w
    q_mu, q_sigma = float(q.mean()), float(q.std() + 1e-6)
    token_ids, token_strings = B.resolve_a0_token_ids(tok)
    v = np.zeros(base_rows.shape[1], dtype=np.float32)
    v[token_ids] = 1.0
    v -= v.mean()

    def teacher_delta(phi: torch.Tensor, scalar: float) -> torch.Tensor:
        wt = torch.from_numpy(w).to(args.device)
        vt = torch.from_numpy(v).to(args.device)
        z = (torch.dot(phi.float(), wt) - q_mu) / q_sigma
        return float(scalar) * (1.0 + 0.05 * z) * vt

    base_rep = float(np.mean([
        B.three_gram_rep_rate(t, tok) for t in calib_texts]))
    base_len = B.median_len_tokens(calib_texts, tok)
    base_nll = float(np.mean([
        B.mean_nll_under_model(model, tok, p, t, device=args.device)
        for p, t in zip(calib_prompts, calib_texts)]))
    base_cal_hits = [int(B.wedding_topic_hit(t)) for t in calib_texts]
    scalar_grid = [1.0, 2.0, 3.0, 4.0] if smoke else [
        0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
    selection = []
    chosen = None
    for s in scalar_grid:
        texts = [generate_with_policy(
            model, tok, p, max_tokens, args.device,
            lambda phi, ss=s: teacher_delta(phi, ss))[1]
                 for p in calib_prompts]
        hits = [int(B.wedding_topic_hit(t)) for t in texts]
        gate = B.degeneracy_gate(
            texts, calib_prompts, base_rep, base_len, base_nll,
            model, tok, device=args.device)
        gain = float(np.mean(hits) - np.mean(base_cal_hits))
        selection.append({"scalar": s, "gain": gain, "hits": hits,
                          "gate": gate_dict(gate)})
        threshold = 0.10 if smoke else 0.30
        if chosen is None and gain >= threshold and not gate.tripped:
            chosen = s
    if chosen is None:
        raise RuntimeError(f"synthetic teacher selection failed: {selection}")

    amp = 1.0 + 0.05 * ((X @ w - q_mu) / q_sigma)
    Y = (chosen * amp[:, None] * v[None, :]).astype(np.float32)
    fit = RidgeFootprint.fit(X, Y, args.device, ridge_frac=1e-2)
    predicted = fit.predict(torch.from_numpy(X).to(args.device))
    Bstar, native_rows = mean_kl_rows(
        base_rows, torch.from_numpy(Y).to(args.device), 1.0,
        args.device, True)
    scalar, achieved, student_rows = bisect_scalar(
        base_rows, predicted, Bstar, args.device)

    base_texts, teacher_texts, student_texts = [], [], []
    for p in eval_prompts:
        base_texts.append(B.base_generate(
            model, tok, p, max_tokens, args.device))
        teacher_texts.append(generate_with_policy(
            model, tok, p, max_tokens, args.device,
            lambda phi, ss=chosen: teacher_delta(phi, ss))[1])
        student_texts.append(generate_with_policy(
            model, tok, p, max_tokens, args.device,
            lambda phi, ss=scalar: ss * fit.predict(phi))[1])
    base_hits = [int(B.wedding_topic_hit(t)) for t in base_texts]
    teacher_hits = [int(B.wedding_topic_hit(t)) for t in teacher_texts]
    student_hits = [int(B.wedding_topic_hit(t)) for t in student_texts]
    refs = {
        "rep": float(np.mean([
            B.three_gram_rep_rate(t, tok) for t in base_texts])),
        "median_len": B.median_len_tokens(base_texts, tok),
        "nll": float(np.mean([
            B.mean_nll_under_model(model, tok, p, t, device=args.device)
            for p, t in zip(eval_prompts, base_texts)])),
    }
    teacher_gate = B.degeneracy_gate(
        teacher_texts, eval_prompts, refs["rep"], refs["median_len"],
        refs["nll"], model, tok, device=args.device)
    student_gate = B.degeneracy_gate(
        student_texts, eval_prompts, refs["rep"], refs["median_len"],
        refs["nll"], model, tok, device=args.device)
    rho = rho_dict(student_hits, teacher_hits, base_hits,
                   seed + 4, args.n_boot)
    effect = float(np.mean(teacher_hits) - np.mean(base_hits))
    floor_resolution = 1.0 / max(len(eval_prompts), 1)
    passed = (not teacher_gate.tripped and not student_gate.tripped and
              effect >= 3 * floor_resolution and
              rho["ci_lo"] is not None and rho["ci_lo"] >= 0.9)
    # For a synthetic teacher the exact oracle is literally the same policy.
    oracle_rows = []
    for p in calib_prompts[:8]:
        ids1, text1 = generate_with_policy(
            model, tok, p, max_tokens, args.device,
            lambda phi, ss=chosen: teacher_delta(phi, ss))
        ids2, text2 = generate_with_policy(
            model, tok, p, max_tokens, args.device,
            lambda phi, ss=chosen: teacher_delta(phi, ss))
        oracle_rows.append({"token_match": ids1 == ids2,
                            "text_match": text1.strip() == text2.strip()})
    oracle = {"n": len(oracle_rows),
              "all_text_match": all(r["text_match"] for r in oracle_rows),
              "all_token_match": all(r["token_match"] for r in oracle_rows),
              "max_logit_error": 0.0,
              "pass": all(r["token_match"] for r in oracle_rows),
              "rows": oracle_rows}
    smoke_pass = (oracle["pass"] and
                  abs(achieved - Bstar) / max(Bstar, 1e-12) < 0.02)

    return {
        "meta": {"amendment": "19", "arm": "synthetic", "smoke": smoke,
                 "model": model_name, "device": args.device,
                 "n_calib": len(calib_prompts), "n_eval": len(eval_prompts),
                 "max_new_tokens": max_tokens, "seed": seed,
                 "timestamp": datetime.now(timezone.utc).isoformat()},
        "teacher": {"form": "affine current-residual -> wedding-token bias",
                    "w_sha256": sha_array(w), "v_sha256": sha_array(v),
                    "token_ids": token_ids, "token_strings": token_strings,
                    "q_mean": q_mu, "q_std": q_sigma,
                    "selection": selection, "chosen_scalar": chosen},
        "fit": fit.meta,
        "oracle_guard": oracle,
        "budget": {"target_kl": Bstar, "student_scalar": scalar,
                   "achieved_kl": achieved,
                   "rel_error": abs(achieved - Bstar) / max(Bstar, 1e-12),
                   "native_kl_min": float(np.min(native_rows)),
                   "native_kl_max": float(np.max(native_rows)),
                   "student_kl_min": float(np.min(student_rows)),
                   "student_kl_max": float(np.max(student_rows))},
        "rates": {"base": rate_dict(base_hits, seed + 1, args.n_boot),
                  "native": rate_dict(teacher_hits, seed + 2, args.n_boot),
                  "student": rate_dict(student_hits, seed + 3, args.n_boot)},
        "rho": rho,
        "gates": {"native": gate_dict(teacher_gate),
                  "student": gate_dict(student_gate)},
        "base_hits": base_hits, "native_hits": teacher_hits,
        "student_hits": student_hits, "base_texts": base_texts,
        "native_texts": teacher_texts, "student_texts": student_texts,
        "decision": {"verdict": (("SMOKE-PASS" if smoke_pass else "SMOKE-FAIL")
                                  if smoke else
                                  ("ANCHOR-PASS" if passed else "ANCHOR-FAIL")),
                     "pass": smoke_pass if smoke else passed,
                     "full_anchor_pass": passed,
                     "native_effect": effect,
                     "floor_resolution": floor_resolution,
                     "effect_ge_3x_floor_resolution":
                         effect >= 3 * floor_resolution,
                     "rho_lo_ge_0_9": rho["ci_lo"] is not None and
                         rho["ci_lo"] >= 0.9},
    }


def write_result(result: dict, outdir: str, arm: str, smoke: bool):
    os.makedirs(outdir, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    path = os.path.join(outdir, f"{arm}{suffix}.json")
    payload = json.dumps(result, indent=2, sort_keys=True, default=jsonable)
    with open(path, "w") as f:
        f.write(payload)
        f.write("\n")
    digest = hashlib.sha256((payload + "\n").encode()).hexdigest()
    log(f"wrote {path} sha256={digest}")
    return path, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["synthetic", "fv", "taskvec"],
                    required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_boot = min(args.n_boot, 300)
    torch.manual_seed(20260710)
    np.random.seed(20260710)
    t0 = time.time()
    log(f"Amendment 19 arm={args.arm} smoke={args.smoke} device={args.device}")
    if args.arm == "synthetic":
        result = run_synthetic(args)
    else:
        result = run_antonym_arm(args.arm, args)
    result["runtime_sec"] = time.time() - t0
    write_result(result, args.outdir, args.arm, args.smoke)
    log(f"done verdict={result['decision']['verdict']} "
        f"runtime={result['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
