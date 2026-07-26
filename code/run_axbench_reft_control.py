"""Matched output-interface control for the AxBench/ReFT-r1 canary.

This is the next-step control after run_axbench_reft_canary.py reproduces a
clean effect. It is not a general AxBench evaluator. It fixes the selected
ReFT cell and asks whether a sparse static logit bias at the same absolute
teacher-forced per-step KL reproduces the wedding-topic behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_axbench_reft_canary import (  # noqa: E402
    DEFAULT_REFT_DIR,
    build_neutral_prompts,
    concept_record,
    generation_gate,
    generate_texts,
    layer_add_hook,
    load_reft_artifact,
    load_wedding_lexicon,
    render_chat,
    score_condition,
    sha256_file,
    wedding_hit,
)


_REPO = _HERE.parents[2]
DEFAULT_OUTDIR = (
    _REPO / "runs/steering-content-audit/2026-07-09-axbench-reft-wedding-control")


def log(msg: str) -> None:
    print(msg, flush=True)


class StaticBiasProcessor(LogitsProcessor):
    def __init__(self, token_ids: Sequence[int], weights: Sequence[float], scalar: float):
        self.token_ids = torch.tensor(list(token_ids), dtype=torch.long)
        self.weights = torch.tensor(list(weights), dtype=torch.float32) * float(scalar)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        ids = self.token_ids.to(scores.device)
        vals = self.weights.to(device=scores.device, dtype=scores.dtype)
        scores[:, ids] = scores[:, ids] + vals
        return scores


def sparse_bias_texts(model, tokenizer, prompts: Sequence[str], *,
                      token_ids: Sequence[int], weights: Sequence[float],
                      scalar: float, batch_size: int,
                      max_new_tokens: int) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    proc = LogitsProcessorList([StaticBiasProcessor(token_ids, weights, scalar)])
    out: list[str] = []
    chat_prompts = [render_chat(tokenizer, p) for p in prompts]
    for i in range(0, len(chat_prompts), batch_size):
        enc = tokenizer(chat_prompts[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=proc,
        )
        cont = ids[:, enc["input_ids"].shape[1]:]
        out.extend(tokenizer.batch_decode(cont, skip_special_tokens=True))
    return out


def prompt_baseline_texts(model, tokenizer, prompts: Sequence[str], *,
                          batch_size: int, max_new_tokens: int) -> list[str]:
    prefixed = [
        "Use wedding-related metaphors or terms where they fit naturally.\n\n" + p
        for p in prompts
    ]
    return generate_texts(
        model, tokenizer, prefixed, add_vec=None, layer=20,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=1.0)


def bootstrap_rho(base_hits: Sequence[int], method_hits: Sequence[int],
                  control_hits: Sequence[int], *, n_boot: int,
                  seed: int) -> dict:
    base = np.array(base_hits, dtype=np.float64)
    method = np.array(method_hits, dtype=np.float64)
    control = np.array(control_hits, dtype=np.float64)
    effect_m = float(np.mean(method - base))
    effect_c = float(np.mean(control - base))
    rho = effect_c / effect_m if abs(effect_m) > 1e-12 else float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    n = len(base)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        em = float(np.mean(method[idx] - base[idx]))
        ec = float(np.mean(control[idx] - base[idx]))
        vals.append(ec / em if abs(em) > 1e-12 else np.nan)
    arr = np.array(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    lo, hi = (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))) if len(arr) else (float("nan"), float("nan"))
    return {
        "rho": rho,
        "rho_lo": lo,
        "rho_hi": hi,
        "effect_method": effect_m,
        "effect_control": effect_c,
        "n_boot": n_boot,
        "finite_boot": int(len(arr)),
    }


def token_string(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def wedding_lexicon_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for word in load_wedding_lexicon():
        for text in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            enc = tokenizer(text, add_special_tokens=False).input_ids
            ids.update(int(x) for x in enc)
    return ids


@torch.no_grad()
def base_continuations(model, tokenizer, prompts: Sequence[str], *,
                       max_new_tokens: int) -> list[list[int]]:
    outs: list[list[int]] = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    for prompt in prompts:
        rendered = render_chat(tokenizer, prompt)
        enc = tokenizer(rendered, return_tensors="pt").to(model.device)
        ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        cont = ids[0, enc["input_ids"].shape[1]:].detach().cpu().tolist()
        outs.append([int(x) for x in cont])
    return outs


@torch.no_grad()
def teacher_forced_stats(model, tokenizer, prompts: Sequence[str],
                         continuations: Sequence[Sequence[int]], *,
                         add_vec: torch.Tensor, layer: int) -> dict:
    vocab = int(model.config.vocab_size)
    sum_delta = torch.zeros(vocab, dtype=torch.float64)
    total_positions = 0
    total_kl = 0.0

    for prompt, cont in zip(prompts, continuations):
        if not cont:
            continue
        rendered = render_chat(tokenizer, prompt)
        prompt_ids = tokenizer(rendered, return_tensors="pt").input_ids[0].to(model.device)
        cont_ids = torch.tensor(list(cont), dtype=torch.long, device=model.device)
        full = torch.cat([prompt_ids, cont_ids]).unsqueeze(0)
        pred_slice = slice(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(cont))

        base_logits = model(full).logits[0, pred_slice, :].float()
        with layer_add_hook(model, layer, add_vec):
            steered_logits = model(full).logits[0, pred_slice, :].float()
        lp_base = torch.log_softmax(base_logits, dim=-1)
        lp_steer = torch.log_softmax(steered_logits, dim=-1)
        p_steer = torch.exp(lp_steer)
        kl = torch.sum(p_steer * (lp_steer - lp_base), dim=-1)
        total_kl += float(kl.sum().detach().cpu())
        sum_delta += (steered_logits - base_logits).double().sum(dim=0).detach().cpu()
        total_positions += int(base_logits.shape[0])

        del base_logits, steered_logits, lp_base, lp_steer, p_steer, kl, full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total_positions <= 0:
        raise SystemExit("no teacher-forced positions found")
    return {
        "target_kl": total_kl / total_positions,
        "mean_delta": sum_delta / total_positions,
        "positions": total_positions,
    }


@torch.no_grad()
def base_probs_on_s(model, tokenizer, prompts: Sequence[str],
                    continuations: Sequence[Sequence[int]], *,
                    token_ids: Sequence[int]) -> torch.Tensor:
    rows = []
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=model.device)
    for prompt, cont in zip(prompts, continuations):
        if not cont:
            continue
        rendered = render_chat(tokenizer, prompt)
        prompt_ids = tokenizer(rendered, return_tensors="pt").input_ids[0].to(model.device)
        cont_ids = torch.tensor(list(cont), dtype=torch.long, device=model.device)
        full = torch.cat([prompt_ids, cont_ids]).unsqueeze(0)
        pred_slice = slice(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(cont))
        logits = model(full).logits[0, pred_slice, :].float()
        log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
        probs = torch.exp(logits[:, ids] - log_z).detach().cpu()
        rows.append(probs)
        del logits, full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(rows, dim=0).double()


def sparse_kl_from_probs(p_s: torch.Tensor, weights: torch.Tensor, scalar: float) -> float:
    delta = weights.double() * float(scalar)
    exp_delta = torch.exp(delta)
    z = 1.0 + torch.sum(p_s * (exp_delta.view(1, -1) - 1.0), dim=1)
    q_s = p_s * exp_delta.view(1, -1) / z.view(-1, 1)
    kl = torch.sum(q_s * delta.view(1, -1), dim=1) - torch.log(z)
    return float(kl.mean().item())


def calibrate_scalar(p_s: torch.Tensor, weights: torch.Tensor, target_kl: float) -> dict:
    lo = 0.0
    hi = 1.0
    for _ in range(80):
        if sparse_kl_from_probs(p_s, weights, hi) >= target_kl:
            break
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if sparse_kl_from_probs(p_s, weights, mid) < target_kl:
            lo = mid
        else:
            hi = mid
    scalar = (lo + hi) / 2.0
    achieved = sparse_kl_from_probs(p_s, weights, scalar)
    return {
        "scalar": scalar,
        "achieved_kl": achieved,
        "target_kl": target_kl,
        "rel_err": abs(achieved - target_kl) / max(target_kl, 1e-12),
    }


def choose_token_set(tokenizer, mean_delta: torch.Tensor, *,
                     top_k: int) -> dict:
    special = set(int(x) for x in (tokenizer.all_special_ids or []))
    wedding_ids = wedding_lexicon_token_ids(tokenizer)
    vals = mean_delta.float()
    order = torch.argsort(vals, descending=True).tolist()
    top_ids = []
    for tid in order:
        if int(tid) in special:
            continue
        text = token_string(tokenizer, int(tid))
        if not text or text.isspace():
            continue
        if float(vals[int(tid)]) <= 0:
            continue
        top_ids.append(int(tid))
        if len(top_ids) >= top_k:
            break

    ids = []
    for tid in top_ids + sorted(wedding_ids):
        if tid in special:
            continue
        if tid not in ids and float(vals[tid]) > 0:
            ids.append(tid)
    weights = [float(vals[tid]) for tid in ids]
    rms = math.sqrt(sum(w * w for w in weights) / max(len(weights), 1))
    normed = [w / max(rms, 1e-12) for w in weights]
    return {
        "token_ids": ids,
        "weights_raw": weights,
        "weights": normed,
        "rms_raw": rms,
        "top_ids": top_ids,
        "wedding_ids_positive": [tid for tid in sorted(wedding_ids) if tid in ids],
        "tokens": [
            {"id": tid, "text": token_string(tokenizer, tid), "weight": normed[i], "raw_delta": weights[i]}
            for i, tid in enumerate(ids)
        ],
    }


def score_texts(texts: Sequence[str], base_gate: dict | None = None) -> dict:
    return {
        "rate": float(np.mean([wedding_hit(t) for t in texts])) if texts else 0.0,
        "hits": [int(wedding_hit(t)) for t in texts],
        "texts": list(texts),
        "gate": generation_gate(texts, base=base_gate),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--reft-dir", default=str(DEFAULT_REFT_DIR))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--concept-id", type=int, default=1028)
    ap.add_argument("--factor", type=float, default=1.2)
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--dose-scales", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifact = load_reft_artifact(Path(args.reft_dir))
    weights: torch.Tensor = artifact["weights"]
    acts = artifact["acts"]
    metadata = artifact["metadata"]
    concept = concept_record(metadata, args.concept_id, weights.shape[0])
    max_act = float(acts.get("max_act", {}).get(str(args.concept_id), acts.get("thresholds", {}).get(str(args.concept_id), 1.0)))
    add_vec = args.factor * max_act * weights[args.concept_id].float()

    prompts = build_neutral_prompts(args.n_calib + args.n_eval, seed=args.seed)
    calib_prompts = prompts[:args.n_calib]
    eval_prompts = prompts[args.n_calib:args.n_calib + args.n_eval]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    log(f"loading model {args.model} device={args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("generating calibration base continuations")
    calib_cont = base_continuations(model, tokenizer, calib_prompts, max_new_tokens=args.max_new_tokens)
    log("computing ReFT teacher-forced KL and logit footprint")
    tf = teacher_forced_stats(
        model, tokenizer, calib_prompts, calib_cont, add_vec=add_vec, layer=args.layer)
    token_set = choose_token_set(tokenizer, tf["mean_delta"], top_k=args.top_k)
    p_s = base_probs_on_s(model, tokenizer, calib_prompts, calib_cont, token_ids=token_set["token_ids"])
    weights_t = torch.tensor(token_set["weights"], dtype=torch.float64)
    cal = calibrate_scalar(p_s, weights_t, tf["target_kl"])
    log(f"target_kl={tf['target_kl']:.6f} achieved={cal['achieved_kl']:.6f} scalar={cal['scalar']:.6f}")

    log("scoring eval base/ReFT/control/prompt")
    base_eval = score_condition(
        model, tokenizer, eval_prompts, layer=args.layer, add_vec=None,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        do_sample=False, temperature=1.0)
    reft_eval = score_condition(
        model, tokenizer, eval_prompts, layer=args.layer, add_vec=add_vec,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        do_sample=False, temperature=1.0, base_gate=base_eval["gate"])

    doses = []
    for scale in [float(x) for x in args.dose_scales.split(",") if x.strip()]:
        scalar = cal["scalar"] * scale
        texts = sparse_bias_texts(
            model, tokenizer, eval_prompts, token_ids=token_set["token_ids"],
            weights=token_set["weights"], scalar=scalar, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens)
        rec = score_texts(texts, base_gate=base_eval["gate"])
        rec["scale"] = scale
        rec["scalar"] = scalar
        rec["kl"] = sparse_kl_from_probs(p_s, weights_t, scalar)
        rec["rho"] = bootstrap_rho(
            base_eval["hits"], reft_eval["hits"], rec["hits"],
            n_boot=args.n_boot, seed=args.seed + int(scale * 1000))
        doses.append(rec)
        log(f"bias scale={scale:g} rate={rec['rate']:.3f} clean={not rec['gate']['degenerate']} rho={rec['rho']['rho']:.3f}")

    prompt_texts = prompt_baseline_texts(
        model, tokenizer, eval_prompts, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens)
    prompt_rec = score_texts(prompt_texts, base_gate=base_eval["gate"])
    prompt_rec["rho_like"] = bootstrap_rho(
        base_eval["hits"], reft_eval["hits"], prompt_rec["hits"],
        n_boot=args.n_boot, seed=args.seed + 4242)

    matched = next((r for r in doses if abs(r["scale"] - 1.0) < 1e-9), doses[-1])
    result = {
        "verdict": {
            "class": "CONTROL-DISSOLVED-CANDIDATE" if (
                not matched["gate"]["degenerate"] and matched["rho"]["rho_lo"] >= 0.9
            ) else "CONTROL-NOT-DISSOLVED" if (
                not matched["gate"]["degenerate"] and matched["rho"]["rho_hi"] <= 0.3
            ) else "CONTROL-MIXED-OR-INCONCLUSIVE",
            "note": "Candidate control result only; add κ/cascade before final master-table row.",
        },
        "meta": {
            "model": args.model,
            "artifact_repo": "pyvene/gemma-reft-r1-2b-it-res",
            "artifact_dir": str(args.reft_dir),
            "artifact_sha256": artifact["sha256"],
            "device": str(model.device),
            "dtype": args.dtype,
            "layer": args.layer,
            "concept": concept,
            "factor": args.factor,
            "max_act": max_act,
            "n_calib": len(calib_prompts),
            "n_eval": len(eval_prompts),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "scoring": "wedding lexicon hit on full continuation",
        },
        "teacher_forced": {
            "target_kl": tf["target_kl"],
            "positions": tf["positions"],
            "calibration": cal,
        },
        "token_set": token_set,
        "base_eval": {
            "rate": base_eval["rate"],
            "hits": base_eval["hits"],
            "texts": base_eval["texts"],
            "gate": base_eval["gate"],
        },
        "reft_eval": {
            "rate": reft_eval["rate"],
            "hits": reft_eval["hits"],
            "texts": reft_eval["texts"],
            "gate": reft_eval["gate"],
        },
        "bias_doses": doses,
        "prompt_baseline": prompt_rec,
    }
    (outdir / "results_control.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# AxBench/ReFT-r1 output-control",
        "",
        f"- ReFT eval rate: {reft_eval['rate']:.3f}",
        f"- Matched-bias eval rate: {matched['rate']:.3f}",
        f"- Matched-bias rho: {matched['rho']['rho']:.3f} [{matched['rho']['rho_lo']:.3f}, {matched['rho']['rho_hi']:.3f}]",
        f"- Matched-bias gate clean: `{not matched['gate']['degenerate']}`",
        f"- Prompt baseline rate: {prompt_rec['rate']:.3f}",
        f"- Verdict: `{result['verdict']['class']}`",
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
