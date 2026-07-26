"""Output-wall resolver for the AxBench/ReFT-r1 wedding control.

This follows Amendment 13. It keeps the ReFT cell/prompt split fixed and
tests whether broader signed output controllers can cleanly reproduce the
wedding effect before the repetition wall.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_axbench_reft_canary import (  # noqa: E402
    DEFAULT_REFT_DIR,
    build_neutral_prompts,
    concept_record,
    layer_add_hook,
    load_reft_artifact,
    render_chat,
    score_condition,
)
from run_axbench_reft_control import (  # noqa: E402
    base_continuations,
    base_probs_on_s,
    bootstrap_rho,
    calibrate_scalar,
    prompt_baseline_texts,
    score_texts,
    sparse_bias_texts,
    sparse_kl_from_probs,
    token_string,
    wedding_lexicon_token_ids,
)


_REPO = _HERE.parents[2]
DEFAULT_OUTDIR = (
    _REPO / "runs/steering-content-audit/2026-07-09-axbench-reft-wedding-wall-resolver")


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_controller(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(pos|signed|bal|baleq)(\d+)", name.strip())
    if not m:
        raise SystemExit(
            f"unsupported controller {name!r}; use e.g. pos100,signed500,bal250,baleq250")
    return m.group(1), int(m.group(2))


def nonblank_token(tokenizer, token_id: int) -> bool:
    text = token_string(tokenizer, token_id)
    return bool(text) and not text.isspace()


def normalize_weights(raw: Sequence[float]) -> tuple[list[float], float]:
    rms = math.sqrt(sum(float(w) * float(w) for w in raw) / max(len(raw), 1))
    return [float(w) / max(rms, 1e-12) for w in raw], rms


def choose_controller_tokens(tokenizer, mean_delta: torch.Tensor, controller: str) -> dict:
    mode, top_k = parse_controller(controller)
    vals = mean_delta.float()
    special = set(int(x) for x in (tokenizer.all_special_ids or []))
    wedding_ids = wedding_lexicon_token_ids(tokenizer)
    ids: list[int] = []
    source_raw: list[float] | None = None

    if mode == "pos":
        order = torch.argsort(vals, descending=True).tolist()
        for tid in order:
            tid = int(tid)
            if tid in special or not nonblank_token(tokenizer, tid):
                continue
            if float(vals[tid]) <= 0:
                continue
            ids.append(tid)
            if len(ids) >= top_k:
                break
        for tid in sorted(wedding_ids):
            if tid not in special and tid not in ids and float(vals[tid]) > 0:
                ids.append(int(tid))
        raw = [max(float(vals[tid]), 0.0) for tid in ids]
    elif mode == "signed":
        order = torch.argsort(torch.abs(vals), descending=True).tolist()
        for tid in order:
            tid = int(tid)
            if tid in special or not nonblank_token(tokenizer, tid):
                continue
            if abs(float(vals[tid])) <= 0:
                continue
            ids.append(tid)
            if len(ids) >= top_k:
                break
        for tid in sorted(wedding_ids):
            if tid not in special and tid not in ids and float(vals[tid]) > 0:
                ids.append(int(tid))
        raw = [float(vals[tid]) for tid in ids]
    else:
        pos_ids: list[int] = []
        neg_ids: list[int] = []
        for tid in torch.argsort(vals, descending=True).tolist():
            tid = int(tid)
            if tid in special or not nonblank_token(tokenizer, tid):
                continue
            if float(vals[tid]) <= 0:
                continue
            pos_ids.append(tid)
            if len(pos_ids) >= top_k:
                break
        for tid in torch.argsort(vals, descending=False).tolist():
            tid = int(tid)
            if tid in special or not nonblank_token(tokenizer, tid):
                continue
            if float(vals[tid]) >= 0:
                continue
            neg_ids.append(tid)
            if len(neg_ids) >= top_k:
                break
        ids = pos_ids + neg_ids
        for tid in sorted(wedding_ids):
            if tid not in special and tid not in ids and float(vals[tid]) > 0:
                ids.append(int(tid))
        source_raw = [float(vals[tid]) for tid in ids]
        if mode == "baleq":
            pos_rms = math.sqrt(
                sum(w * w for w in source_raw if w > 0) / max(sum(1 for w in source_raw if w > 0), 1))
            neg_rms = math.sqrt(
                sum(w * w for w in source_raw if w < 0) / max(sum(1 for w in source_raw if w < 0), 1))
            raw = [
                (w / max(pos_rms, 1e-12)) if w > 0 else (w / max(neg_rms, 1e-12))
                for w in source_raw
            ]
        else:
            raw = source_raw

    if source_raw is None:
        source_raw = list(raw)
    weights, rms = normalize_weights(raw)
    return {
        "controller": controller,
        "mode": mode,
        "top_k": top_k,
        "token_ids": ids,
        "weights_raw": raw,
        "weights": weights,
        "rms_raw": rms,
        "positive_weight_count": int(sum(1 for w in raw if w > 0)),
        "negative_weight_count": int(sum(1 for w in raw if w < 0)),
        "wedding_ids_positive": [int(tid) for tid in sorted(wedding_ids) if tid in ids and float(vals[tid]) > 0],
        "tokens": [
            {
                "id": int(tid),
                "text": token_string(tokenizer, int(tid)),
                "weight": float(weights[i]),
                "raw_delta": float(raw[i]),
                "source_delta": float(source_raw[i]),
            }
            for i, tid in enumerate(ids)
        ],
    }


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


def classify_wall(records: Sequence[dict], target_kl: float) -> dict:
    within_budget = [
        r for r in records
        if float(r["kl"]) <= target_kl * 1.01 and not r["gate"]["degenerate"]
    ]
    dissolved = [r for r in within_budget if r["rho"]["rho_lo"] >= 0.9]
    if dissolved:
        return {
            "class": "WALL-RESOLVED-DISSOLVED-CANDIDATE",
            "reason": "at least one clean predeclared controller/dose within matched KL has rho_lo >= 0.9",
            "selected": {
                "controller": dissolved[0]["controller"],
                "scale": dissolved[0]["scale"],
                "rho": dissolved[0]["rho"],
                "rate": dissolved[0]["rate"],
                "kl": dissolved[0]["kl"],
            },
        }

    if within_budget and max(r["rho"]["rho_hi"] for r in within_budget) <= 0.3:
        return {
            "class": "WALL-RESOLVED-SURVIVAL-SIGNAL",
            "reason": "all clean predeclared controller/doses within matched KL have rho_hi <= 0.3",
            "max_clean_rho_hi": max(r["rho"]["rho_hi"] for r in within_budget),
        }

    dirty_high = [
        r for r in records
        if r["gate"]["degenerate"] and r["rho"]["rho"] >= 0.5
    ]
    return {
        "class": "WALL-PERSISTS-OR-MIXED",
        "reason": "no clean within-budget dissolved candidate and clean frontier does not satisfy survival",
        "max_clean_rho_hi": max((r["rho"]["rho_hi"] for r in within_budget), default=float("nan")),
        "dirty_high_count": len(dirty_high),
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
    ap.add_argument("--controllers", default="pos100,pos500,signed500,signed1000")
    ap.add_argument("--dose-scales", default="0.50,0.55,0.60,0.65,0.70,0.75,1.00")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifact = load_reft_artifact(Path(args.reft_dir))
    weights: torch.Tensor = artifact["weights"]
    acts = artifact["acts"]
    metadata = artifact["metadata"]
    concept = concept_record(metadata, args.concept_id, weights.shape[0])
    max_act = float(acts.get("max_act", {}).get(
        str(args.concept_id), acts.get("thresholds", {}).get(str(args.concept_id), 1.0)))
    add_vec = args.factor * max_act * weights[args.concept_id].float()

    prompts = build_neutral_prompts(args.n_calib + args.n_eval, seed=args.seed)
    calib_prompts = prompts[:args.n_calib]
    eval_prompts = prompts[args.n_calib:args.n_calib + args.n_eval]
    controller_names = [x.strip() for x in args.controllers.split(",") if x.strip()]
    dose_scales = [float(x) for x in args.dose_scales.split(",") if x.strip()]

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
    log(f"target_kl={tf['target_kl']:.6f} positions={tf['positions']}")

    log("scoring eval base/ReFT/prompt")
    base_eval = score_condition(
        model, tokenizer, eval_prompts, layer=args.layer, add_vec=None,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        do_sample=False, temperature=1.0)
    reft_eval = score_condition(
        model, tokenizer, eval_prompts, layer=args.layer, add_vec=add_vec,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        do_sample=False, temperature=1.0, base_gate=base_eval["gate"])
    prompt_texts = prompt_baseline_texts(
        model, tokenizer, eval_prompts, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens)
    prompt_rec = score_texts(prompt_texts, base_gate=base_eval["gate"])
    prompt_rec["rho_like"] = bootstrap_rho(
        base_eval["hits"], reft_eval["hits"], prompt_rec["hits"],
        n_boot=args.n_boot, seed=args.seed + 4242)

    controller_results = []
    flat_records = []
    for ci, controller in enumerate(controller_names):
        log(f"controller={controller} selecting tokens")
        token_set = choose_controller_tokens(tokenizer, tf["mean_delta"], controller)
        p_s = base_probs_on_s(
            model, tokenizer, calib_prompts, calib_cont, token_ids=token_set["token_ids"])
        weights_t = torch.tensor(token_set["weights"], dtype=torch.float64)
        cal = calibrate_scalar(p_s, weights_t, tf["target_kl"])
        log(
            f"controller={controller} n_tokens={len(token_set['token_ids'])} "
            f"scalar={cal['scalar']:.6f} achieved={cal['achieved_kl']:.6f}")

        doses = []
        for scale in dose_scales:
            scalar = cal["scalar"] * scale
            texts = sparse_bias_texts(
                model, tokenizer, eval_prompts, token_ids=token_set["token_ids"],
                weights=token_set["weights"], scalar=scalar, batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens)
            rec = score_texts(texts, base_gate=base_eval["gate"])
            rec["controller"] = controller
            rec["scale"] = scale
            rec["scalar"] = scalar
            rec["kl"] = sparse_kl_from_probs(p_s, weights_t, scalar)
            rec["rho"] = bootstrap_rho(
                base_eval["hits"], reft_eval["hits"], rec["hits"],
                n_boot=args.n_boot, seed=args.seed + ci * 10000 + int(scale * 1000))
            doses.append(rec)
            flat_records.append(rec)
            log(
                f"controller={controller} scale={scale:g} rate={rec['rate']:.3f} "
                f"clean={not rec['gate']['degenerate']} rho={rec['rho']['rho']:.3f} "
                f"kl={rec['kl']:.6f}")

        controller_results.append({
            "controller": controller,
            "token_set": token_set,
            "calibration": cal,
            "doses": doses,
        })
        del p_s, weights_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    verdict = classify_wall(flat_records, tf["target_kl"])
    result = {
        "verdict": verdict,
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
            "controllers": controller_names,
            "dose_scales": dose_scales,
            "scoring": "wedding lexicon hit on full continuation",
        },
        "teacher_forced": {
            "target_kl": tf["target_kl"],
            "positions": tf["positions"],
        },
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
        "prompt_baseline": prompt_rec,
        "controllers": controller_results,
    }
    (outdir / "results_wall_resolver.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    clean_records = [r for r in flat_records if not r["gate"]["degenerate"]]
    best_clean = max(clean_records, key=lambda r: r["rho"]["rho"], default=None)
    lines = [
        "# AxBench/ReFT-r1 output-wall resolver",
        "",
        f"- Base eval rate: {base_eval['rate']:.3f}",
        f"- ReFT eval rate: {reft_eval['rate']:.3f}",
        f"- Prompt baseline rate: {prompt_rec['rate']:.3f}",
        f"- Target KL: {tf['target_kl']:.6f}",
        f"- Verdict: `{verdict['class']}`",
    ]
    if best_clean is not None:
        lines.extend([
            f"- Best clean controller: `{best_clean['controller']}` scale `{best_clean['scale']}`",
            f"- Best clean rate: {best_clean['rate']:.3f}",
            (
                f"- Best clean rho: {best_clean['rho']['rho']:.3f} "
                f"[{best_clean['rho']['rho_lo']:.3f}, {best_clean['rho']['rho_hi']:.3f}]"
            ),
        ])
    lines.append("")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
