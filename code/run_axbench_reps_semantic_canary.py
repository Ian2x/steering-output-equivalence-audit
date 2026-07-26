"""AxBench RePS PreferenceVector semantic reproduce-first canary.

This is a bounded RePS feasibility branch for the same six Concept500 semantic
targets used in the ReFT-r1 and DiffMean panels. The public AxBench checkout has
RePS configs and code but no released PreferenceVector weights, and local
`pyvene` is absent, so this driver implements only the rank-1 PreferenceVector
path directly with a layer-output hook.

Generation is GPU-only and secret-free. Judging reuses the AxBench LMJudge
semantic harness from `run_axbench_reft_semantic_canary.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_REPO = _HERE.parents[2]

from run_axbench_reft_canary import render_chat  # noqa: E402
from run_axbench_reft_semantic_canary import (  # noqa: E402
    JudgeItem,
    bootstrap_ratio,
    component_means,
    generate_texts,
    judge_missing,
    keys_for,
    parse_floats,
    parse_ints,
    prompt_baseline_prompts,
    ratings_for_items,
    sample_prompts,
    score_generation_gate,
    sha256_obj,
    write_json,
)


DEFAULT_MODEL_DIR = _REPO / "data/external/hf_models/gemma-2-2b-it"
DEFAULT_TRAIN_DIR = (
    _REPO
    / "data/external/axbench/axbench/concept500/prod_2b_l20_v1/generate"
)
DEFAULT_DATA_DIR = (
    _REPO
    / "data/external/axbench/axbench/concept500/prod_2b_l20_v1/inference"
)
DEFAULT_OUTDIR = (
    _REPO
    / "runs/steering-content-audit/2026-07-09-axbench-reps-semantic-canary"
)
DEFAULT_CONFIG = (
    _REPO
    / "data/external/axbench/axbench/sweep/wuzhengx/reps/experiments/"
    "p_vector_dps_g2-2b_axbench.yaml"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def concept_records(train_df: pd.DataFrame, concept_ids: Sequence[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cid in concept_ids:
        rows = train_df[(train_df["concept_id"] == cid) & (train_df["category"] == "positive")]
        if rows.empty:
            raise SystemExit(f"no positive train rows for concept_id={cid}")
        row = rows.iloc[0]
        out.append({
            "concept_id": int(cid),
            "concept": str(row["output_concept"]),
            "concept_genre": str(row.get("concept_genre", "")),
        })
    return out


def select_positive_rows(
    train_df: pd.DataFrame,
    concept_id: int,
    *,
    n_train: int | None,
    seed: int,
) -> pd.DataFrame:
    df = train_df.reset_index(names="source_row")
    rows = df[(df["concept_id"] == concept_id) & (df["category"] == "positive")].copy()
    if rows.empty:
        raise SystemExit(f"no positive train rows for concept_id={concept_id}")
    if n_train is not None:
        if len(rows) < n_train:
            raise SystemExit(
                f"concept_id={concept_id} has only {len(rows)} positive rows; need {n_train}")
        rows = rows.sample(n=n_train, random_state=seed + concept_id)
    return rows.sort_values("source_row").reset_index(drop=True)


@torch.no_grad()
def generate_base_responses(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = [render_chat(tokenizer, p) for p in prompts]
    out: list[str] = []
    for i in range(0, len(rendered), batch_size):
        enc = tokenizer(rendered[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        cont = ids[:, enc["input_ids"].shape[1]:]
        out.extend(tokenizer.batch_decode(cont, skip_special_tokens=True))
    return out


def render_prompt_and_full(tokenizer, prompt: str, output: str) -> tuple[list[int], list[int]]:
    prompt_text = render_chat(tokenizer, prompt)
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + str(output) + eos
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    return [int(x) for x in prompt_ids], [int(x) for x in full_ids]


def encode_batch(
    tokenizer,
    items: Sequence[dict[str, Any]],
    *,
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded: list[dict[str, Any]] = []
    max_len = 0
    for item in items:
        prompt_ids, full_ids = render_prompt_and_full(
            tokenizer,
            str(item["input"]),
            str(item["output"]),
        )
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if all(x == -100 for x in labels):
            labels[-1] = full_ids[-1]
        encoded.append({
            "input_ids": full_ids,
            "labels": labels,
            "factor": float(item["factor"]),
        })
        max_len = max(max_len, len(full_ids))
    pad = int(tokenizer.pad_token_id)
    input_ids, labels, masks, factors = [], [], [], []
    for item in encoded:
        n = len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad] * (max_len - n))
        labels.append(item["labels"] + [-100] * (max_len - n))
        masks.append([1] * n + [0] * (max_len - n))
        factors.append(item["factor"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "factors": torch.tensor(factors, dtype=torch.float32, device=device),
    }


def batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100
    labels_safe = labels.masked_fill(~loss_mask, 0)
    token_logps = torch.gather(
        logits.log_softmax(-1),
        dim=2,
        index=labels_safe.unsqueeze(2),
    ).squeeze(2)
    seq_logps = (token_logps * loss_mask).sum(dim=1)
    lengths = loss_mask.sum(dim=1).clamp_min(1)
    return seq_logps, lengths


@contextmanager
def reps_hook(
    model,
    layer: int,
    vector: torch.Tensor,
    bias: torch.Tensor,
    factors: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
):
    if factors is None:
        yield
        return
    module = model.model.layers[layer]

    def hook(_mod, _args, output):
        hs = output[0] if isinstance(output, tuple) else output
        v = vector.to(device=hs.device, dtype=torch.float32)
        b = bias.to(device=hs.device, dtype=torch.float32)
        f = factors.to(device=hs.device, dtype=torch.float32).view(-1, 1, 1)
        mask = attention_mask.to(device=hs.device, dtype=torch.float32).unsqueeze(-1)
        pos_mask = (f != 0.0).to(torch.float32)
        zero_mask = (f == 0.0).to(torch.float32)
        hs_float = hs.to(torch.float32)
        latent = torch.relu(torch.matmul(hs_float, v) + b).unsqueeze(-1)
        v_norm_sq = torch.sum(v * v).clamp_min(1e-8)
        null_factor = -(latent / v_norm_sq) * zero_mask
        add_factor = (f + b) * pos_mask
        delta = (null_factor + add_factor) * v.view(1, 1, -1)
        delta = delta * mask
        new_hs = hs + delta.to(dtype=hs.dtype)
        if isinstance(output, tuple):
            return (new_hs,) + tuple(output[1:])
        return new_hs

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def scaled_simpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    chosen_lens: torch.Tensor,
    rejected_lens: torch.Tensor,
    *,
    simpo_scaler: float,
) -> torch.Tensor:
    ref_reverse = ref_rejected - ref_chosen
    scale = torch.maximum(ref_reverse * simpo_scaler, torch.ones_like(ref_reverse))
    chosen = (scale / chosen_lens.to(scale.dtype)) * policy_chosen
    rejected = (1.0 / rejected_lens.to(scale.dtype)) * policy_rejected
    return -F.logsigmoid(chosen - rejected).mean()


def train_reps_vector(
    model,
    tokenizer,
    examples: Sequence[dict[str, str]],
    *,
    layer: int,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    factors: Sequence[float],
    seed: int,
    simpo_scaler: float,
) -> dict[str, Any]:
    hidden = int(model.config.hidden_size)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bound = 1.0 / math.sqrt(hidden)
    initial = torch.empty(hidden, dtype=torch.float32).uniform_(-bound, bound, generator=gen)
    vector = initial.to(model.device).clone().detach()
    vector.requires_grad_(True)
    bias = torch.zeros((), device=model.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([vector, bias], lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    metrics: list[dict[str, Any]] = []

    for epoch in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            idxs = order[start:start + batch_size]
            chosen_items: list[dict[str, Any]] = []
            rejected_items: list[dict[str, Any]] = []
            for idx in idxs:
                ex = examples[idx]
                factor = float(rng.choice(list(factors)))
                chosen_items.append({
                    "input": ex["input"],
                    "output": ex["winning_output"],
                    "factor": factor,
                })
                rejected_items.append({
                    "input": ex["input"],
                    "output": ex["losing_output"],
                    "factor": factor,
                })
                chosen_items.append({
                    "input": ex["input"],
                    "output": ex["losing_output"],
                    "factor": 0.0,
                })
                rejected_items.append({
                    "input": ex["input"],
                    "output": ex["winning_output"],
                    "factor": 0.0,
                })

            chosen = encode_batch(tokenizer, chosen_items, max_length=max_length, device=model.device)
            rejected = encode_batch(tokenizer, rejected_items, max_length=max_length, device=model.device)
            all_input_ids = torch.cat([chosen["input_ids"], rejected["input_ids"]], dim=0)
            all_labels = torch.cat([chosen["labels"], rejected["labels"]], dim=0)
            all_mask = torch.cat([chosen["attention_mask"], rejected["attention_mask"]], dim=0)
            all_factors = torch.cat([chosen["factors"], rejected["factors"]], dim=0)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                ref_outputs = model(
                    input_ids=all_input_ids,
                    attention_mask=all_mask,
                    use_cache=False,
                )
                ref_logps, lens = batch_logps(ref_outputs.logits, all_labels)

            with reps_hook(model, layer, vector, bias, all_factors, all_mask):
                policy_outputs = model(
                    input_ids=all_input_ids,
                    attention_mask=all_mask,
                    use_cache=False,
                )
            policy_logps, _ = batch_logps(policy_outputs.logits, all_labels)
            n = len(chosen_items)
            loss = scaled_simpo_loss(
                policy_logps[:n],
                policy_logps[n:],
                ref_logps[:n],
                ref_logps[n:],
                lens[:n],
                lens[n:],
                simpo_scaler=simpo_scaler,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

            del ref_outputs, policy_outputs
            torch.cuda.empty_cache()
        metrics.append({
            "epoch": int(epoch + 1),
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "loss_last": float(losses[-1]) if losses else float("nan"),
            "vector_norm": float(torch.linalg.vector_norm(vector.detach()).cpu()),
            "bias": float(bias.detach().cpu()),
        })
    return {
        "initial_vector": initial.detach().cpu(),
        "vector": vector.detach().cpu(),
        "bias": float(bias.detach().cpu()),
        "metrics": metrics,
    }


def vector_vocab_probe(model, tokenizer, vector: torch.Tensor, *, k: int = 12) -> dict[str, Any]:
    weights = model.lm_head.weight.detach()
    scores = weights @ vector.to(device=weights.device, dtype=weights.dtype)
    scores = scores.detach().float().cpu()
    top_vals, top_idx = torch.topk(scores, k=k, largest=True, sorted=True)
    low_vals, low_idx = torch.topk(scores, k=k, largest=False, sorted=True)

    def rows(vals: torch.Tensor, idx: torch.Tensor) -> list[dict[str, Any]]:
        return [
            {"token_id": int(i), "token": tokenizer.decode([int(i)]), "score": float(v)}
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    return {"top_positive": rows(top_vals, top_idx), "top_negative": rows(low_vals, low_idx)}


def run_generate(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(args.train_dir)
    data_dir = Path(args.data_dir)
    train_df = pd.read_parquet(train_dir / "train_data.parquet")
    concept_ids = parse_ints(args.concept_ids)
    concepts = concept_records(train_df, concept_ids)
    factors = parse_floats(args.factors)
    train_factors = parse_floats(args.train_factors)
    n_total = args.n_calib + args.n_eval
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32

    log(f"loading model {args.model} device={args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    result: dict[str, Any] = {
        "meta": {
            "mode": "generate",
            "method": "AxBench RePS PreferenceVector",
            "method_label": "RePS",
            "method_slug": "reps",
            "model": args.model,
            "train_dir": str(train_dir),
            "data_dir": str(data_dir),
            "train_data_sha256": sha256_file(train_dir / "train_data.parquet"),
            "latent_eval_data_sha256": sha256_file(data_dir / "latent_eval_data.parquet"),
            "source_config_sha256": sha256_file(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None,
            "device": str(model.device),
            "dtype": args.dtype,
            "layer": args.layer,
            "concept_ids": concept_ids,
            "concepts": concepts,
            "factors": factors,
            "official_reps_factor_grid": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50],
            "n_calib": args.n_calib,
            "n_eval": args.n_eval,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "train_examples_per_concept": args.train_examples_per_concept,
            "train_epochs": args.train_epochs,
            "train_batch_size": args.train_batch_size,
            "train_lr": args.train_lr,
            "train_max_length": args.train_max_length,
            "train_factors": train_factors,
            "train_loss": "reference-free scaled SimPO over orig_add and orig_sub pairs",
            "dpo_losing_output": "local greedy Gemma-2-2B-IT response to the same positive train input",
            "intervention": (
                "direct hook implementation of AxBench PreferenceVectorIntervention: positive "
                "h <- h + (factor + bias) v; zero-factor suppression removes ReLU(h dot v + bias) "
                "projection along v"
            ),
            "scoring": "AxBench LMJudge semantic concept/instruction/fluency; judged in a separate local pass",
        },
        "concept_runs": [],
    }
    vector_sidecar: dict[str, np.ndarray] = {}
    vector_manifest: dict[str, Any] = {"entries": []}

    for concept in concepts:
        cid = int(concept["concept_id"])
        concept_text = str(concept["concept"])
        log(f"concept={cid} preparing RePS preference data")
        train_rows = select_positive_rows(
            train_df,
            cid,
            n_train=args.train_examples_per_concept,
            seed=args.seed,
        )
        prompts = [str(x) for x in train_rows["input"].tolist()]
        winning_outputs = [str(x) for x in train_rows["output"].tolist()]
        losing_outputs = generate_base_responses(
            model,
            tokenizer,
            prompts,
            batch_size=args.train_gen_batch_size,
            max_new_tokens=args.dpo_max_new_tokens,
        )
        examples = [
            {"input": p, "winning_output": w, "losing_output": l}
            for p, w, l in zip(prompts, winning_outputs, losing_outputs)
        ]
        log(f"concept={cid} training RePS vector")
        rec = train_reps_vector(
            model,
            tokenizer,
            examples,
            layer=args.layer,
            epochs=args.train_epochs,
            batch_size=args.train_batch_size,
            lr=args.train_lr,
            max_length=args.train_max_length,
            factors=train_factors,
            seed=args.seed + cid,
            simpo_scaler=args.simpo_scaler,
        )
        vector: torch.Tensor = rec["vector"].float()
        initial_vector: torch.Tensor = rec["initial_vector"].float()
        bias = float(rec["bias"])
        vector_norm = float(torch.linalg.vector_norm(vector).item())
        vector_sha = hashlib.sha256(vector.numpy().astype("float32").tobytes()).hexdigest()
        initial_sha = hashlib.sha256(
            initial_vector.numpy().astype("float32").tobytes()
        ).hexdigest()
        vector_sidecar[f"cid{cid}_vector"] = vector.numpy().astype("float32")
        vector_sidecar[f"cid{cid}_initial_vector"] = initial_vector.numpy().astype("float32")
        vector_sidecar[f"cid{cid}_bias"] = np.array([bias], dtype=np.float32)
        vector_manifest["entries"].append({
            "concept_id": cid,
            "vector_key": f"cid{cid}_vector",
            "initial_vector_key": f"cid{cid}_initial_vector",
            "bias_key": f"cid{cid}_bias",
            "vector_sha256": vector_sha,
            "initial_vector_sha256": initial_sha,
            "vector_norm": vector_norm,
            "bias": bias,
        })

        rows = sample_prompts(data_dir, cid, n_total=n_total, seed=args.seed)
        calib_rows = rows[:args.n_calib]
        eval_rows = rows[args.n_calib:]
        calib_prompts = [r["prompt"] for r in calib_rows]
        eval_prompts = [r["prompt"] for r in eval_rows]

        model.config.use_cache = True
        log(f"concept={cid} generating base/prompt baselines")
        base_calib = generate_texts(
            model, tokenizer, calib_prompts, add_vec=None, layer=args.layer,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
        base_eval = generate_texts(
            model, tokenizer, eval_prompts, add_vec=None, layer=args.layer,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
        base_calib_gate = score_generation_gate(base_calib)
        base_eval_gate = score_generation_gate(base_eval)
        prompt_calib = generate_texts(
            model, tokenizer, prompt_baseline_prompts(concept_text, calib_prompts),
            add_vec=None, layer=args.layer, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens)
        prompt_eval = generate_texts(
            model, tokenizer, prompt_baseline_prompts(concept_text, eval_prompts),
            add_vec=None, layer=args.layer, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens)

        sweep = []
        random_floor_sweep = []
        for factor in factors:
            add_vec = (float(factor) + bias) * vector
            log(f"concept={cid} factor={factor:g} generating RePS")
            calib_texts = generate_texts(
                model, tokenizer, calib_prompts, add_vec=add_vec, layer=args.layer,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
            eval_texts = generate_texts(
                model, tokenizer, eval_prompts, add_vec=add_vec, layer=args.layer,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
            sweep.append({
                "factor": float(factor),
                "calib_texts": calib_texts,
                "eval_texts": eval_texts,
                "calib_gate": score_generation_gate(calib_texts, base=base_calib_gate),
                "eval_gate": score_generation_gate(eval_texts, base=base_eval_gate),
            })
            floor_vec = float(factor) * initial_vector
            log(f"concept={cid} factor={factor:g} generating random floor")
            floor_calib_texts = generate_texts(
                model, tokenizer, calib_prompts, add_vec=floor_vec, layer=args.layer,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
            floor_eval_texts = generate_texts(
                model, tokenizer, eval_prompts, add_vec=floor_vec, layer=args.layer,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
            random_floor_sweep.append({
                "factor": float(factor),
                "calib_texts": floor_calib_texts,
                "eval_texts": floor_eval_texts,
                "calib_gate": score_generation_gate(floor_calib_texts, base=base_calib_gate),
                "eval_gate": score_generation_gate(floor_eval_texts, base=base_eval_gate),
            })

        result["concept_runs"].append({
            "concept": concept,
            "rows": rows,
            "reps_vector": {
                "source_train_rows": [int(x) for x in train_rows["source_row"].tolist()],
                "source_train_count": int(len(train_rows)),
                "losing_output_sha256": hashlib.sha256(
                    json.dumps(losing_outputs, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "initial_vector_sha256": initial_sha,
                "vector_norm": vector_norm,
                "vector_sha256": vector_sha,
                "bias": bias,
                "train_metrics": rec["metrics"],
                "vocab_probe": vector_vocab_probe(model, tokenizer, vector),
            },
            "base": {
                "calib_texts": base_calib,
                "eval_texts": base_eval,
                "calib_gate": base_calib_gate,
                "eval_gate": base_eval_gate,
            },
            "prompt_baseline": {
                "calib_texts": prompt_calib,
                "eval_texts": prompt_eval,
                "calib_gate": score_generation_gate(prompt_calib, base=base_calib_gate),
                "eval_gate": score_generation_gate(prompt_eval, base=base_eval_gate),
            },
            "sweep": sweep,
            "random_floor_sweep": random_floor_sweep,
        })
        model.config.use_cache = False

    vector_npz = outdir / "reps_vectors.npz"
    np.savez_compressed(vector_npz, **vector_sidecar)
    vector_manifest["npz_file"] = vector_npz.name
    vector_manifest["npz_sha256"] = sha256_file(vector_npz)
    write_json(outdir / "reps_vectors_manifest.json", vector_manifest)
    result["meta"]["vector_sidecar"] = {
        "npz_file": vector_npz.name,
        "npz_sha256": vector_manifest["npz_sha256"],
        "manifest_file": "reps_vectors_manifest.json",
        "manifest_sha256": sha256_file(outdir / "reps_vectors_manifest.json"),
        "entry_count": len(vector_manifest["entries"]),
    }
    write_json(outdir / "semantic_generations.json", result)
    write_json(outdir / "dry_run_manifest.json", {
        "generation_sha256": sha256_obj(result),
        "concept_ids": concept_ids,
        "factors": factors,
        "method": "AxBench RePS PreferenceVector",
        "n_texts": sum(
            len(c["base"]["calib_texts"]) + len(c["base"]["eval_texts"])
            + len(c["prompt_baseline"]["calib_texts"]) + len(c["prompt_baseline"]["eval_texts"])
            + sum(len(s["calib_texts"]) + len(s["eval_texts"]) for s in c["sweep"])
            + sum(len(s["calib_texts"]) + len(s["eval_texts"]) for s in c["random_floor_sweep"])
            for c in result["concept_runs"]
        ),
    })
    log(f"wrote {outdir / 'semantic_generations.json'}")


def run_dry(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(args.train_dir)
    data_dir = Path(args.data_dir)
    train_df = pd.read_parquet(train_dir / "train_data.parquet")
    concept_ids = parse_ints(args.concept_ids)
    concepts = concept_records(train_df, concept_ids)
    prompt_counts = {}
    rows = pd.read_parquet(data_dir / "latent_eval_data.parquet")
    for cid in concept_ids:
        prompt_counts[str(cid)] = int((rows["concept_id"] == cid).sum())
    train_counts = {
        str(cid): int(((train_df["concept_id"] == cid) & (train_df["category"] == "positive")).sum())
        for cid in concept_ids
    }
    factors = parse_floats(args.factors)
    manifest = {
        "mode": "dry",
        "method": "AxBench RePS PreferenceVector",
        "model": args.model,
        "train_dir": str(train_dir),
        "data_dir": str(data_dir),
        "train_data_sha256": sha256_file(train_dir / "train_data.parquet"),
        "latent_eval_data_sha256": sha256_file(data_dir / "latent_eval_data.parquet"),
        "source_config_sha256": sha256_file(DEFAULT_CONFIG) if DEFAULT_CONFIG.exists() else None,
        "concepts": concepts,
        "factors": factors,
        "train_factors": parse_floats(args.train_factors),
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "n_total_per_concept": args.n_calib + args.n_eval,
        "available_prompt_counts": prompt_counts,
        "train_positive_counts": train_counts,
        "train_examples_per_concept": args.train_examples_per_concept,
        "planned_generation_texts": len(concept_ids) * (2 + 2 * len(factors)) * (args.n_calib + args.n_eval),
        "planned_dpo_losing_outputs": len(concept_ids) * (
            args.train_examples_per_concept
            if args.train_examples_per_concept is not None
            else min(train_counts.values())
        ),
        "planned_judge_prompts": len(concept_ids) * (2 + 2 * len(factors)) * (args.n_calib + args.n_eval) * 3,
    }
    write_json(outdir / "semantic_dry_run.json", manifest)
    log(f"wrote {outdir / 'semantic_dry_run.json'}")


def collect_items(gens: dict) -> list[JudgeItem]:
    items: list[JudgeItem] = []
    for cr in gens["concept_runs"]:
        cid = int(cr["concept"]["concept_id"])
        concept = str(cr["concept"]["concept"])
        rows = cr["rows"]
        n_calib = int(gens["meta"]["n_calib"])
        for split, row_subset in (("calib", rows[:n_calib]), ("eval", rows[n_calib:])):
            prompts = [r["prompt"] for r in row_subset]
            for cond, texts in (
                ("base", cr["base"][f"{split}_texts"]),
                ("prompt", cr["prompt_baseline"][f"{split}_texts"]),
            ):
                for j, (prompt, text) in enumerate(zip(prompts, texts)):
                    items.append(JudgeItem(
                        key=f"cid{cid}:{split}:{cond}:i{j}",
                        concept=concept,
                        instruction=prompt,
                        text=text,
                    ))
            for sr in cr["sweep"]:
                cond = f"reps_f{float(sr['factor']):g}"
                for j, (prompt, text) in enumerate(zip(prompts, sr[f"{split}_texts"])):
                    items.append(JudgeItem(
                        key=f"cid{cid}:{split}:{cond}:i{j}",
                        concept=concept,
                        instruction=prompt,
                        text=text,
                    ))
            for sr in cr["random_floor_sweep"]:
                cond = f"floor_f{float(sr['factor']):g}"
                for j, (prompt, text) in enumerate(zip(prompts, sr[f"{split}_texts"])):
                    items.append(JudgeItem(
                        key=f"cid{cid}:{split}:{cond}:i{j}",
                        concept=concept,
                        instruction=prompt,
                        text=text,
                    ))
    return items


def row_summary(
    judged: dict[str, dict],
    *,
    cid: int,
    concept: str,
    split: str,
    cond: str,
    factor: float,
    n: int,
    gate: dict[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    base_keys = keys_for(cid, split, "base", n)
    cond_keys = keys_for(cid, split, cond, n)
    base_vals = [judged[k]["aggregate_norm"] for k in base_keys]
    cond_vals = [judged[k]["aggregate_norm"] for k in cond_keys]
    stats = bootstrap_ratio(base_vals, cond_vals, n_boot=n_boot, seed=seed)
    base_comp = component_means(judged, base_keys)
    cond_comp = component_means(judged, cond_keys)
    return {
        "concept_id": cid,
        "concept": concept,
        "factor": float(factor),
        "condition": cond,
        "base_mean": float(np.mean(base_vals)),
        "method_mean": float(np.mean(cond_vals)),
        "base_components": base_comp,
        "method_components": cond_comp,
        "instruction_drop": base_comp["instruction_mean"] - cond_comp["instruction_mean"],
        "fluency_drop": base_comp["fluency_mean"] - cond_comp["fluency_mean"],
        "effect": stats["effect_method"],
        "effect_lo": stats["effect_lo"],
        "effect_hi": stats["effect_hi"],
        "gate": gate,
    }


def summarize_judged(gens: dict, judged: dict[str, dict], *, n_boot: int, seed: int,
                     reproduce_threshold: float) -> dict[str, Any]:
    n_calib = int(gens["meta"]["n_calib"])
    n_eval = int(gens["meta"]["n_eval"])
    calib_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    floor_calib_rows: list[dict[str, Any]] = []
    floor_eval_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []

    for cr in gens["concept_runs"]:
        cid = int(cr["concept"]["concept_id"])
        concept = str(cr["concept"]["concept"])
        for sr in cr["sweep"]:
            factor = float(sr["factor"])
            cond = f"reps_f{factor:g}"
            calib_rows.append(row_summary(
                judged, cid=cid, concept=concept, split="calib", cond=cond,
                factor=factor, n=n_calib, gate=sr["calib_gate"], n_boot=n_boot,
                seed=seed + cid + int(factor * 100)))
            eval_rows.append(row_summary(
                judged, cid=cid, concept=concept, split="eval", cond=cond,
                factor=factor, n=n_eval, gate=sr["eval_gate"], n_boot=n_boot,
                seed=seed + 100000 + cid + int(factor * 100)))
        for sr in cr["random_floor_sweep"]:
            factor = float(sr["factor"])
            cond = f"floor_f{factor:g}"
            floor_calib_rows.append(row_summary(
                judged, cid=cid, concept=concept, split="calib", cond=cond,
                factor=factor, n=n_calib, gate=sr["calib_gate"], n_boot=n_boot,
                seed=seed + 200000 + cid + int(factor * 100)))
            floor_eval_rows.append(row_summary(
                judged, cid=cid, concept=concept, split="eval", cond=cond,
                factor=factor, n=n_eval, gate=sr["eval_gate"], n_boot=n_boot,
                seed=seed + 300000 + cid + int(factor * 100)))

        base_eval_keys = keys_for(cid, "eval", "base", n_eval)
        prompt_eval_keys = keys_for(cid, "eval", "prompt", n_eval)
        base_eval_vals = [judged[k]["aggregate_norm"] for k in base_eval_keys]
        prompt_eval_vals = [judged[k]["aggregate_norm"] for k in prompt_eval_keys]
        prompt_rows.append({
            "concept_id": cid,
            "concept": concept,
            "base_mean": float(np.mean(base_eval_vals)),
            "prompt_mean": float(np.mean(prompt_eval_vals)),
            "effect": float(np.mean(np.asarray(prompt_eval_vals) - np.asarray(base_eval_vals))),
            "gate": cr["prompt_baseline"]["eval_gate"],
        })

    clean_calib = [r for r in calib_rows if not r["gate"]["degenerate"]]
    chosen = max(clean_calib if clean_calib else calib_rows,
                 key=lambda r: (r["effect"], r["method_mean"], -r["factor"]))
    chosen_eval = next(
        r for r in eval_rows
        if r["concept_id"] == chosen["concept_id"] and abs(r["factor"] - chosen["factor"]) < 1e-9
    )

    per_concept: list[dict[str, Any]] = []
    for cr in gens["concept_runs"]:
        cid = int(cr["concept"]["concept_id"])
        candidates = [r for r in calib_rows if r["concept_id"] == cid and not r["gate"]["degenerate"]]
        if not candidates:
            candidates = [r for r in calib_rows if r["concept_id"] == cid]
        selected = max(candidates, key=lambda r: (r["effect"], r["method_mean"], -r["factor"]))
        selected_eval = next(
            r for r in eval_rows
            if r["concept_id"] == cid and abs(r["factor"] - selected["factor"]) < 1e-9
        )
        reproduced = (
            bool(not selected_eval["gate"]["degenerate"])
            and float(selected_eval["effect"]) >= reproduce_threshold
            and float(selected_eval["effect_lo"]) > 0.0
            and float(selected_eval["instruction_drop"]) <= 0.20
            and float(selected_eval["fluency_drop"]) <= 0.20
        )
        per_concept.append({
            "selected": selected,
            "selected_eval": selected_eval,
            "reproduced": bool(reproduced),
        })

    reproduced_count = sum(1 for x in per_concept if x["reproduced"])
    panel_reproduced = reproduced_count >= 2
    return {
        "verdict": {
            "class": (
                "REPS-SEMANTIC-PANEL-REPRODUCED"
                if panel_reproduced
                else "REPS-SEMANTIC-PANEL-NOT-REPRODUCED"
            ),
            "panel_reproduced": bool(panel_reproduced),
            "concepts_reproduced": int(reproduced_count),
            "concepts_total": int(len(per_concept)),
            "threshold_norm_aggregate_gain": reproduce_threshold,
            "max_instruction_drop": 0.20,
            "max_fluency_drop": 0.20,
            "note": (
                "Reproduce-first RePS semantic canary only; matched output-control "
                "and kappa are required before any master-table row."
            ),
        },
        "chosen": chosen,
        "chosen_eval": chosen_eval,
        "per_concept": per_concept,
        "calib_rows": sorted(calib_rows, key=lambda r: (r["effect"], r["method_mean"]), reverse=True),
        "eval_rows": sorted(eval_rows, key=lambda r: (r["effect"], r["method_mean"]), reverse=True),
        "random_floor_calib_rows": sorted(
            floor_calib_rows, key=lambda r: (r["effect"], r["method_mean"]), reverse=True),
        "random_floor_eval_rows": sorted(
            floor_eval_rows, key=lambda r: (r["effect"], r["method_mean"]), reverse=True),
        "prompt_eval_rows": sorted(prompt_rows, key=lambda r: r["effect"], reverse=True),
    }


def write_report(outdir: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    chosen = summary["chosen"]
    chosen_eval = summary["chosen_eval"]
    lines = [
        "# AxBench/RePS semantic canary",
        "",
        "This is a reproduce-first canary scored with AxBench-style LMJudge prompts, not a rho/kappa verdict.",
        "",
        f"- Judge model: `{result['meta']['judge_model']}`",
        f"- Candidate concepts: `{result['meta']['concept_ids']}`",
        f"- Factors: `{result['meta']['factors']}`",
        f"- Panel reproduced concepts: {summary['verdict']['concepts_reproduced']}/{summary['verdict']['concepts_total']}",
        f"- Chosen global concept: {chosen['concept_id']} - {chosen['concept']}",
        f"- Chosen factor: {chosen['factor']}",
        f"- Calibration normalized aggregate gain: {chosen['effect']:+.3f} [{chosen['effect_lo']:+.3f}, {chosen['effect_hi']:+.3f}]",
        f"- Eval normalized aggregate gain: {chosen_eval['effect']:+.3f} [{chosen_eval['effect_lo']:+.3f}, {chosen_eval['effect_hi']:+.3f}]",
        f"- Eval gate clean: `{not chosen_eval['gate']['degenerate']}`",
        f"- Eval instruction drop: {chosen_eval['instruction_drop']:+.3f}",
        f"- Eval fluency drop: {chosen_eval['fluency_drop']:+.3f}",
        f"- Verdict: `{summary['verdict']['class']}`",
        "",
        "Per-concept selected cells:",
        "",
        "| concept | factor | eval gain | eval lo | clean | instr drop | fluency drop | reproduced |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_concept"]:
        ev = row["selected_eval"]
        lines.append(
            f"| {ev['concept_id']} | {ev['factor']:.2f} | {ev['effect']:+.3f} | "
            f"{ev['effect_lo']:+.3f} | {not ev['gate']['degenerate']} | "
            f"{ev['instruction_drop']:+.3f} | {ev['fluency_drop']:+.3f} | "
            f"{row['reproduced']} |"
        )
    lines.extend([
        "",
        "Top calibration rows:",
        "",
        "| concept | factor | base | RePS | gain | instr drop | fluency drop | clean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["calib_rows"][:12]:
        lines.append(
            f"| {row['concept_id']} | {row['factor']:.2f} | {row['base_mean']:.3f} | "
            f"{row['method_mean']:.3f} | {row['effect']:+.3f} | "
            f"{row['instruction_drop']:+.3f} | {row['fluency_drop']:+.3f} | "
            f"{not row['gate']['degenerate']} |"
        )
    lines.extend([
        "",
        "Top random-floor eval rows:",
        "",
        "| concept | factor | base | floor | gain | clean |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["random_floor_eval_rows"][:8]:
        lines.append(
            f"| {row['concept_id']} | {row['factor']:.2f} | {row['base_mean']:.3f} | "
            f"{row['method_mean']:.3f} | {row['effect']:+.3f} | {not row['gate']['degenerate']} |"
        )
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_judge(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    gens_path = outdir / "semantic_generations.json"
    if not gens_path.exists():
        raise SystemExit(f"missing generations file: {gens_path}")
    gens = json.loads(gens_path.read_text(encoding="utf-8"))
    items = collect_items(gens)
    cache_path = outdir / "semantic_judge_cache.json"
    cache = asyncio.run(judge_missing(
        items,
        model=args.judge_model,
        cache_path=cache_path,
        batch_size=args.judge_batch_size,
    ))
    judged = ratings_for_items(items, cache, model=args.judge_model)
    summary = summarize_judged(
        gens,
        judged,
        n_boot=args.n_boot,
        seed=args.seed,
        reproduce_threshold=args.reproduce_threshold,
    )
    result = {
        "meta": {
            **gens["meta"],
            "mode": "judge",
            "judge_model": args.judge_model,
            "generation_sha256": sha256_obj(gens),
            "judge_cache_sha256": sha256_obj(cache),
            "metric": "normalized AxBench LMJudge harmonic aggregate in [0,1]",
        },
        "summary": summary,
        "judged": judged,
    }
    write_json(outdir / "semantic_judged.json", result)
    write_report(outdir, result)
    log(f"wrote {outdir / 'semantic_judged.json'}")
    log(
        f"verdict={summary['verdict']['class']} "
        f"panel={summary['verdict']['concepts_reproduced']}/{summary['verdict']['concepts_total']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "generate", "judge"], default="dry")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--concept-ids", default="13,47,96,146,157,288")
    ap.add_argument("--factors", default="2,4,6,8,10,12,14,16,18,20,25,30,40,50")
    ap.add_argument("--train-factors", default="2,4,6,8,10,12,14,16,18,20")
    ap.add_argument("--n-calib", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-examples-per-concept", type=int, default=72)
    ap.add_argument("--train-epochs", type=int, default=18)
    ap.add_argument("--train-batch-size", type=int, default=3)
    ap.add_argument("--train-gen-batch-size", type=int, default=8)
    ap.add_argument("--train-lr", type=float, default=0.04)
    ap.add_argument("--train-max-length", type=int, default=512)
    ap.add_argument("--dpo-max-new-tokens", type=int, default=192)
    ap.add_argument("--simpo-scaler", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--judge-batch-size", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--reproduce-threshold", type=float, default=0.20)
    args = ap.parse_args()

    if args.mode == "dry":
        run_dry(args)
    elif args.mode == "generate":
        run_generate(args)
    elif args.mode == "judge":
        run_judge(args)
    else:
        raise SystemExit(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
