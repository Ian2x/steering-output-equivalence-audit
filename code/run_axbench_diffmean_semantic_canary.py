"""AxBench DiffMean semantic reproduce-first canary.

This is the post-ReFT AxBench method-breadth branch. DiffMean is fit directly
from AxBench concept500 train data as a class-mean residual vector:

    v = mean(h | positive concept input+output examples)
        - mean(h | same-genre negative input+output examples)

with AxBench binarized chat-formatted train examples, activations collected at
the Gemma layer-20 block output, BOS/prefix token excluded, then L2-normalized.
Generation and LMJudge scoring reuse the Amendment 15 semantic-panel harness so
the ReFT and DiffMean reads are comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_REPO = _HERE.parents[2]

from run_axbench_reft_canary import layer_add_hook  # noqa: E402
from run_axbench_reft_semantic_canary import (  # noqa: E402
    generate_texts,
    parse_floats,
    parse_ints,
    prompt_baseline_prompts,
    run_judge,
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
    / "runs/steering-content-audit/2026-07-09-axbench-diffmean-semantic-canary"
)
DEFAULT_CONFIG_FILES = [
    _REPO / "data/external/axbench/axbench/sweep/wuzhengx/2b/l20/no_grad.yaml",
    _REPO / "data/external/axbench/axbench/sweep/wuzhengx/2b/l20/16k_diffmean.yaml",
]
HAS_SYSTEM_PROMPT_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
}


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


def chat_suffix_length(tokenizer) -> int:
    msg_a = [{"role": "user", "content": "1"}]
    msg_b = [{"role": "user", "content": "2"}]
    toks_a = chat_template_ids(tokenizer.apply_chat_template(msg_a, tokenize=True))
    toks_b = chat_template_ids(tokenizer.apply_chat_template(msg_b, tokenize=True))
    for i, (ta, tb) in enumerate(zip(reversed(toks_a), reversed(toks_b))):
        if ta != tb:
            return i
    return 0


def chat_template_ids(obj: Any) -> list[int]:
    if isinstance(obj, dict):
        obj = obj["input_ids"]
    elif hasattr(obj, "data") and isinstance(obj.data, dict) and "input_ids" in obj.data:
        obj = obj.data["input_ids"]
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if obj and isinstance(obj[0], list):
        obj = obj[0]
    return [int(x) for x in obj]


def axbench_binarized_texts(
    tokenizer,
    df: pd.DataFrame,
    *,
    model_name: str,
    is_chat_model: bool,
) -> list[str]:
    if not is_chat_model:
        return [str(row["input"]) + str(row["output"]) for _, row in df.iterrows()]

    suffix_len = chat_suffix_length(tokenizer)
    use_system = model_name in HAS_SYSTEM_PROMPT_MODELS
    out: list[str] = []
    for _, row in df.iterrows():
        messages = []
        if use_system:
            messages.append({"role": "system", "content": "You are a helpful assistant."})
        messages.extend([
            {"role": "user", "content": str(row["input"])},
            {"role": "assistant", "content": str(row["output"])},
        ])
        tokens = chat_template_ids(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        ))
        tokens = tokens[1:-suffix_len] if suffix_len > 0 else tokens[1:]
        out.append(tokenizer.decode(tokens))
    return out


def train_rows_for_concept(
    train_df: pd.DataFrame,
    concept_id: int,
    *,
    seed: int,
    n_pos: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = train_df.reset_index(names="source_row")
    pos = df[(df["concept_id"] == concept_id) & (df["category"] == "positive")].copy()
    if pos.empty:
        raise SystemExit(f"no positive rows for concept_id={concept_id}")
    genres = sorted({str(x) for x in pos["concept_genre"].dropna().tolist()})
    if len(genres) != 1:
        raise SystemExit(f"concept_id={concept_id} has non-unique concept genres: {genres}")
    neg_pool = df[
        (df["concept_id"] == -1)
        & (df["category"] == "negative")
        & (df["concept_genre"] == genres[0])
    ].copy()
    if n_pos is not None:
        if len(pos) < n_pos:
            raise SystemExit(
                f"concept_id={concept_id} has only {len(pos)} positive rows, need {n_pos}")
        pos = pos.sample(n=n_pos, random_state=seed + concept_id).sort_values("source_row")
    n_neg = len(pos)
    if len(neg_pool) < n_neg:
        raise SystemExit(
            f"negative pool has only {len(neg_pool)} rows, need {n_neg} for concept_id={concept_id}")
    rng = np.random.default_rng(seed + concept_id)
    neg_idx = rng.permutation(len(neg_pool))[:n_neg]
    neg = neg_pool.iloc[neg_idx].sort_values("source_row").copy()
    return pos.reset_index(drop=True), neg.reset_index(drop=True)


@contextmanager
def capture_layer_output(model, layer: int):
    module = model.model.layers[layer]
    box: dict[str, torch.Tensor] = {}

    def hook(_mod, _args, output):
        hs = output[0] if isinstance(output, tuple) else output
        box["hidden"] = hs.detach()
        return output

    handle = module.register_forward_hook(hook)
    try:
        yield box
    finally:
        handle.remove()


@torch.no_grad()
def layer_activation_mean(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    layer: int,
    batch_size: int,
    prefix_length: int,
    max_length: int,
) -> tuple[torch.Tensor, int]:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    total: torch.Tensor | None = None
    count = 0
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i:i + batch_size])
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with capture_layer_output(model, layer) as box:
            _ = model(**enc, use_cache=False)
        if "hidden" not in box:
            raise RuntimeError(f"layer hook did not capture activations for layer={layer}")
        hidden = box["hidden"]
        valid = enc["attention_mask"].bool()
        if prefix_length > 0:
            valid[:, :prefix_length] = False
        selected = hidden[valid]
        if selected.numel() == 0:
            continue
        selected = selected.float()
        batch_sum = selected.sum(dim=0).detach().cpu()
        total = batch_sum if total is None else total + batch_sum
        count += int(selected.shape[0])
    if total is None or count == 0:
        raise RuntimeError("no non-prefix train activations collected")
    return total / float(count), count


@torch.no_grad()
def max_diffmean_activation(
    model,
    tokenizer,
    prompts: Sequence[str],
    vector: torch.Tensor,
    *,
    layer: int,
    batch_size: int,
    prefix_length: int,
    max_length: int,
) -> dict[str, Any]:
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_act = -float("inf")
    mean_vals: list[float] = []
    token_count = 0
    vec = vector.to(dtype=torch.float32)
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(
            list(prompts[i:i + batch_size]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with capture_layer_output(model, layer) as box:
            _ = model(**enc, use_cache=False)
        if "hidden" not in box:
            raise RuntimeError(f"layer hook did not capture activations for layer={layer}")
        hidden = box["hidden"].float()
        valid = enc["attention_mask"].bool()
        if prefix_length > 0:
            valid[:, :prefix_length] = False
        selected = hidden[valid]
        if selected.numel() == 0:
            continue
        vals = (selected.detach().cpu() @ vec).numpy()
        token_count += int(vals.shape[0])
        max_act = max(max_act, float(np.max(vals)))
        mean_vals.append(float(np.mean(vals)))
    if token_count == 0:
        raise RuntimeError("no calibration activations collected for max_act")
    fallback_used = max_act <= 0.0 or not np.isfinite(max_act)
    return {
        "max_act": float(max_act if not fallback_used else 50.0),
        "raw_max_act": float(max_act),
        "fallback_used": bool(fallback_used),
        "mean_batch_activation": float(np.mean(mean_vals)) if mean_vals else float("nan"),
        "token_count": int(token_count),
    }


def vector_vocab_probe(model, tokenizer, vector: torch.Tensor, *, k: int = 12) -> dict[str, Any]:
    weights = model.lm_head.weight.detach()
    scores = weights @ vector.to(device=weights.device, dtype=weights.dtype)
    scores = scores.detach().float().cpu()
    top_vals, top_idx = torch.topk(scores, k=k, largest=True, sorted=True)
    low_vals, low_idx = torch.topk(scores, k=k, largest=False, sorted=True)

    def rows(vals: torch.Tensor, idx: torch.Tensor) -> list[dict[str, Any]]:
        return [
            {
                "token_id": int(i),
                "token": tokenizer.decode([int(i)]),
                "score": float(v),
            }
            for v, i in zip(vals.tolist(), idx.tolist())
        ]

    return {
        "top_positive": rows(top_vals, top_idx),
        "top_negative": rows(low_vals, low_idx),
    }


@torch.no_grad()
def fit_diffmean_vector(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    concept: dict[str, Any],
    *,
    seed: int,
    n_pos: int | None,
    layer: int,
    batch_size: int,
    prefix_length: int,
    max_length: int,
    model_name: str,
    is_chat_model: bool,
) -> dict[str, Any]:
    cid = int(concept["concept_id"])
    pos, neg = train_rows_for_concept(train_df, cid, seed=seed, n_pos=n_pos)
    pos_texts = axbench_binarized_texts(
        tokenizer,
        pos,
        model_name=model_name,
        is_chat_model=is_chat_model,
    )
    neg_texts = axbench_binarized_texts(
        tokenizer,
        neg,
        model_name=model_name,
        is_chat_model=is_chat_model,
    )
    pos_mean, pos_tokens = layer_activation_mean(
        model,
        tokenizer,
        pos_texts,
        layer=layer,
        batch_size=batch_size,
        prefix_length=prefix_length,
        max_length=max_length,
    )
    neg_mean, neg_tokens = layer_activation_mean(
        model,
        tokenizer,
        neg_texts,
        layer=layer,
        batch_size=batch_size,
        prefix_length=prefix_length,
        max_length=max_length,
    )
    raw_vec = pos_mean - neg_mean
    raw_norm = float(torch.linalg.vector_norm(raw_vec).item())
    if not np.isfinite(raw_norm) or raw_norm <= 0.0:
        raise RuntimeError(f"invalid DiffMean vector norm for concept_id={cid}: {raw_norm}")
    unit_vec = raw_vec / raw_norm
    vec_sha = hashlib.sha256(unit_vec.numpy().astype("float32").tobytes()).hexdigest()
    return {
        "concept_id": cid,
        "vector": unit_vec,
        "raw_norm": raw_norm,
        "unit_norm": float(torch.linalg.vector_norm(unit_vec).item()),
        "vector_sha256": vec_sha,
        "vocab_probe": vector_vocab_probe(model, tokenizer, unit_vec),
        "pos_rows": [int(x) for x in pos["source_row"].tolist()],
        "neg_rows": [int(x) for x in neg["source_row"].tolist()],
        "pos_genres": sorted({str(x) for x in pos["concept_genre"].dropna().tolist()}),
        "neg_genres": sorted({str(x) for x in neg["concept_genre"].dropna().tolist()}),
        "pos_count": int(len(pos)),
        "neg_count": int(len(neg)),
        "pos_token_count": int(pos_tokens),
        "neg_token_count": int(neg_tokens),
    }


def prompt_count_by_concept(data_dir: Path, concept_ids: Sequence[int]) -> dict[str, int]:
    rows = pd.read_parquet(data_dir / "latent_eval_data.parquet")
    return {str(cid): int((rows["concept_id"] == cid).sum()) for cid in concept_ids}


def run_generate(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(args.train_dir)
    data_dir = Path(args.data_dir)
    train_df = pd.read_parquet(train_dir / "train_data.parquet")
    concept_ids = parse_ints(args.concept_ids)
    concepts = concept_records(train_df, concept_ids)
    factors = parse_floats(args.factors)
    n_total = args.n_calib + args.n_eval
    dtype = (
        torch.bfloat16
        if args.dtype == "bf16"
        else torch.float16
        if args.dtype == "fp16"
        else torch.float32
    )

    log(f"loading model {args.model} device={args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    result: dict[str, Any] = {
        "meta": {
            "mode": "generate",
            "method": "AxBench DiffMean",
            "method_label": "DiffMean",
            "method_slug": "diffmean",
            "model": args.model,
            "train_dir": str(train_dir),
            "data_dir": str(data_dir),
            "train_data_sha256": sha256_file(train_dir / "train_data.parquet"),
            "latent_eval_data_sha256": sha256_file(data_dir / "latent_eval_data.parquet"),
            "source_config_sha256": {
                str(p): sha256_file(p) for p in DEFAULT_CONFIG_FILES if p.exists()
            },
            "device": str(model.device),
            "dtype": args.dtype,
            "layer": args.layer,
            "concept_ids": concept_ids,
            "concepts": concepts,
            "factors": factors,
            "official_diffmean_grid": [
                0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
                1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
            ],
            "n_calib": args.n_calib,
            "n_eval": args.n_eval,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "train_batch_size": args.train_batch_size,
            "train_prefix_length": args.train_prefix_length,
            "train_max_length": args.train_max_length,
            "train_n_pos": args.train_n_pos,
            "train_is_chat_model": bool(args.train_is_chat_model),
            "model_name_for_chat": args.model_name_for_chat,
            "vector_construction": (
                "mean(layer20 block-output activations over AxBench binarized chat-formatted "
                "input+output positive train tokens after prefix_length) minus equal-count "
                "same-genre negative train tokens; L2-normalized"
            ),
            "max_act_construction": (
                "calibration-only maximum dot activation of the unit DiffMean vector on "
                "AxBench-prepared calib input+output examples; fallback 50 if nonpositive"
            ),
            "scoring": "AxBench LMJudge semantic concept/instruction/fluency; judged in a separate local pass",
        },
        "concept_runs": [],
    }

    vector_records: dict[str, Any] = {}
    for concept in concepts:
        cid = int(concept["concept_id"])
        concept_text = str(concept["concept"])
        log(f"concept={cid} fitting DiffMean vector")
        vec_rec = fit_diffmean_vector(
            model,
            tokenizer,
            train_df,
            concept,
            seed=args.seed,
            n_pos=args.train_n_pos,
            layer=args.layer,
            batch_size=args.train_batch_size,
            prefix_length=args.train_prefix_length,
            max_length=args.train_max_length,
            model_name=args.model_name_for_chat,
            is_chat_model=args.train_is_chat_model,
        )
        unit_vec: torch.Tensor = vec_rec.pop("vector")
        vector_records[str(cid)] = vec_rec

        rows = sample_prompts(data_dir, cid, n_total=n_total, seed=args.seed)
        calib_rows = rows[:args.n_calib]
        eval_rows = rows[args.n_calib:]
        calib_prompts = [r["prompt"] for r in calib_rows]
        eval_prompts = [r["prompt"] for r in eval_rows]
        eval_df = pd.read_parquet(data_dir / "latent_eval_data.parquet")
        calib_eval_df = eval_df[eval_df["concept_id"] == cid].reset_index(drop=True).iloc[
            [int(r["source_row"]) for r in calib_rows]
        ].copy()
        calib_prepared = axbench_binarized_texts(
            tokenizer,
            calib_eval_df,
            model_name=args.model_name_for_chat,
            is_chat_model=args.train_is_chat_model,
        )
        max_act_rec = max_diffmean_activation(
            model,
            tokenizer,
            calib_prepared,
            unit_vec,
            layer=args.layer,
            batch_size=args.train_batch_size,
            prefix_length=args.train_prefix_length,
            max_length=args.train_max_length,
        )
        vector_records[str(cid)]["max_act"] = max_act_rec

        model.config.use_cache = True
        log(f"concept={cid} generating base/prompt baselines")
        base_calib = generate_texts(
            model,
            tokenizer,
            calib_prompts,
            add_vec=None,
            layer=args.layer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        base_eval = generate_texts(
            model,
            tokenizer,
            eval_prompts,
            add_vec=None,
            layer=args.layer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        base_calib_gate = score_generation_gate(base_calib)
        base_eval_gate = score_generation_gate(base_eval)
        prompt_calib = generate_texts(
            model,
            tokenizer,
            prompt_baseline_prompts(concept_text, calib_prompts),
            add_vec=None,
            layer=args.layer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        prompt_eval = generate_texts(
            model,
            tokenizer,
            prompt_baseline_prompts(concept_text, eval_prompts),
            add_vec=None,
            layer=args.layer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

        sweep = []
        for factor in factors:
            add_vec = float(factor) * float(max_act_rec["max_act"]) * unit_vec.float()
            log(f"concept={cid} factor={factor:g} generating DiffMean")
            calib_texts = generate_texts(
                model,
                tokenizer,
                calib_prompts,
                add_vec=add_vec,
                layer=args.layer,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            eval_texts = generate_texts(
                model,
                tokenizer,
                eval_prompts,
                add_vec=add_vec,
                layer=args.layer,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            sweep.append({
                "factor": float(factor),
                "calib_texts": calib_texts,
                "eval_texts": eval_texts,
                "calib_gate": score_generation_gate(calib_texts, base=base_calib_gate),
                "eval_gate": score_generation_gate(eval_texts, base=base_eval_gate),
            })

        result["concept_runs"].append({
            "concept": concept,
            "diffmean_vector": vector_records[str(cid)],
            "max_act": float(max_act_rec["max_act"]),
            "rows": rows,
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
        })

    write_json(outdir / "semantic_generations.json", result)
    write_json(outdir / "dry_run_manifest.json", {
        "generation_sha256": sha256_obj(result),
        "concept_ids": concept_ids,
        "factors": factors,
        "method": "AxBench DiffMean",
        "vector_records": vector_records,
        "n_texts": sum(
            len(c["base"]["calib_texts"]) + len(c["base"]["eval_texts"])
            + len(c["prompt_baseline"]["calib_texts"]) + len(c["prompt_baseline"]["eval_texts"])
            + sum(len(s["calib_texts"]) + len(s["eval_texts"]) for s in c["sweep"])
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
    train_counts = {}
    neg_count = int(((train_df["concept_id"] == -1) & (train_df["category"] == "negative")).sum())
    for cid in concept_ids:
        train_counts[str(cid)] = int(
            ((train_df["concept_id"] == cid) & (train_df["category"] == "positive")).sum()
        )
    neg_counts_by_genre = (
        train_df[(train_df["concept_id"] == -1) & (train_df["category"] == "negative")]
        .groupby("concept_genre")
        .size()
        .to_dict()
    )
    pos_genres = {
        str(cid): sorted({
            str(x)
            for x in train_df[
                (train_df["concept_id"] == cid)
                & (train_df["category"] == "positive")
            ]["concept_genre"].dropna().tolist()
        })
        for cid in concept_ids
    }
    manifest = {
        "mode": "dry",
        "method": "AxBench DiffMean",
        "model": args.model,
        "train_dir": str(train_dir),
        "data_dir": str(data_dir),
        "train_data_sha256": sha256_file(train_dir / "train_data.parquet"),
        "latent_eval_data_sha256": sha256_file(data_dir / "latent_eval_data.parquet"),
        "source_config_sha256": {
            str(p): sha256_file(p) for p in DEFAULT_CONFIG_FILES if p.exists()
        },
        "concepts": concepts,
        "factors": parse_floats(args.factors),
        "official_diffmean_grid": [
            0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4,
            1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0,
        ],
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "n_total_per_concept": args.n_calib + args.n_eval,
        "available_prompt_counts": prompt_count_by_concept(data_dir, concept_ids),
        "train_positive_counts": train_counts,
        "train_negative_pool_count": neg_count,
        "train_negative_counts_by_genre": {
            str(k): int(v) for k, v in neg_counts_by_genre.items()
        },
        "train_positive_genres": pos_genres,
        "train_prefix_length": args.train_prefix_length,
        "train_max_length": args.train_max_length,
        "train_is_chat_model": bool(args.train_is_chat_model),
        "model_name_for_chat": args.model_name_for_chat,
        "max_act_construction": "calibration-only AxBench-prepared input+output latent max",
        "planned_generation_texts": (
            len(concept_ids)
            * (2 + len(parse_floats(args.factors)))
            * (args.n_calib + args.n_eval)
        ),
        "planned_judge_prompts": (
            len(concept_ids)
            * (2 + len(parse_floats(args.factors)))
            * (args.n_calib + args.n_eval)
            * 3
        ),
    }
    write_json(outdir / "semantic_dry_run.json", manifest)
    log(f"wrote {outdir / 'semantic_dry_run.json'}")


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
    ap.add_argument("--factors", default="0.6,1.0,1.4,1.8")
    ap.add_argument("--n-calib", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-batch-size", type=int, default=6)
    ap.add_argument("--train-prefix-length", type=int, default=1)
    ap.add_argument("--train-max-length", type=int, default=1024)
    ap.add_argument("--train-n-pos", type=int, default=None)
    ap.add_argument("--model-name-for-chat", default="google/gemma-2-2b-it")
    ap.add_argument("--train-is-chat-model", action=argparse.BooleanOptionalAction, default=True)
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
