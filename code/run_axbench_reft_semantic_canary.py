"""Semantic AxBench/ReFT-r1 canary using AxBench-style LMJudge scoring.

This is the post-wedding AxBench branch. It deliberately avoids a hand-built
keyword lexicon: GPU generation is run first without any API credentials, then
the saved generations are judged locally with the AxBench LMJudge prompt
templates. The output is still a reproduce-first canary, not a rho/kappa row.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_REPO = _HERE.parents[2]
_AXBENCH = _REPO / "data/external/axbench"
_PROMPT_TEMPLATES = (
    _AXBENCH / "axbench/evaluators/prompt_templates.py")
_SPEC = importlib.util.spec_from_file_location("axbench_prompt_templates", _PROMPT_TEMPLATES)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"could not load AxBench prompt templates from {_PROMPT_TEMPLATES}")
_TPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TPL)
UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE = (
    _TPL.UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE)
UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE = (
    _TPL.UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE)
UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE = (
    _TPL.UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE)
from run_axbench_reft_canary import (  # noqa: E402
    DEFAULT_REFT_DIR,
    concept_record,
    generation_gate,
    layer_add_hook,
    load_reft_artifact,
    render_chat,
)


DEFAULT_MODEL_DIR = _REPO / "data/external/hf_models/gemma-2-2b-it"
DEFAULT_DATA_DIR = (
    _REPO
    / "data/external/axbench/axbench/concept500/prod_2b_l20_v1/inference"
)
DEFAULT_OUTDIR = (
    _REPO
    / "runs/steering-content-audit/2026-07-09-axbench-reft-semantic-canary"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ngram_rep_rate(text: str, n: int = 3) -> float:
    toks = re.findall(r"\S+", text.lower())
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams)) if grams else 0.0


def sample_prompts(data_dir: Path, concept_id: int, *, n_total: int, seed: int) -> list[dict]:
    df = pd.read_parquet(data_dir / "latent_eval_data.parquet")
    rows = df[df["concept_id"] == concept_id].copy().reset_index(drop=True)
    if len(rows) < n_total:
        raise SystemExit(
            f"concept_id={concept_id} has only {len(rows)} eval prompts, need {n_total}")
    rng = np.random.default_rng(seed + 9973 * concept_id)
    idx = rng.permutation(len(rows))[:n_total]
    out = []
    for local_id, row_idx in enumerate(idx.tolist()):
        row = rows.iloc[row_idx]
        out.append({
            "local_id": local_id,
            "source_row": int(row_idx),
            "prompt": str(row["input"]),
            "output_concept": str(row["output_concept"]),
            "concept_genre": str(row.get("concept_genre", "")),
            "sae_link": str(row.get("sae_link", "")),
            "sae_id": int(row.get("sae_id", -1)),
        })
    return out


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: Sequence[str], *,
                   add_vec: torch.Tensor | None,
                   layer: int,
                   batch_size: int,
                   max_new_tokens: int) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    out: list[str] = []
    rendered = [render_chat(tokenizer, p) for p in prompts]
    for i in range(0, len(rendered), batch_size):
        enc = tokenizer(rendered[i:i + batch_size], return_tensors="pt", padding=True).to(model.device)
        with layer_add_hook(model, layer, add_vec):
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


def score_generation_gate(texts: Sequence[str], base: dict | None = None) -> dict:
    return generation_gate(texts, base=base)


def prompt_baseline_prompts(concept: str, prompts: Sequence[str]) -> list[str]:
    prefix = (
        "While answering, naturally incorporate this concept when it fits: "
        f"{concept}\n\n"
    )
    return [prefix + p for p in prompts]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_generate(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artifact = load_reft_artifact(Path(args.reft_dir))
    weights: torch.Tensor = artifact["weights"]
    acts = artifact["acts"]
    metadata = artifact["metadata"]
    concept_ids = parse_ints(args.concept_ids)
    factors = parse_floats(args.factors)
    n_total = args.n_calib + args.n_eval
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32

    concepts = [concept_record(metadata, cid, weights.shape[0]) for cid in concept_ids]
    log(f"loading model {args.model} device={args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    result: dict[str, Any] = {
        "meta": {
            "mode": "generate",
            "model": args.model,
            "artifact_repo": "pyvene/gemma-reft-r1-2b-it-res",
            "artifact_dir": str(args.reft_dir),
            "artifact_sha256": artifact["sha256"],
            "data_dir": str(args.data_dir),
            "device": str(model.device),
            "dtype": args.dtype,
            "layer": args.layer,
            "concept_ids": concept_ids,
            "concepts": concepts,
            "factors": factors,
            "n_calib": args.n_calib,
            "n_eval": args.n_eval,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "scoring": "AxBench LMJudge semantic concept/instruction/fluency; judged in a separate local pass",
        },
        "concept_runs": [],
    }

    thresholds = acts.get("thresholds", {})
    max_acts = acts.get("max_act", {})
    for concept in concepts:
        cid = int(concept["concept_id"])
        concept_text = str(concept["concept"])
        max_act = float(max_acts.get(str(cid), thresholds.get(str(cid), 1.0)))
        add_base = weights[cid].float()
        rows = sample_prompts(Path(args.data_dir), cid, n_total=n_total, seed=args.seed)
        calib_rows = rows[:args.n_calib]
        eval_rows = rows[args.n_calib:]
        calib_prompts = [r["prompt"] for r in calib_rows]
        eval_prompts = [r["prompt"] for r in eval_rows]

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
        for factor in factors:
            add_vec = float(factor) * max_act * add_base
            log(f"concept={cid} factor={factor:g} generating ReFT")
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

        result["concept_runs"].append({
            "concept": concept,
            "max_act": max_act,
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
        "n_texts": sum(
            len(c["base"]["calib_texts"]) + len(c["base"]["eval_texts"])
            + len(c["prompt_baseline"]["calib_texts"]) + len(c["prompt_baseline"]["eval_texts"])
            + sum(len(s["calib_texts"]) + len(s["eval_texts"]) for s in c["sweep"])
            for c in result["concept_runs"]
        ),
    })
    log(f"wrote {outdir / 'semantic_generations.json'}")


def parse_rating(completion: str) -> float:
    if "Rating:" not in completion:
        return 0.0
    text = completion.split("Rating:")[-1].strip().split("\n")[0].strip()
    text = text.replace("[", "").replace("]", "").rstrip(".").strip("\"' *")
    try:
        val = float(text)
    except ValueError:
        return 0.0
    return float(min(2.0, max(0.0, val)))


def harmonic(scores: Sequence[float]) -> float:
    vals = [float(x) for x in scores]
    if any(x <= 0 for x in vals):
        return 0.0
    return len(vals) / sum(1.0 / x for x in vals)


@dataclass(frozen=True)
class JudgeItem:
    key: str
    concept: str
    instruction: str
    text: str


def judge_prompts(item: JudgeItem) -> dict[str, str]:
    return {
        "concept": UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
            concept=item.concept, sentence=item.text),
        "instruction": UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE.format(
            instruction=item.instruction, sentence=item.text),
        "fluency": UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
            sentence=item.text),
    }


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(path)


async def judge_missing(items: Sequence[JudgeItem], *, model: str, cache_path: Path,
                        batch_size: int) -> dict[str, dict]:
    from openai import AsyncOpenAI

    cache = load_cache(cache_path)
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=60.0, max_retries=3)
    pending: list[tuple[str, str]] = []
    key_to_meta: dict[str, tuple[str, str]] = {}
    for item in items:
        for kind, prompt in judge_prompts(item).items():
            cache_key = hashlib.sha256(
                json.dumps(
                    {"model": model, "kind": kind, "prompt": prompt},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if cache_key not in cache:
                pending.append((cache_key, prompt))
                key_to_meta[cache_key] = (item.key, kind)

    log(f"judge cache hits={3 * len(items) - len(pending)} missing={len(pending)} model={model}")
    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]
        tasks = [
            client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            for _, prompt in batch
        ]
        responses = await asyncio.gather(*tasks)
        for (cache_key, prompt), response in zip(batch, responses):
            raw = response.to_dict()
            completion = raw["choices"][0]["message"]["content"].strip()
            item_key, kind = key_to_meta[cache_key]
            cache[cache_key] = {
                "item_key": item_key,
                "kind": kind,
                "completion": completion,
                "rating": parse_rating(completion),
                "usage": raw.get("usage", {}),
            }
        save_cache(cache_path, cache)
        log(f"judged {min(i + batch_size, len(pending))}/{len(pending)} missing prompts")
    await client.close()
    return cache


def collect_items(gens: dict) -> list[JudgeItem]:
    items: list[JudgeItem] = []
    method_slug = str(gens.get("meta", {}).get("method_slug", "reft"))
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
                factor = sr["factor"]
                for j, (prompt, text) in enumerate(zip(prompts, sr[f"{split}_texts"])):
                    items.append(JudgeItem(
                        key=f"cid{cid}:{split}:{method_slug}_f{factor:g}:i{j}",
                        concept=concept,
                        instruction=prompt,
                        text=text,
                    ))
    return items


def ratings_for_items(items: Sequence[JudgeItem], cache: dict[str, dict], *, model: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in items:
        ratings = {}
        completions = {}
        for kind, prompt in judge_prompts(item).items():
            cache_key = hashlib.sha256(
                json.dumps(
                    {"model": model, "kind": kind, "prompt": prompt},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            rec = cache[cache_key]
            ratings[kind] = float(rec["rating"])
            completions[kind] = rec["completion"]
        agg = harmonic([ratings["concept"], ratings["instruction"], ratings["fluency"]])
        out[item.key] = {
            "ratings": ratings,
            "aggregate": agg,
            "aggregate_norm": agg / 2.0,
            "pass_1": bool(agg >= 1.0),
            "completions": completions,
        }
    return out


def mean_metric(judged: dict[str, dict], keys: Sequence[str], field: str = "aggregate_norm") -> float:
    vals = [float(judged[k][field]) for k in keys]
    return float(np.mean(vals)) if vals else float("nan")


def component_means(judged: dict[str, dict], keys: Sequence[str]) -> dict[str, float]:
    out = {}
    for kind in ("concept", "instruction", "fluency"):
        vals = [float(judged[k]["ratings"][kind]) / 2.0 for k in keys]
        out[f"{kind}_mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


def bootstrap_ratio(base_vals: Sequence[float], method_vals: Sequence[float],
                    control_vals: Sequence[float] | None = None, *,
                    n_boot: int, seed: int) -> dict:
    base = np.asarray(base_vals, dtype=np.float64)
    method = np.asarray(method_vals, dtype=np.float64)
    ctrl = np.asarray(control_vals, dtype=np.float64) if control_vals is not None else None
    effect_m = float(np.mean(method - base))
    out = {"effect_method": effect_m, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    if ctrl is not None:
        effect_c = float(np.mean(ctrl - base))
        rho = effect_c / effect_m if abs(effect_m) > 1e-12 else float("nan")
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(base), size=len(base))
            em = float(np.mean(method[idx] - base[idx]))
            ec = float(np.mean(ctrl[idx] - base[idx]))
            vals.append(ec / em if abs(em) > 1e-12 else np.nan)
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        out.update({
            "effect_control": effect_c,
            "rho": rho,
            "rho_lo": float(np.quantile(arr, 0.025)) if len(arr) else float("nan"),
            "rho_hi": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
            "finite_boot": int(len(arr)),
        })
    else:
        vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(base), size=len(base))
            vals.append(float(np.mean(method[idx] - base[idx])))
        arr = np.asarray(vals, dtype=np.float64)
        out.update({
            "effect_lo": float(np.quantile(arr, 0.025)),
            "effect_hi": float(np.quantile(arr, 0.975)),
        })
    return out


def keys_for(cid: int, split: str, cond: str, n: int) -> list[str]:
    return [f"cid{cid}:{split}:{cond}:i{i}" for i in range(n)]


def summarize_judged(gens: dict, judged: dict[str, dict], *, n_boot: int, seed: int,
                     reproduce_threshold: float) -> dict:
    n_calib = int(gens["meta"]["n_calib"])
    n_eval = int(gens["meta"]["n_eval"])
    method_slug = str(gens.get("meta", {}).get("method_slug", "reft"))
    calib_rows = []
    eval_rows = []
    for cr in gens["concept_runs"]:
        cid = int(cr["concept"]["concept_id"])
        concept = str(cr["concept"]["concept"])
        base_calib_keys = keys_for(cid, "calib", "base", n_calib)
        base_eval_keys = keys_for(cid, "eval", "base", n_eval)
        base_calib_vals = [judged[k]["aggregate_norm"] for k in base_calib_keys]
        base_eval_vals = [judged[k]["aggregate_norm"] for k in base_eval_keys]
        prompt_calib_keys = keys_for(cid, "calib", "prompt", n_calib)
        prompt_eval_keys = keys_for(cid, "eval", "prompt", n_eval)

        for sr in cr["sweep"]:
            factor = float(sr["factor"])
            cond = f"{method_slug}_f{factor:g}"
            reft_calib_keys = keys_for(cid, "calib", cond, n_calib)
            reft_eval_keys = keys_for(cid, "eval", cond, n_eval)
            reft_calib_vals = [judged[k]["aggregate_norm"] for k in reft_calib_keys]
            reft_eval_vals = [judged[k]["aggregate_norm"] for k in reft_eval_keys]
            calib_stats = bootstrap_ratio(
                base_calib_vals, reft_calib_vals, n_boot=n_boot, seed=seed + cid + int(factor * 100))
            eval_stats = bootstrap_ratio(
                base_eval_vals, reft_eval_vals, n_boot=n_boot, seed=seed + 100000 + cid + int(factor * 100))
            base_calib_components = component_means(judged, base_calib_keys)
            reft_calib_components = component_means(judged, reft_calib_keys)
            base_eval_components = component_means(judged, base_eval_keys)
            reft_eval_components = component_means(judged, reft_eval_keys)
            row_common = {
                "concept_id": cid,
                "concept": concept,
                "factor": factor,
                "base_mean": float(np.mean(base_calib_vals)),
                "reft_mean": float(np.mean(reft_calib_vals)),
                "base_components": base_calib_components,
                "reft_components": reft_calib_components,
                "instruction_drop": (
                    base_calib_components["instruction_mean"]
                    - reft_calib_components["instruction_mean"]
                ),
                "fluency_drop": (
                    base_calib_components["fluency_mean"]
                    - reft_calib_components["fluency_mean"]
                ),
                "effect": calib_stats["effect_method"],
                "effect_lo": calib_stats["effect_lo"],
                "effect_hi": calib_stats["effect_hi"],
                "reft_gate": sr["calib_gate"],
            }
            calib_rows.append(row_common)
            eval_rows.append({
                "concept_id": cid,
                "concept": concept,
                "factor": factor,
                "base_mean": float(np.mean(base_eval_vals)),
                "reft_mean": float(np.mean(reft_eval_vals)),
                "base_components": base_eval_components,
                "reft_components": reft_eval_components,
                "instruction_drop": (
                    base_eval_components["instruction_mean"]
                    - reft_eval_components["instruction_mean"]
                ),
                "fluency_drop": (
                    base_eval_components["fluency_mean"]
                    - reft_eval_components["fluency_mean"]
                ),
                "effect": eval_stats["effect_method"],
                "effect_lo": eval_stats["effect_lo"],
                "effect_hi": eval_stats["effect_hi"],
                "reft_gate": sr["eval_gate"],
            })

        prompt_calib_vals = [judged[k]["aggregate_norm"] for k in prompt_calib_keys]
        prompt_eval_vals = [judged[k]["aggregate_norm"] for k in prompt_eval_keys]
        cr["prompt_baseline"]["calib_rho_like"] = bootstrap_ratio(
            base_calib_vals,
            [judged[k]["aggregate_norm"] for k in keys_for(cid, "calib", "reft_f0", 0)],
            prompt_calib_vals,
            n_boot=0,
            seed=seed,
        ) if False else {
            "base_mean": float(np.mean(base_calib_vals)),
            "prompt_mean": float(np.mean(prompt_calib_vals)),
            "effect_prompt": float(np.mean(np.asarray(prompt_calib_vals) - np.asarray(base_calib_vals))),
        }
        cr["prompt_baseline"]["eval_summary"] = {
            "base_mean": float(np.mean(base_eval_vals)),
            "prompt_mean": float(np.mean(prompt_eval_vals)),
            "effect_prompt": float(np.mean(np.asarray(prompt_eval_vals) - np.asarray(base_eval_vals))),
        }

    clean_calib = [r for r in calib_rows if not r["reft_gate"]["degenerate"]]
    if clean_calib:
        chosen = max(clean_calib, key=lambda r: (r["effect"], r["reft_mean"], -r["factor"]))
    else:
        chosen = max(calib_rows, key=lambda r: (r["effect"], r["reft_mean"], -r["factor"]))
    chosen_eval = next(
        r for r in eval_rows
        if r["concept_id"] == chosen["concept_id"] and abs(r["factor"] - chosen["factor"]) < 1e-9
    )
    reproduced = (
        bool(not chosen_eval["reft_gate"]["degenerate"])
        and float(chosen_eval["effect"]) >= reproduce_threshold
        and float(chosen_eval["effect_lo"]) > 0.0
        and float(chosen_eval["instruction_drop"]) <= 0.20
        and float(chosen_eval["fluency_drop"]) <= 0.20
    )
    return {
        "verdict": {
            "class": "SEMANTIC-CANARY-REPRODUCED" if reproduced else "SEMANTIC-CANARY-NOT-REPRODUCED",
            "reproduced": reproduced,
            "threshold_norm_aggregate_gain": reproduce_threshold,
            "max_instruction_drop": 0.20,
            "max_fluency_drop": 0.20,
            "note": "Reproduce-first semantic canary only; matched output-control and kappa are required before any master-table row.",
        },
        "chosen": chosen,
        "chosen_eval": chosen_eval,
        "calib_rows": sorted(calib_rows, key=lambda r: (r["effect"], r["reft_mean"]), reverse=True),
        "eval_rows": sorted(eval_rows, key=lambda r: (r["effect"], r["reft_mean"]), reverse=True),
    }


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
    log(f"verdict={summary['verdict']['class']} chosen={summary['chosen']['concept_id']} factor={summary['chosen']['factor']}")


def write_report(outdir: Path, result: dict) -> None:
    summary = result["summary"]
    chosen = summary["chosen"]
    chosen_eval = summary["chosen_eval"]
    method_label = str(result["meta"].get("method_label", "ReFT"))
    lines = [
        f"# AxBench/{method_label} semantic canary",
        "",
        "This is a reproduce-first canary scored with AxBench-style LMJudge prompts, not a rho/kappa verdict.",
        "",
        f"- Judge model: `{result['meta']['judge_model']}`",
        f"- Candidate concepts: `{result['meta']['concept_ids']}`",
        f"- Factors: `{result['meta']['factors']}`",
        f"- Chosen concept: {chosen['concept_id']} - {chosen['concept']}",
        f"- Chosen factor: {chosen['factor']}",
        f"- Calibration normalized aggregate gain: {chosen['effect']:+.3f} [{chosen['effect_lo']:+.3f}, {chosen['effect_hi']:+.3f}]",
        f"- Eval normalized aggregate gain: {chosen_eval['effect']:+.3f} [{chosen_eval['effect_lo']:+.3f}, {chosen_eval['effect_hi']:+.3f}]",
        f"- Eval gate clean: `{not chosen_eval['reft_gate']['degenerate']}`",
        f"- Eval instruction drop: {chosen_eval['instruction_drop']:+.3f}",
        f"- Eval fluency drop: {chosen_eval['fluency_drop']:+.3f}",
        f"- Verdict: `{summary['verdict']['class']}`",
        "",
        "Top calibration rows:",
        "",
        f"| concept | factor | base | {method_label} | gain | instr drop | fluency drop | clean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["calib_rows"][:12]:
        lines.append(
            f"| {row['concept_id']} | {row['factor']:.2f} | {row['base_mean']:.3f} | "
            f"{row['reft_mean']:.3f} | {row['effect']:+.3f} | "
            f"{row['instruction_drop']:+.3f} | {row['fluency_drop']:+.3f} | "
            f"{not row['reft_gate']['degenerate']} |"
        )
    lines.append("")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_dry(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifact = load_reft_artifact(Path(args.reft_dir))
    metadata = artifact["metadata"]
    weights: torch.Tensor = artifact["weights"]
    concept_ids = parse_ints(args.concept_ids)
    n_total = args.n_calib + args.n_eval
    concepts = [concept_record(metadata, cid, weights.shape[0]) for cid in concept_ids]
    prompt_counts = {}
    for cid in concept_ids:
        rows = pd.read_parquet(Path(args.data_dir) / "latent_eval_data.parquet")
        prompt_counts[str(cid)] = int((rows["concept_id"] == cid).sum())
    manifest = {
        "mode": "dry",
        "model": args.model,
        "reft_dir": str(args.reft_dir),
        "artifact_sha256": artifact["sha256"],
        "data_dir": str(args.data_dir),
        "concepts": concepts,
        "factors": parse_floats(args.factors),
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "n_total_per_concept": n_total,
        "available_prompt_counts": prompt_counts,
        "planned_generation_texts": len(concept_ids) * (2 + len(parse_floats(args.factors))) * n_total,
        "planned_judge_prompts": len(concept_ids) * (2 + len(parse_floats(args.factors))) * n_total * 3,
    }
    write_json(outdir / "semantic_dry_run.json", manifest)
    log(f"wrote {outdir / 'semantic_dry_run.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "generate", "judge"], default="dry")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--reft-dir", default=str(DEFAULT_REFT_DIR))
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
