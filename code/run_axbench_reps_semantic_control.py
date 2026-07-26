"""Matched semantic output-control follow-up for the AxBench/RePS canary.

This is the conditional Amendment 18/18a control for reproduced RePS semantic
concepts only. It first recovers the exact rank-1 PreferenceVector tensors from
the frozen training recipe and shipped JSON hashes, then matches each RePS cell's
teacher-forced absolute per-step KL with sparse output-interface logit biases.

GPU generation is secret-free. Judging runs separately with the local OpenAI key.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
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

from run_axbench_reft_control import (  # noqa: E402
    base_continuations,
    base_probs_on_s,
    calibrate_scalar,
    sparse_bias_texts,
    sparse_kl_from_probs,
    teacher_forced_stats,
    token_string,
)
from run_axbench_reft_semantic_canary import (  # noqa: E402
    JudgeItem,
    bootstrap_ratio,
    component_means,
    generate_texts,
    judge_missing,
    ratings_for_items,
    score_generation_gate,
    sha256_obj,
    write_json,
)
from run_axbench_reps_semantic_canary import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_TRAIN_DIR,
    generate_base_responses,
    train_reps_vector,
)


DEFAULT_SOURCE_OUTDIR = (
    _REPO / "runs/steering-content-audit/2026-07-09-axbench-reps-semantic-canary"
)
DEFAULT_OUTDIR = (
    _REPO / "runs/steering-content-audit/2026-07-10-axbench-reps-semantic-control"
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_controller(name: str) -> tuple[str, int]:
    m = re.fullmatch(r"(pos|signed|bal|baleq)(\d+)", name.strip())
    if not m:
        raise SystemExit(
            f"unsupported controller {name!r}; use e.g. pos100,signed500,bal250,baleq250")
    return m.group(1), int(m.group(2))


def scale_label(scale: float) -> str:
    return f"{float(scale):g}"


def cond_for_control(controller: str, scale: float) -> str:
    return f"ctrl_{controller}_s{scale_label(scale)}"


def keys_for(cid: int, cond: str, n: int) -> list[str]:
    return [f"cid{cid}:eval:{cond}:i{i}" for i in range(n)]


def vector_sha256(vector: torch.Tensor) -> str:
    arr = vector.detach().cpu().float().numpy().astype("float32")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def nonblank_token(tokenizer, token_id: int) -> bool:
    text = token_string(tokenizer, token_id)
    return bool(text) and not text.isspace()


def normalize_weights(raw: Sequence[float]) -> tuple[list[float], float]:
    rms = math.sqrt(sum(float(w) * float(w) for w in raw) / max(len(raw), 1))
    return [float(w) / max(rms, 1e-12) for w in raw], rms


def choose_controller_tokens(tokenizer, mean_delta: torch.Tensor, controller: str) -> dict[str, Any]:
    mode, top_k = parse_controller(controller)
    vals = mean_delta.float()
    special = set(int(x) for x in (tokenizer.all_special_ids or []))
    ids: list[int] = []
    source_raw: list[float] | None = None

    if mode == "pos":
        for tid in torch.argsort(vals, descending=True).tolist():
            tid = int(tid)
            if tid in special or not nonblank_token(tokenizer, tid):
                continue
            if float(vals[tid]) <= 0:
                continue
            ids.append(tid)
            if len(ids) >= top_k:
                break
        raw = [max(float(vals[tid]), 0.0) for tid in ids]
    elif mode == "signed":
        for tid in torch.argsort(torch.abs(vals), descending=True).tolist():
            tid = int(tid)
            if tid in special or not nonblank_token(tokenizer, tid):
                continue
            if abs(float(vals[tid])) <= 0:
                continue
            ids.append(tid)
            if len(ids) >= top_k:
                break
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
        source_raw = [float(vals[tid]) for tid in ids]
        if mode == "baleq":
            pos_rms = math.sqrt(
                sum(w * w for w in source_raw if w > 0)
                / max(sum(1 for w in source_raw if w > 0), 1)
            )
            neg_rms = math.sqrt(
                sum(w * w for w in source_raw if w < 0)
                / max(sum(1 for w in source_raw if w < 0), 1)
            )
            raw = [
                (w / max(pos_rms, 1e-12)) if w > 0 else (w / max(neg_rms, 1e-12))
                for w in source_raw
            ]
        else:
            raw = list(source_raw)

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


def find_concept_run(gens: dict[str, Any], concept_id: int) -> dict[str, Any]:
    for cr in gens["concept_runs"]:
        if int(cr["concept"]["concept_id"]) == int(concept_id):
            return cr
    raise KeyError(f"missing concept_id={concept_id}")


def selected_reproduced(judged: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in judged["summary"]["per_concept"]:
        if bool(row.get("reproduced")):
            ev = row["selected_eval"]
            rows.append({
                "concept_id": int(ev["concept_id"]),
                "concept": str(ev["concept"]),
                "factor": float(ev["factor"]),
                "source_selected": row,
            })
    rows.sort(key=lambda x: x["concept_id"])
    return rows


def exact_train_rows(train_df: pd.DataFrame, cr: dict[str, Any]) -> pd.DataFrame:
    cid = int(cr["concept"]["concept_id"])
    wanted = [int(x) for x in cr["reps_vector"]["source_train_rows"]]
    df = train_df.reset_index(names="source_row")
    rows = df[df["source_row"].isin(wanted)].copy()
    rows = rows.sort_values("source_row").reset_index(drop=True)
    found = [int(x) for x in rows["source_row"].tolist()]
    if found != wanted:
        raise SystemExit(f"source row mismatch for concept={cid}: found={found} wanted={wanted}")
    bad = rows[(rows["concept_id"] != cid) | (rows["category"] != "positive")]
    if not bad.empty:
        raise SystemExit(f"non-positive or wrong-concept training rows for concept={cid}")
    return rows


def recover_reps_vector(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    cr: dict[str, Any],
    *,
    layer: int,
    train_epochs: int,
    train_batch_size: int,
    train_lr: float,
    train_max_length: int,
    train_factors: Sequence[float],
    train_gen_batch_size: int,
    dpo_max_new_tokens: int,
    simpo_scaler: float,
    seed: int,
    bias_tol: float,
    norm_tol: float,
) -> dict[str, Any]:
    cid = int(cr["concept"]["concept_id"])
    expected = cr["reps_vector"]
    train_rows = exact_train_rows(train_df, cr)
    prompts = [str(x) for x in train_rows["input"].tolist()]
    winning_outputs = [str(x) for x in train_rows["output"].tolist()]

    model.config.use_cache = False
    losing_outputs = generate_base_responses(
        model,
        tokenizer,
        prompts,
        batch_size=train_gen_batch_size,
        max_new_tokens=dpo_max_new_tokens,
    )
    losing_sha = hashlib.sha256(
        json.dumps(losing_outputs, sort_keys=True).encode("utf-8")
    ).hexdigest()
    examples = [
        {"input": p, "winning_output": w, "losing_output": l}
        for p, w, l in zip(prompts, winning_outputs, losing_outputs)
    ]
    rec = train_reps_vector(
        model,
        tokenizer,
        examples,
        layer=layer,
        epochs=train_epochs,
        batch_size=train_batch_size,
        lr=train_lr,
        max_length=train_max_length,
        factors=train_factors,
        seed=seed + cid,
        simpo_scaler=simpo_scaler,
    )
    initial: torch.Tensor = rec["initial_vector"].float()
    vector: torch.Tensor = rec["vector"].float()
    bias = float(rec["bias"])
    initial_sha = vector_sha256(initial)
    vector_sha = vector_sha256(vector)
    vector_norm = float(torch.linalg.vector_norm(vector).item())
    checks = {
        "concept_id": cid,
        "source_train_rows_match": True,
        "source_train_count": int(len(train_rows)),
        "losing_output_sha256": losing_sha,
        "expected_losing_output_sha256": expected["losing_output_sha256"],
        "losing_output_sha256_match": losing_sha == expected["losing_output_sha256"],
        "initial_vector_sha256": initial_sha,
        "expected_initial_vector_sha256": expected["initial_vector_sha256"],
        "initial_vector_sha256_match": initial_sha == expected["initial_vector_sha256"],
        "vector_sha256": vector_sha,
        "expected_vector_sha256": expected["vector_sha256"],
        "vector_sha256_match": vector_sha == expected["vector_sha256"],
        "vector_norm": vector_norm,
        "expected_vector_norm": float(expected["vector_norm"]),
        "vector_norm_abs_err": abs(vector_norm - float(expected["vector_norm"])),
        "vector_norm_match": abs(vector_norm - float(expected["vector_norm"])) <= norm_tol,
        "bias": bias,
        "expected_bias": float(expected["bias"]),
        "bias_abs_err": abs(bias - float(expected["bias"])),
        "bias_match": abs(bias - float(expected["bias"])) <= bias_tol,
        "train_metrics": rec["metrics"],
    }
    checks["all_match"] = bool(
        checks["losing_output_sha256_match"]
        and checks["initial_vector_sha256_match"]
        and checks["vector_sha256_match"]
        and checks["vector_norm_match"]
        and checks["bias_match"]
    )
    return {
        "vector": vector,
        "bias": bias,
        "checks": checks,
        "losing_outputs": losing_outputs,
    }


def load_persisted_reps_vector(
    vector_file: Path,
    cr: dict[str, Any],
    *,
    bias_tol: float,
    norm_tol: float,
) -> dict[str, Any]:
    cid = int(cr["concept"]["concept_id"])
    expected = cr["reps_vector"]
    with np.load(vector_file) as data:
        vector = torch.from_numpy(np.asarray(data[f"cid{cid}_vector"], dtype=np.float32).copy()).float()
        initial = torch.from_numpy(
            np.asarray(data[f"cid{cid}_initial_vector"], dtype=np.float32).copy()
        ).float()
        bias = float(np.asarray(data[f"cid{cid}_bias"], dtype=np.float32).reshape(-1)[0])
    initial_sha = vector_sha256(initial)
    vec_sha = vector_sha256(vector)
    vector_norm = float(torch.linalg.vector_norm(vector).item())
    checks = {
        "concept_id": cid,
        "source": "persisted_vector_sidecar",
        "vector_file": str(vector_file),
        "initial_vector_sha256": initial_sha,
        "expected_initial_vector_sha256": expected["initial_vector_sha256"],
        "initial_vector_sha256_match": initial_sha == expected["initial_vector_sha256"],
        "vector_sha256": vec_sha,
        "expected_vector_sha256": expected["vector_sha256"],
        "vector_sha256_match": vec_sha == expected["vector_sha256"],
        "vector_norm": vector_norm,
        "expected_vector_norm": float(expected["vector_norm"]),
        "vector_norm_abs_err": abs(vector_norm - float(expected["vector_norm"])),
        "vector_norm_match": abs(vector_norm - float(expected["vector_norm"])) <= norm_tol,
        "bias": bias,
        "expected_bias": float(expected["bias"]),
        "bias_abs_err": abs(bias - float(expected["bias"])),
        "bias_match": abs(bias - float(expected["bias"])) <= bias_tol,
    }
    checks["all_match"] = bool(
        checks["initial_vector_sha256_match"]
        and checks["vector_sha256_match"]
        and checks["vector_norm_match"]
        and checks["bias_match"]
    )
    return {
        "vector": vector,
        "bias": bias,
        "checks": checks,
        "losing_outputs": None,
    }


def collect_control_items(gens: dict[str, Any]) -> list[JudgeItem]:
    items: list[JudgeItem] = []
    n_eval = int(gens["meta"]["n_eval"])
    for cr in gens["concept_runs"]:
        cid = int(cr["concept"]["concept_id"])
        concept = str(cr["concept"]["concept"])
        prompts = [r["prompt"] for r in cr["rows"][-n_eval:]]
        factor = float(cr["selected_factor"])
        for cond, texts in (
            ("base", cr["base_eval_texts"]),
            (f"reps_f{factor:g}", cr["method_eval_texts"]),
        ):
            for j, (prompt, text) in enumerate(zip(prompts, texts)):
                items.append(JudgeItem(
                    key=f"cid{cid}:eval:{cond}:i{j}",
                    concept=concept,
                    instruction=prompt,
                    text=text,
                ))
        for controller in cr["controllers"]:
            name = str(controller["controller"])
            for dose in controller["doses"]:
                cond = cond_for_control(name, float(dose["scale"]))
                for j, (prompt, text) in enumerate(zip(prompts, dose["eval_texts"])):
                    items.append(JudgeItem(
                        key=f"cid{cid}:eval:{cond}:i{j}",
                        concept=concept,
                        instruction=prompt,
                        text=text,
                    ))
    return items


def preseed_cache(seed_cache: Path, cache_path: Path) -> dict[str, dict]:
    cache = load_json(cache_path) if cache_path.exists() else {}
    if seed_cache.exists():
        seed = load_json(seed_cache)
        added = 0
        for key, rec in seed.items():
            if key not in cache:
                cache[key] = rec
                added += 1
        if added:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        log(f"preseeded judge cache from {seed_cache}: added={added} total={len(cache)}")
    return cache


def clean_semantic_row(row: dict[str, Any], *, max_drop: float) -> bool:
    return bool(
        not row["gate"]["degenerate"]
        and float(row["instruction_drop"]) <= max_drop
        and float(row["fluency_drop"]) <= max_drop
    )


def summarize_control(
    gens: dict[str, Any],
    judged: dict[str, dict],
    *,
    n_boot: int,
    seed: int,
    max_component_drop: float,
    kl_rel_budget: float,
) -> dict[str, Any]:
    n_eval = int(gens["meta"]["n_eval"])
    concept_summaries: list[dict[str, Any]] = []
    aggregate_buckets: dict[tuple[str, float], dict[str, list[float]]] = {}

    for cr in gens["concept_runs"]:
        cid = int(cr["concept"]["concept_id"])
        factor = float(cr["selected_factor"])
        target_kl = float(cr["teacher_forced"]["target_kl"])
        base_keys = keys_for(cid, "base", n_eval)
        method_keys = keys_for(cid, f"reps_f{factor:g}", n_eval)
        base_vals = [float(judged[k]["aggregate_norm"]) for k in base_keys]
        method_vals = [float(judged[k]["aggregate_norm"]) for k in method_keys]
        method_stats = bootstrap_ratio(
            base_vals, method_vals, n_boot=n_boot, seed=seed + cid)
        base_comp = component_means(judged, base_keys)
        method_comp = component_means(judged, method_keys)
        method_row = {
            "condition": f"reps_f{factor:g}",
            "factor": factor,
            "base_mean": float(np.mean(base_vals)),
            "method_mean": float(np.mean(method_vals)),
            "effect": method_stats["effect_method"],
            "effect_lo": method_stats["effect_lo"],
            "effect_hi": method_stats["effect_hi"],
            "base_components": base_comp,
            "method_components": method_comp,
            "instruction_drop": base_comp["instruction_mean"] - method_comp["instruction_mean"],
            "fluency_drop": base_comp["fluency_mean"] - method_comp["fluency_mean"],
            "gate": cr["method_eval_gate"],
        }

        control_rows: list[dict[str, Any]] = []
        for ci, controller in enumerate(cr["controllers"]):
            name = str(controller["controller"])
            for dose in controller["doses"]:
                scale = float(dose["scale"])
                cond = cond_for_control(name, scale)
                ctrl_keys = keys_for(cid, cond, n_eval)
                ctrl_vals = [float(judged[k]["aggregate_norm"]) for k in ctrl_keys]
                rho_stats = bootstrap_ratio(
                    base_vals,
                    method_vals,
                    ctrl_vals,
                    n_boot=n_boot,
                    seed=seed + cid * 1000 + ci * 10000 + int(scale * 1000),
                )
                ctrl_comp = component_means(judged, ctrl_keys)
                row = {
                    "concept_id": cid,
                    "concept": cr["concept"]["concept"],
                    "controller": name,
                    "scale": scale,
                    "condition": cond,
                    "kl": float(dose["kl"]),
                    "target_kl": target_kl,
                    "kl_budget_ratio": float(dose["kl"]) / max(target_kl, 1e-12),
                    "within_budget": bool(float(dose["kl"]) <= target_kl * kl_rel_budget),
                    "base_mean": float(np.mean(base_vals)),
                    "method_mean": float(np.mean(method_vals)),
                    "control_mean": float(np.mean(ctrl_vals)),
                    "control_components": ctrl_comp,
                    "instruction_drop": base_comp["instruction_mean"] - ctrl_comp["instruction_mean"],
                    "fluency_drop": base_comp["fluency_mean"] - ctrl_comp["fluency_mean"],
                    "gate": dose["eval_gate"],
                    "rho": rho_stats,
                }
                row["clean"] = clean_semantic_row(row, max_drop=max_component_drop)
                control_rows.append(row)
                bucket = aggregate_buckets.setdefault((name, scale), {
                    "base": [],
                    "method": [],
                    "control": [],
                    "all_clean": [],
                    "all_within_budget": [],
                })
                bucket["base"].extend(base_vals)
                bucket["method"].extend(method_vals)
                bucket["control"].extend(ctrl_vals)
                bucket["all_clean"].append(bool(row["clean"]))
                bucket["all_within_budget"].append(bool(row["within_budget"]))

        clean_within = [r for r in control_rows if r["clean"] and r["within_budget"]]
        dissolved = [r for r in clean_within if float(r["rho"]["rho_lo"]) >= 0.9]
        if dissolved:
            best = max(dissolved, key=lambda r: (r["rho"]["rho_lo"], r["rho"]["rho"]))
            verdict = {
                "class": "CONCEPT-DISSOLVED-CANDIDATE",
                "reason": "at least one clean within-budget controller/dose has rho_lo >= 0.9",
                "selected": {
                    "controller": best["controller"],
                    "scale": best["scale"],
                    "rho": best["rho"],
                    "kl": best["kl"],
                },
            }
        elif clean_within and max(float(r["rho"]["rho_hi"]) for r in clean_within) <= 0.3:
            verdict = {
                "class": "CONCEPT-SURVIVAL-SIGNAL",
                "reason": "all clean within-budget controller/doses have rho_hi <= 0.3",
                "max_clean_rho_hi": max(float(r["rho"]["rho_hi"]) for r in clean_within),
            }
        elif not clean_within:
            verdict = {
                "class": "CONCEPT-OUTPUT-WALL-NO-CLEAN-WITHIN-BUDGET",
                "reason": "no controller/dose is both clean and within the matched KL budget",
            }
        else:
            verdict = {
                "class": "CONCEPT-MIXED-OR-INCONCLUSIVE",
                "reason": "clean within-budget frontier is between dissolved and survival criteria",
                "max_clean_rho_hi": max(float(r["rho"]["rho_hi"]) for r in clean_within),
                "max_clean_rho": max(float(r["rho"]["rho"]) for r in clean_within),
            }

        concept_summaries.append({
            "concept_id": cid,
            "concept": cr["concept"],
            "selected_factor": factor,
            "source_reps_selected_eval": cr["source_reps_selected_eval"],
            "method_eval": method_row,
            "teacher_forced": cr["teacher_forced"],
            "recovery_checks": cr["recovery_checks"],
            "control_rows": sorted(
                control_rows,
                key=lambda r: (
                    r["clean"],
                    r["within_budget"],
                    float(r["rho"]["rho"]) if math.isfinite(float(r["rho"]["rho"])) else -999,
                ),
                reverse=True,
            ),
            "verdict": verdict,
        })

    aggregate_rows: list[dict[str, Any]] = []
    for (controller, scale), vals in aggregate_buckets.items():
        stats = bootstrap_ratio(
            vals["base"],
            vals["method"],
            vals["control"],
            n_boot=n_boot,
            seed=seed + 777 + len(aggregate_rows),
        )
        aggregate_rows.append({
            "controller": controller,
            "scale": scale,
            "n": int(len(vals["base"])),
            "all_concepts_clean": bool(all(vals["all_clean"])),
            "all_concepts_within_budget": bool(all(vals["all_within_budget"])),
            "any_concept_within_budget": bool(any(vals["all_within_budget"])),
            "base_mean": float(np.mean(vals["base"])),
            "method_mean": float(np.mean(vals["method"])),
            "control_mean": float(np.mean(vals["control"])),
            "rho": stats,
        })
    aggregate_rows.sort(
        key=lambda r: (
            r["all_concepts_clean"],
            r["all_concepts_within_budget"],
            float(r["rho"]["rho"]) if math.isfinite(float(r["rho"]["rho"])) else -999,
        ),
        reverse=True,
    )

    dissolved_count = sum(
        1 for c in concept_summaries if c["verdict"]["class"] == "CONCEPT-DISSOLVED-CANDIDATE")
    survival_count = sum(
        1 for c in concept_summaries if c["verdict"]["class"] == "CONCEPT-SURVIVAL-SIGNAL")
    wall_count = sum(
        1 for c in concept_summaries if "OUTPUT-WALL" in c["verdict"]["class"])
    if dissolved_count > 0:
        panel_class = "REPS-SEMANTIC-CONTROL-DISSOLVED-CANDIDATE"
        reason = "at least one reproduced RePS concept has a clean within-budget rho_lo >= 0.9 control"
    elif survival_count == len(concept_summaries) and concept_summaries:
        panel_class = "REPS-SEMANTIC-CONTROL-SURVIVAL-SIGNAL"
        reason = "every reproduced RePS concept satisfies the survival criterion"
    else:
        panel_class = "REPS-SEMANTIC-CONTROL-MIXED-OR-WALL"
        reason = "controls neither dissolve nor cleanly fail across all reproduced concepts"

    return {
        "verdict": {
            "class": panel_class,
            "reason": reason,
            "concepts_controlled": int(len(concept_summaries)),
            "concepts_dissolved": int(dissolved_count),
            "concepts_survival_signal": int(survival_count),
            "concepts_output_wall": int(wall_count),
            "kl_rel_budget": kl_rel_budget,
            "max_component_drop": max_component_drop,
            "note": "If Mixed/output-wall, RePS is not countable in the master table.",
        },
        "concept_summaries": concept_summaries,
        "aggregate_controller_rows": aggregate_rows,
    }


def write_report(outdir: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# AxBench/RePS semantic output-control",
        "",
        "Matched-KL sparse output-interface controls on reproduced RePS semantic concepts.",
        "",
        f"- Judge model: `{result['meta']['judge_model']}`",
        f"- Source generation SHA256: `{result['meta']['source_generation_file_sha256']}`",
        f"- Source judged SHA256: `{result['meta']['source_judged_file_sha256']}`",
        f"- Controlled concepts: {summary['verdict']['concepts_controlled']}",
        f"- Verdict: `{summary['verdict']['class']}`",
        f"- Reason: {summary['verdict']['reason']}",
        "",
        "Per-concept verdicts:",
        "",
        "| concept | factor | target KL | method gain | best clean within-budget rho | verdict |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for cs in summary["concept_summaries"]:
        clean_within = [
            r for r in cs["control_rows"]
            if r["clean"] and r["within_budget"] and math.isfinite(float(r["rho"]["rho"]))
        ]
        best = max(clean_within, key=lambda r: r["rho"]["rho"], default=None)
        best_rho = f"{best['rho']['rho']:.3f} [{best['rho']['rho_lo']:.3f},{best['rho']['rho_hi']:.3f}]" if best else "NA"
        lines.append(
            f"| {cs['concept_id']} | {cs['selected_factor']:.2f} | "
            f"{cs['teacher_forced']['target_kl']:.6f} | "
            f"{cs['method_eval']['effect']:+.3f} | {best_rho} | "
            f"`{cs['verdict']['class']}` |"
        )
    lines.extend([
        "",
        "Top aggregate controller rows:",
        "",
        "| controller | scale | all clean | all within budget | rho | control gain |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["aggregate_controller_rows"][:12]:
        lines.append(
            f"| {row['controller']} | {row['scale']:.2f} | "
            f"{row['all_concepts_clean']} | {row['all_concepts_within_budget']} | "
            f"{row['rho']['rho']:.3f} [{row['rho']['rho_lo']:.3f},{row['rho']['rho_hi']:.3f}] | "
            f"{row['rho']['effect_control']:+.3f} |"
        )
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_dry(args: argparse.Namespace) -> None:
    source_outdir = Path(args.source_outdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gens_path = source_outdir / "semantic_generations.json"
    judged_path = source_outdir / "semantic_judged.json"
    gens = load_json(gens_path)
    judged = load_json(judged_path)
    selected = selected_reproduced(judged)
    controllers = [x.strip() for x in args.controllers.split(",") if x.strip()]
    scales = parse_floats(args.dose_scales)
    manifest = {
        "mode": "dry",
        "source_outdir": str(source_outdir),
        "source_generation_file_sha256": sha256_file(gens_path),
        "source_generation_canonical_sha256": sha256_obj(gens),
        "source_judged_file_sha256": sha256_file(judged_path),
        "source_judged_canonical_sha256": sha256_obj(judged),
        "selected_reproduced": selected,
        "controllers": controllers,
        "dose_scales": scales,
        "planned_control_texts": len(selected) * len(controllers) * len(scales) * int(gens["meta"]["n_eval"]),
        "planned_judge_items": len(selected) * (2 + len(controllers) * len(scales)) * int(gens["meta"]["n_eval"]),
        "planned_judge_prompts": len(selected) * (2 + len(controllers) * len(scales)) * int(gens["meta"]["n_eval"]) * 3,
        "requires_recovery_hash_match": True,
    }
    write_json(outdir / "dry_run_manifest.json", manifest)
    log(f"wrote {outdir / 'dry_run_manifest.json'}")


def run_generate(args: argparse.Namespace) -> None:
    source_outdir = Path(args.source_outdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gens_path = source_outdir / "semantic_generations.json"
    judged_path = source_outdir / "semantic_judged.json"
    gens = load_json(gens_path)
    judged = load_json(judged_path)
    selected = selected_reproduced(judged)
    if not selected:
        raise SystemExit("no reproduced RePS concepts found; output-control follow-up is not triggered")

    controllers = [x.strip() for x in args.controllers.split(",") if x.strip()]
    dose_scales = parse_floats(args.dose_scales)
    train_df = pd.read_parquet(Path(args.train_dir) / "train_data.parquet")
    train_factors = [float(x) for x in gens["meta"].get("train_factors", parse_floats(args.train_factors))]
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    vector_file = Path(args.vector_file) if args.vector_file else source_outdir / "reps_vectors.npz"
    use_persisted_vectors = vector_file.exists()

    log(f"loading model {args.model} device={args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    result: dict[str, Any] = {
        "meta": {
            "mode": "generate",
            "method": "AxBench RePS PreferenceVector semantic matched output-control",
            "model": args.model,
            "source_outdir": str(source_outdir),
            "source_generation_file_sha256": sha256_file(gens_path),
            "source_generation_canonical_sha256": sha256_obj(gens),
            "source_judged_file_sha256": sha256_file(judged_path),
            "source_judged_canonical_sha256": sha256_obj(judged),
            "train_dir": str(args.train_dir),
            "data_dir": str(args.data_dir),
            "device": str(model.device),
            "dtype": args.dtype,
            "layer": int(gens["meta"]["layer"]),
            "n_eval": int(gens["meta"]["n_eval"]),
            "n_calib": int(gens["meta"]["n_calib"]),
            "seed": int(gens["meta"]["seed"]),
            "max_new_tokens": int(gens["meta"]["max_new_tokens"]),
            "batch_size": args.batch_size,
            "controllers": controllers,
            "dose_scales": dose_scales,
            "vector_file": str(vector_file) if use_persisted_vectors else None,
            "vector_source": "persisted_sidecar" if use_persisted_vectors else "deterministic_retraining",
            "kl_rel_budget": args.kl_rel_budget,
            "max_component_drop": args.max_component_drop,
            "integrity_guard": (
                "recover exact RePS vectors from frozen training bytes before controls; "
                "no lexicon/concept-token injection in semantic controllers"
            ),
        },
        "source_reps_verdict": judged["summary"]["verdict"],
        "concept_runs": [],
    }

    layer = int(gens["meta"]["layer"])
    n_calib = int(gens["meta"]["n_calib"])
    n_eval = int(gens["meta"]["n_eval"])
    max_new_tokens = int(gens["meta"]["max_new_tokens"])
    seed = int(gens["meta"]["seed"])
    for sel in selected:
        cid = int(sel["concept_id"])
        factor = float(sel["factor"])
        cr = find_concept_run(gens, cid)
        sweep_row = next(s for s in cr["sweep"] if abs(float(s["factor"]) - factor) < 1e-9)
        rows = cr["rows"]
        calib_rows = rows[:n_calib]
        eval_rows = rows[n_calib:n_calib + n_eval]
        calib_prompts = [r["prompt"] for r in calib_rows]
        eval_prompts = [r["prompt"] for r in eval_rows]

        log(f"concept={cid} recovering RePS vector")
        if use_persisted_vectors:
            recovery = load_persisted_reps_vector(
                vector_file,
                cr,
                bias_tol=args.bias_tol,
                norm_tol=args.norm_tol,
            )
        else:
            recovery = recover_reps_vector(
                model,
                tokenizer,
                train_df,
                cr,
                layer=layer,
                train_epochs=int(gens["meta"]["train_epochs"]),
                train_batch_size=int(gens["meta"]["train_batch_size"]),
                train_lr=float(gens["meta"]["train_lr"]),
                train_max_length=int(gens["meta"]["train_max_length"]),
                train_factors=train_factors,
                train_gen_batch_size=args.train_gen_batch_size,
                dpo_max_new_tokens=args.dpo_max_new_tokens,
                simpo_scaler=args.simpo_scaler,
                seed=seed,
                bias_tol=args.bias_tol,
                norm_tol=args.norm_tol,
            )
        if not recovery["checks"]["all_match"]:
            write_json(outdir / "recovery_failed.json", {
                "concept_id": cid,
                "checks": recovery["checks"],
                "source_generation_file_sha256": sha256_file(gens_path),
            })
            raise SystemExit(
                f"RePS vector recovery failed for concept={cid}; wrote recovery_failed.json")

        vector: torch.Tensor = recovery["vector"].float()
        bias = float(recovery["bias"])
        add_vec = (factor + bias) * vector
        add_vec_sha = vector_sha256(add_vec)

        smoke_n = min(int(args.smoke_per_split), len(calib_prompts), len(eval_prompts))
        model.config.use_cache = True
        smoke_calib = generate_texts(
            model,
            tokenizer,
            calib_prompts[:smoke_n],
            add_vec=add_vec,
            layer=layer,
            batch_size=args.batch_size,
            max_new_tokens=max_new_tokens,
        )
        smoke_eval = generate_texts(
            model,
            tokenizer,
            eval_prompts[:smoke_n],
            add_vec=add_vec,
            layer=layer,
            batch_size=args.batch_size,
            max_new_tokens=max_new_tokens,
        )
        smoke_expected_calib = sweep_row["calib_texts"][:smoke_n]
        smoke_expected_eval = sweep_row["eval_texts"][:smoke_n]
        smoke_match = (
            smoke_calib == smoke_expected_calib
            and smoke_eval == smoke_expected_eval
        )
        smoke = {
            "smoke_per_split": smoke_n,
            "add_vec_sha256": add_vec_sha,
            "calib_match_count": int(sum(a == b for a, b in zip(smoke_calib, smoke_expected_calib))),
            "eval_match_count": int(sum(a == b for a, b in zip(smoke_eval, smoke_expected_eval))),
            "total_match_count": int(
                sum(a == b for a, b in zip(smoke_calib, smoke_expected_calib))
                + sum(a == b for a, b in zip(smoke_eval, smoke_expected_eval))
            ),
            "total": int(2 * smoke_n),
            "all_match": bool(smoke_match),
        }
        if not smoke_match:
            write_json(outdir / "recovery_failed.json", {
                "concept_id": cid,
                "checks": recovery["checks"],
                "smoke": smoke,
                "source_generation_file_sha256": sha256_file(gens_path),
            })
            raise SystemExit(
                f"RePS recovered-vector generation smoke failed for concept={cid}; "
                "wrote recovery_failed.json")

        log(f"concept={cid} generating calibration base continuations")
        calib_cont = base_continuations(
            model, tokenizer, calib_prompts, max_new_tokens=max_new_tokens)
        decoded_calib = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in calib_cont
        ]
        base_match = sum(
            1 for a, b in zip(decoded_calib, cr["base"]["calib_texts"])
            if str(a) == str(b)
        )

        log(f"concept={cid} computing RePS teacher-forced KL")
        tf = teacher_forced_stats(
            model, tokenizer, calib_prompts, calib_cont, add_vec=add_vec, layer=layer)
        log(f"concept={cid} factor={factor:g} target_kl={tf['target_kl']:.6f}")

        controller_results = []
        for ci, controller in enumerate(controllers):
            log(f"concept={cid} controller={controller} selecting semantic tokens")
            token_set = choose_controller_tokens(tokenizer, tf["mean_delta"], controller)
            p_s = base_probs_on_s(
                model, tokenizer, calib_prompts, calib_cont, token_ids=token_set["token_ids"])
            weights_t = torch.tensor(token_set["weights"], dtype=torch.float64)
            cal = calibrate_scalar(p_s, weights_t, float(tf["target_kl"]))
            log(
                f"concept={cid} controller={controller} tokens={len(token_set['token_ids'])} "
                f"scalar={cal['scalar']:.6f} achieved={cal['achieved_kl']:.6f}"
            )
            doses = []
            for scale in dose_scales:
                scalar = float(cal["scalar"]) * float(scale)
                texts = sparse_bias_texts(
                    model,
                    tokenizer,
                    eval_prompts,
                    token_ids=token_set["token_ids"],
                    weights=token_set["weights"],
                    scalar=scalar,
                    batch_size=args.batch_size,
                    max_new_tokens=max_new_tokens,
                )
                kl = sparse_kl_from_probs(p_s, weights_t, scalar)
                gate = score_generation_gate(texts, base=cr["base"]["eval_gate"])
                doses.append({
                    "controller": controller,
                    "scale": float(scale),
                    "scalar": scalar,
                    "kl": kl,
                    "eval_gate": gate,
                    "eval_texts": texts,
                })
                log(
                    f"concept={cid} controller={controller} scale={scale:g} "
                    f"kl={kl:.6f} clean_gate={not gate['degenerate']}"
                )
            controller_results.append({
                "controller": controller,
                "token_set": token_set,
                "calibration": cal,
                "doses": doses,
            })
            del p_s, weights_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result["concept_runs"].append({
            "concept": cr["concept"],
            "selected_factor": factor,
            "rows": rows,
            "source_reps_selected_eval": sel["source_selected"]["selected_eval"],
            "source_reps_selected_calib": sel["source_selected"]["selected"],
            "recovery_checks": recovery["checks"],
            "generation_smoke": smoke,
            "teacher_forced": {
                "target_kl": float(tf["target_kl"]),
                "positions": int(tf["positions"]),
                "calib_base_regenerated_text_match_count": int(base_match),
                "calib_base_regenerated_text_total": int(len(calib_prompts)),
            },
            "base_eval_texts": cr["base"]["eval_texts"],
            "base_eval_gate": cr["base"]["eval_gate"],
            "method_eval_texts": sweep_row["eval_texts"],
            "method_eval_gate": sweep_row["eval_gate"],
            "controllers": controller_results,
        })
        del recovery, vector, add_vec, tf, calib_cont
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(outdir / "semantic_control_generations.json", result)
    log(f"wrote {outdir / 'semantic_control_generations.json'}")


def run_judge(args: argparse.Namespace) -> None:
    source_outdir = Path(args.source_outdir)
    outdir = Path(args.outdir)
    gens_path = outdir / "semantic_control_generations.json"
    if not gens_path.exists():
        raise SystemExit(f"missing control generation file: {gens_path}")
    gens = load_json(gens_path)
    items = collect_control_items(gens)
    cache_path = outdir / "semantic_control_judge_cache.json"
    preseed_cache(source_outdir / "semantic_judge_cache.json", cache_path)
    cache = asyncio.run(judge_missing(
        items,
        model=args.judge_model,
        cache_path=cache_path,
        batch_size=args.judge_batch_size,
    ))
    judged = ratings_for_items(items, cache, model=args.judge_model)
    summary = summarize_control(
        gens,
        judged,
        n_boot=args.n_boot,
        seed=int(gens["meta"]["seed"]),
        max_component_drop=args.max_component_drop,
        kl_rel_budget=args.kl_rel_budget,
    )
    result = {
        "meta": {
            **gens["meta"],
            "mode": "judge",
            "judge_model": args.judge_model,
            "control_generation_file_sha256": sha256_file(gens_path),
            "control_generation_canonical_sha256": sha256_obj(gens),
            "judge_cache_sha256": sha256_obj(cache),
            "metric": "normalized AxBench LMJudge harmonic aggregate in [0,1]",
        },
        "summary": summary,
        "judged": judged,
    }
    write_json(outdir / "semantic_control_judged.json", result)
    write_report(outdir, result)
    log(f"wrote {outdir / 'semantic_control_judged.json'}")
    log(f"verdict={summary['verdict']['class']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "generate", "judge"], default="dry")
    ap.add_argument("--source-outdir", default=str(DEFAULT_SOURCE_OUTDIR))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--vector-file", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--controllers", default="pos100,pos500,signed500,bal250,baleq250")
    ap.add_argument("--dose-scales", default="0.50,0.60,0.70,0.80,0.90,1.00")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--train-factors", default="2,4,6,8,10,12,14,16,18,20")
    ap.add_argument("--train-gen-batch-size", type=int, default=8)
    ap.add_argument("--dpo-max-new-tokens", type=int, default=192)
    ap.add_argument("--simpo-scaler", type=float, default=1.0)
    ap.add_argument("--bias-tol", type=float, default=1e-6)
    ap.add_argument("--norm-tol", type=float, default=1e-5)
    ap.add_argument("--smoke-per-split", type=int, default=4)
    ap.add_argument("--kl-rel-budget", type=float, default=1.01)
    ap.add_argument("--max-component-drop", type=float, default=0.20)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--judge-batch-size", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=10000)
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
