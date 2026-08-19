"""Director-authorized E3 calibration-size fidelity curve.

This driver is deliberately separate from the frozen Amendment-19 driver and
supplement.  Gate 0 calls the shipped implementation directly at 400 rows.
Only after both arms pass parity may this driver build nested calibration-row
prefixes at 800, 1600, 3200, and 6400 rows.

The long curve is restartable.  Checkpoints are immutable/atomic, bind the
code, model revision, inputs, protocol, and explicit RNG seeds, and fail closed
on any mismatch.  Decision-bearing summaries retain per-position sufficient
statistics so the fidelity headline can be re-derived from raw JSON bytes.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import battery as B  # noqa: E402
import fv_extract as FV  # noqa: E402
import run_output_footprint_distill as A19  # noqa: E402
from actlib.models import resolve_name  # noqa: E402


MODEL_ID = "EleutherAI/pythia-2.8b"
MODEL_REVISION = "2a259cdd96a4beb1cdf467512e3904197345f6a9"
ROW_SIZES = (400, 800, 1600, 3200, 6400)
KL_GRID = (0.03, 0.07, 0.11, 0.15, 0.18)
MAX_ROWS = max(ROW_SIZES)
N_EVAL_TOTAL = 200
N_ORIGINAL_CALIB_PROMPTS = 50
MAX_NEW_TOKENS = 8
RIDGE_FRAC = 1e-2
SEED_GLOBAL = 20260710

SHIPPED_DIR = (
    REPO
    / "projects/steering-content-audit/paper/supplement/results"
    / "2026-07-10-output-footprint-distill"
)
INPUT_PATHS = (
    REPO / "projects/steering-content-audit/exp/run_e3_calibration_curve.py",
    REPO / "projects/steering-content-audit/exp/reduce_e3_calibration_curve.py",
    REPO / "projects/steering-content-audit/exp/run_output_footprint_distill.py",
    REPO / "projects/steering-content-audit/exp/battery.py",
    REPO / "projects/steering-content-audit/exp/fv_extract.py",
    REPO / "projects/steering-content-audit/exp/run_a1.py",
    REPO / "projects/steering-content-audit/exp/taskvec.py",
    REPO / "data/external/function_vectors/dataset_files/abstractive/antonym.json",
    REPO / "runs/steering-content-audit/2026-07-06-a1-anchor/stage1_antonym_pythia-2.8b.pt",
    REPO / "runs/steering-content-audit/2026-07-06-a1-anchor/results_full.json",
    REPO / "runs/steering-content-audit/2026-07-07-taskvec-arm/stage_antonym_full.json",
    REPO / "runs/steering-content-audit/2026-07-07-taskvec-arm/results_full.json",
    REPO / "requirements.lock.txt",
    REPO / "projects/steering-content-audit/mandate_e3_calibration_curve.md",
    REPO / "projects/steering-content-audit/prereg_e3_calibration_curve_20260804.md",
    REPO / "projects/steering-content-audit/e3_calibration_index_manifest_20260804.json",
    SHIPPED_DIR / "fv.json",
    SHIPPED_DIR / "taskvec.json",
)

# Frozen before Gate 0.  Exact structural guards are additional to these
# cross-backend numerical tolerances.
PARITY_TOLERANCE = {
    "coverage_abs": 0.005,
    "frontier_achieved_kl_rel": 0.02,
    "oracle_max_logit_error": 1e-5,
}

_MODEL_SNAPSHOT_MANIFEST: dict | None = None


def execution_device() -> str:
    """Return the Amendment E3-A3 matched execution backend or fail closed."""
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("E3 Stage A requires an available MPS device")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=A19.jsonable)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object JSON at {path}")
    return value


def input_manifest() -> List[dict]:
    paths = list(INPUT_PATHS)
    paths.extend(sorted((REPO / "tools/actlib").glob("*.py")))
    rows = []
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"required frozen input missing: {path}")
        rows.append({
            "path": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def stable_environment_identity() -> dict:
    import transformers

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    driver = None
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[0]
        except Exception as exc:  # recorded and then held fixed, not ignored
            driver = f"UNAVAILABLE:{type(exc).__name__}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numpy_config": buf.getvalue(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cuda_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None
        ),
        "nvidia_driver": driver,
        "tf32_allowed": (
            bool(torch.backends.cuda.matmul.allow_tf32)
            if torch.cuda.is_available() else None
        ),
    }


def model_snapshot_manifest() -> dict:
    global _MODEL_SNAPSHOT_MANIFEST
    if _MODEL_SNAPSHOT_MANIFEST is not None:
        return _MODEL_SNAPSHOT_MANIFEST

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    rows = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        rows.append({
            "path": str(path.relative_to(snapshot)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if not rows:
        raise RuntimeError(f"empty model snapshot: {snapshot}")
    _MODEL_SNAPSHOT_MANIFEST = {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": rows,
        "manifest_sha256": canonical_sha(rows),
    }
    return _MODEL_SNAPSHOT_MANIFEST


def hydrate_model_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION))


def identity_payload(arm: str) -> dict:
    if arm not in {"fv", "taskvec"}:
        raise ValueError(arm)
    payload = {
        "schema": "steering-content-audit-e3-identity-v1",
        "arm": arm,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "torch.bfloat16",
        "device": execution_device(),
        "row_sizes": list(ROW_SIZES),
        "kl_grid": list(KL_GRID),
        "ridge_frac": RIDGE_FRAC,
        "n_eval_total": N_EVAL_TOTAL,
        "n_original_calib_prompts": N_ORIGINAL_CALIB_PROMPTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed_global": SEED_GLOBAL,
        "n_boot": 10000,
        "arm_seed": 20260706 if arm == "fv" else 20260707,
        "calibration_extension_rule": (
            "original 50 calibration prompts first; then frozen train-pool "
            "order after excluding reconstructed intervention-construction "
            "indices and heldout-lexeme overlaps; admit only exact eight-row "
            "prompt blocks and skip short eligible prompts"
        ),
        "parity_tolerance": PARITY_TOLERANCE,
        "runtime_environment": stable_environment_identity(),
        "model_snapshot": model_snapshot_manifest(),
        "inputs": input_manifest(),
    }
    payload["identity_sha256"] = canonical_sha(payload)
    return payload


def ensure_identity(root: Path, arm: str) -> dict:
    expected = identity_payload(arm)
    path = root / "identity" / f"{arm}.json"
    if path.exists():
        observed = load_json(path)
        if observed != expected:
            raise RuntimeError(
                f"resume identity mismatch for {arm}: "
                f"stored={observed.get('identity_sha256')} "
                f"current={expected.get('identity_sha256')}"
            )
    else:
        atomic_write_json(path, expected)
    return expected


def install_pinned_loader() -> None:
    """Pin Pythia to the revision used by the original local cache."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    original = B.load_model

    def pinned(name: str, device: str = "cpu", dtype=None, seed: int = 0):
        hf_id = resolve_name(name)
        if hf_id != MODEL_ID:
            return original(name, device=device, dtype=dtype, seed=seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        expected_device = execution_device()
        if device != expected_device:
            raise RuntimeError(
                f"E3 requires device={expected_device}, got {device}")
        if dtype is not torch.bfloat16:
            raise RuntimeError(f"E3 requires bfloat16, got {dtype}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
        model.to(device)
        model.eval()
        observed = getattr(model.config, "_commit_hash", None)
        if observed not in (None, MODEL_REVISION):
            raise RuntimeError(
                f"model revision mismatch: {observed} != {MODEL_REVISION}")
        return model, tokenizer

    B.load_model = pinned
    A19.B.load_model = pinned


def configure_determinism() -> None:
    torch.manual_seed(SEED_GLOBAL)
    np.random.seed(SEED_GLOBAL)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_GLOBAL)
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def environment_record() -> dict:
    return {
        "timestamp": utc_now(),
        **stable_environment_identity(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
    }


def compare_scalar(name: str, observed, expected, tol: float) -> dict:
    delta = abs(float(observed) - float(expected))
    return {
        "name": name,
        "observed": observed,
        "expected": expected,
        "abs_delta": delta,
        "tolerance": tol,
        "pass": delta <= tol,
    }


def adjudicate_parity(result: dict, shipped: dict) -> dict:
    checks: List[dict] = []
    exact = [
        ("fit.n_rows", result["fit"]["n_rows"], shipped["fit"]["n_rows"]),
        ("fit.feature_dim", result["fit"]["feature_dim"], shipped["fit"]["feature_dim"]),
        ("fit.vocab_size", result["fit"]["vocab_size"], shipped["fit"]["vocab_size"]),
        ("fit.effective_rank", result["fit"]["effective_rank"], shipped["fit"]["effective_rank"]),
        ("frozen_count_guard.pass", result["frozen_count_guard"]["pass"], True),
        ("rates.base.count", result["rates"]["base"]["count"], shipped["rates"]["base"]["count"]),
        ("rates.native.count", result["rates"]["native"]["count"], shipped["rates"]["native"]["count"]),
        ("base_hits", result["base_hits"], shipped["base_hits"]),
        ("native_hits", result["native_hits"], shipped["native_hits"]),
        ("n_frontier", len(result["frontier"]), len(shipped["frontier"])),
        ("fidelity.all_top1_agreement", result["heldout_footprint_fidelity"]["all_top1_agreement"], shipped["heldout_footprint_fidelity"]["all_top1_agreement"]),
        ("fidelity.changed_top1_recovery", result["heldout_footprint_fidelity"]["changed_top1_recovery"], shipped["heldout_footprint_fidelity"]["changed_top1_recovery"]),
        ("fidelity.native_flip_positions", result["heldout_footprint_fidelity"]["native_flip_positions"], shipped["heldout_footprint_fidelity"]["native_flip_positions"]),
        ("fidelity.native_flip_recovered", result["heldout_footprint_fidelity"]["native_flip_recovered"], shipped["heldout_footprint_fidelity"]["native_flip_recovered"]),
    ]
    for name, observed, expected in exact:
        checks.append({
            "name": name, "observed": observed, "expected": expected,
            "pass": observed == expected, "mode": "exact",
        })

    rf = result["heldout_footprint_fidelity"]
    sf = shipped["heldout_footprint_fidelity"]
    checks.extend([
        compare_scalar("fidelity.coverage", rf["coverage"], sf["coverage"],
                       PARITY_TOLERANCE["coverage_abs"]),
    ])
    checks.append({
        "name": "oracle_guard.pass",
        "observed": result["oracle_guard"]["pass"],
        "expected": True,
        "pass": bool(result["oracle_guard"]["pass"]),
        "mode": "exact",
    })
    checks.append({
        "name": "oracle_guard.max_logit_error",
        "observed": result["oracle_guard"]["max_logit_error"],
        "tolerance": PARITY_TOLERANCE["oracle_max_logit_error"],
        "pass": result["oracle_guard"]["max_logit_error"]
        <= PARITY_TOLERANCE["oracle_max_logit_error"],
    })

    for idx, (row, old) in enumerate(zip(result["frontier"], shipped["frontier"])):
        prefix = f"frontier[{idx}]"
        checks.append({
            "name": f"{prefix}.target_kl", "observed": row["target_kl"],
            "expected": old["target_kl"],
            "pass": float(row["target_kl"]) == float(old["target_kl"]),
            "mode": "exact",
        })
        rel = abs(row["achieved_kl"] - old["achieved_kl"]) / max(
            abs(old["achieved_kl"]), 1e-12)
        checks.append({
            "name": f"{prefix}.achieved_kl", "observed": row["achieved_kl"],
            "expected": old["achieved_kl"], "rel_delta": rel,
            "tolerance": PARITY_TOLERANCE["frontier_achieved_kl_rel"],
            "pass": rel <= PARITY_TOLERANCE["frontier_achieved_kl_rel"],
        })
        for suffix, observed, expected in (
            ("hits", row["hits"], old["hits"]),
            ("rate", row["rate"], old["rate"]),
            ("rho", row["rho"], old["rho"]),
        ):
            checks.append({
                "name": f"{prefix}.{suffix}", "observed": observed,
                "expected": expected, "pass": observed == expected,
                "mode": "exact",
            })
        checks.append({
            "name": f"{prefix}.gate.tripped",
            "observed": row["gate"]["tripped"],
            "expected": old["gate"]["tripped"],
            "pass": row["gate"]["tripped"] == old["gate"]["tripped"],
            "mode": "exact",
        })
    return {
        "schema": "steering-content-audit-e3-parity-v1",
        "pass": all(row["pass"] for row in checks),
        "checks": checks,
        "tolerance": PARITY_TOLERANCE,
    }


def run_parity(root: Path, arm_name: str, n_boot: int) -> Path:
    identity = ensure_identity(root, arm_name)
    out = root / "parity" / f"{arm_name}.json"
    decision_path = root / "parity" / f"{arm_name}_decision.json"
    if out.exists() and decision_path.exists():
        decision = load_json(decision_path)
        if decision.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("parity artifact identity mismatch")
        log(f"reuse completed parity {arm_name}: pass={decision['pass']}")
        return decision_path

    class Args:
        smoke = False
        device = execution_device()

    args = Args()
    args.n_boot = n_boot
    log(f"Gate 0 parity start arm={arm_name}")
    result = A19.run_antonym_arm(arm_name, args)
    result["e3_identity_sha256"] = identity["identity_sha256"]
    result["e3_environment"] = environment_record()
    atomic_write_json(out, result)
    shipped = load_json(SHIPPED_DIR / f"{arm_name}.json")
    parity = adjudicate_parity(result, shipped)
    parity.update({
        "arm": arm_name,
        "timestamp": utc_now(),
        "identity_sha256": identity["identity_sha256"],
        "raw_result": str(out.relative_to(root)),
        "raw_result_sha256": sha256_file(out),
        "shipped_sha256": sha256_file(SHIPPED_DIR / f"{arm_name}.json"),
    })
    atomic_write_json(decision_path, parity)
    log(f"Gate 0 parity arm={arm_name} pass={parity['pass']}")
    return decision_path


def require_both_parity(root: Path) -> dict:
    decisions = {}
    for arm in ("fv", "taskvec"):
        path = root / "parity" / f"{arm}_decision.json"
        if not path.is_file():
            raise RuntimeError(
                f"blocking Gate 0 incomplete: missing {path}; no larger N allowed")
        value = load_json(path)
        expected = ensure_identity(root, arm)["identity_sha256"]
        if value.get("identity_sha256") != expected:
            raise RuntimeError(f"stale Gate 0 identity for {arm}")
        if not value.get("pass"):
            raise RuntimeError(
                f"blocking Gate 0 failed for {arm}; no larger N allowed")
        decisions[arm] = value
    return decisions


def normalized_lexeme(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def sampled_train_dataset_indices(train_idx: Sequence[int], n_prompts: int,
                                  n_shots: int, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    for _ in range(n_prompts):
        picked = rng.choice(len(train_idx), size=n_shots + 1, replace=False)
        used.update(int(train_idx[int(i)]) for i in picked)
    return used


def calibration_index_plan(arm_name: str) -> dict:
    if arm_name not in {"fv", "taskvec"}:
        raise ValueError(arm_name)
    seed = 20260706 if arm_name == "fv" else 20260707
    pairs = FV.load_pairs(str(A19.DATA_DIR + "/antonym.json"))
    split_rng = np.random.default_rng(seed)
    permutation = split_rng.permutation(len(pairs))
    eval_idx = [int(x) for x in permutation[:N_EVAL_TOTAL]]
    train_idx = [int(x) for x in permutation[N_EVAL_TOTAL:]]
    eval_pairs = [pairs[i] for i in eval_idx]
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(eval_pairs))
    original_idx = [eval_idx[int(i)] for i in order[:N_ORIGINAL_CALIB_PROMPTS]]
    heldout_idx = [eval_idx[int(i)] for i in order[N_ORIGINAL_CALIB_PROMPTS:]]
    original = [pairs[i] for i in original_idx]
    heldout = [pairs[i] for i in heldout_idx]
    frozen_path = (
        REPO / "runs/steering-content-audit"
        / ("2026-07-06-a1-anchor" if arm_name == "fv" else "2026-07-07-taskvec-arm")
        / "results_full.json"
    )
    result_meta = load_json(frozen_path)["meta"]
    n_mean = int(result_meta["n_mean"])
    n_shots = int(result_meta["n_shots"])
    construction_used = sampled_train_dataset_indices(
        train_idx, n_mean, n_shots, seed + 2)
    if arm_name == "fv":
        construction_used |= sampled_train_dataset_indices(
            train_idx, int(result_meta["n_cie"]), n_shots, seed + 3)

    heldout_lexemes = {
        normalized_lexeme(value)
        for idx in heldout_idx for value in pairs[idx]
    }
    eligible_idx = []
    for idx in train_idx:
        if idx in construction_used:
            continue
        x, y = pairs[idx]
        if {normalized_lexeme(x), normalized_lexeme(y)} & heldout_lexemes:
            continue
        eligible_idx.append(idx)
    if len(eligible_idx) < (MAX_ROWS // MAX_NEW_TOKENS - N_ORIGINAL_CALIB_PROMPTS):
        raise RuntimeError(
            f"only {len(eligible_idx)} uncontaminated extension pairs")
    return {
        "arm": arm_name,
        "seed": seed,
        "dataset_sha256": sha256_file(
            REPO / "data/external/function_vectors/dataset_files/abstractive/antonym.json"),
        "dataset_size": len(pairs),
        "original_calibration_indices": original_idx,
        "heldout_indices": heldout_idx,
        "construction_used_train_indices": sorted(construction_used),
        "construction_used_count": len(construction_used),
        "heldout_normalized_lexemes": sorted(heldout_lexemes),
        "eligible_extension_indices_in_order": eligible_idx,
        "eligible_extension_count": len(eligible_idx),
        "rule": (
            "exclude every reconstructed intervention-construction dataset index; "
            "exclude any candidate whose normalized input or output lexeme occurs "
            "in heldout evaluation; preserve frozen train-pool order"
        ),
    }


def calibration_pairs(arm: A19.Arm) -> List[Tuple[int, str, str, str]]:
    plan = calibration_index_plan(arm.name)
    pairs = FV.load_pairs(str(A19.DATA_DIR + "/antonym.json"))
    original_idx = plan["original_calibration_indices"]
    heldout_idx = plan["heldout_indices"]
    original = [pairs[i] for i in original_idx]
    heldout = [pairs[i] for i in heldout_idx]
    if [FV.zero_shot_prompt(x) for x, _ in original] != arm.calib_prompts:
        raise RuntimeError("original calibration prompt sequence mismatch")
    if [y for _, y in original] != arm.calib_golds:
        raise RuntimeError("original calibration gold sequence mismatch")
    if [FV.zero_shot_prompt(x) for x, _ in heldout] != arm.eval_prompts:
        raise RuntimeError("held-out prompt sequence mismatch")
    if [y for _, y in heldout] != arm.eval_golds:
        raise RuntimeError("held-out gold sequence mismatch")
    out = [
        (idx, pairs[idx][0], pairs[idx][1], "original_eval_calibration")
        for idx in original_idx
    ]
    out.extend(
        (idx, pairs[idx][0], pairs[idx][1], "unused_train_pool_no_heldout_lexeme")
        for idx in plan["eligible_extension_indices_in_order"]
    )
    return out


def array_record(path: Path, array: np.ndarray) -> dict:
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "array_sha256": A19.sha_array(np.asarray(array)),
    }


def load_checked_array(directory: Path, record: dict) -> np.ndarray:
    path = directory / record["path"]
    if not path.is_file() or sha256_file(path) != record["file_sha256"]:
        raise RuntimeError(f"checkpoint file mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
        raise RuntimeError(f"checkpoint array metadata mismatch: {path}")
    return array


def prepare_calibration_bank(root: Path, arm: A19.Arm, identity: dict
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    directory = root / "checkpoints" / arm.name / "calibration_bank_6400"
    meta_path = directory / "manifest.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("calibration checkpoint identity mismatch")
        X = load_checked_array(directory, meta["arrays"]["X"])
        base = load_checked_array(directory, meta["arrays"]["base_logits"])
        Y = load_checked_array(directory, meta["arrays"]["Y"])
        return X, base, Y, meta

    candidates = calibration_pairs(arm)
    shard_dir = directory / "prompt_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.json"
    if progress_path.exists():
        progress = load_json(progress_path)
        if progress.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("calibration progress identity mismatch")
    else:
        progress = {
            "schema": "steering-content-audit-e3-calibration-progress-v1",
            "identity_sha256": identity["identity_sha256"],
            "candidate_cursor": 0,
            "admitted": [],
            "skipped_short_candidates": [],
        }
        atomic_write_json(progress_path, progress)

    for candidate_pos in range(int(progress["candidate_cursor"]), len(candidates)):
        dataset_idx, x, y, source = candidates[candidate_pos]
        prompt = FV.zero_shot_prompt(x)
        ids, _text = A19.base_generate_ids_text(
            arm.model, arm.tokenizer, prompt, arm.max_new_tokens, arm.device)
        if len(ids) != MAX_NEW_TOKENS:
            if source == "original_eval_calibration":
                raise RuntimeError(
                    f"Gate-0 calibration prompt {dataset_idx} yielded "
                    f"{len(ids)} rows, expected {MAX_NEW_TOKENS}")
            progress["skipped_short_candidates"].append({
                "dataset_index": dataset_idx,
                "observed_rows": len(ids),
                "rule": "skip and take next eligible candidate",
            })
            progress["candidate_cursor"] = candidate_pos + 1
            atomic_write_json(progress_path, progress)
            continue

        X_one, base_one, spans_one = A19.collect_base_rows(
            arm.model, arm.tokenizer, [prompt], [ids], arm.device)
        Y_one = A19.collect_native_targets(
            arm, [prompt], [ids], base_one)
        if spans_one != [(0, MAX_NEW_TOKENS)]:
            raise RuntimeError("prompt shard row-boundary mismatch")
        admitted_index = len(progress["admitted"])
        shard_path = shard_dir / (
            f"shard_{admitted_index:04d}_dataset_{dataset_idx}.npz")
        tmp = shard_path.with_name(f".{shard_path.name}.tmp-{os.getpid()}")
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                X=np.asarray(X_one, dtype=np.float32),
                base_logits=np.asarray(base_one, dtype=np.float32),
                Y=np.asarray(Y_one, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, shard_path)
        progress["admitted"].append({
            "dataset_index": dataset_idx,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "gold_sha256": hashlib.sha256(y.encode()).hexdigest(),
            "source": source,
            "continuation_ids": list(ids),
            "rows_used": MAX_NEW_TOKENS,
            "shard": shard_path.name,
            "shard_sha256": sha256_file(shard_path),
        })
        progress["candidate_cursor"] = candidate_pos + 1
        atomic_write_json(progress_path, progress)
        if len(progress["admitted"]) % 25 == 0:
            log(f"calibration prompt shards arm={arm.name} "
                f"{len(progress['admitted'])}/{MAX_ROWS // MAX_NEW_TOKENS}")
        if len(progress["admitted"]) == MAX_ROWS // MAX_NEW_TOKENS:
            break
    if len(progress["admitted"]) != MAX_ROWS // MAX_NEW_TOKENS:
        raise RuntimeError(
            f"insufficient admitted calibration prompts: "
            f"{len(progress['admitted'])}")

    xs, bases, ys = [], [], []
    for row in progress["admitted"]:
        shard_path = shard_dir / row["shard"]
        if sha256_file(shard_path) != row["shard_sha256"]:
            raise RuntimeError(f"prompt shard byte mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as shard:
            xs.append(shard["X"])
            bases.append(shard["base_logits"])
            ys.append(shard["Y"])
    X_all = np.concatenate(xs, axis=0)
    base_all = np.concatenate(bases, axis=0)
    Y_all = np.concatenate(ys, axis=0)
    if MAX_ROWS % MAX_NEW_TOKENS:
        raise RuntimeError("row ladder cuts through a prompt")
    X = np.asarray(X_all[:MAX_ROWS], dtype=np.float32)
    base = np.asarray(base_all[:MAX_ROWS], dtype=np.float32)
    Y = np.asarray(Y_all[:MAX_ROWS], dtype=np.float32)

    parity = load_json(root / "parity" / f"{arm.name}.json")
    prefix_checks = {
        "X_sha256_observed": A19.sha_array(X[:400]),
        "X_sha256_expected": parity["fit"]["X_sha256"],
        "Y_sha256_observed": A19.sha_array(Y[:400]),
        "Y_sha256_expected": parity["fit"]["Y_sha256"],
    }
    prefix_checks["pass"] = (
        prefix_checks["X_sha256_observed"] == prefix_checks["X_sha256_expected"]
        and prefix_checks["Y_sha256_observed"] == prefix_checks["Y_sha256_expected"]
    )
    if not prefix_checks["pass"]:
        raise RuntimeError(f"400-row prefix drifted from Gate 0: {prefix_checks}")

    directory.mkdir(parents=True, exist_ok=True)
    array_paths = {
        "X": directory / "X.npy",
        "base_logits": directory / "base_logits.npy",
        "Y": directory / "Y.npy",
    }
    for key, array in (("X", X), ("base_logits", base), ("Y", Y)):
        atomic_save_npy(array_paths[key], array)

    prompt_rows = []
    for index, row in enumerate(progress["admitted"]):
        prompt_rows.append({
            **{k: row[k] for k in (
                "dataset_index", "prompt_sha256", "gold_sha256", "source",
                "continuation_ids", "rows_used", "shard", "shard_sha256")},
            "source_span": [
                index * MAX_NEW_TOKENS, (index + 1) * MAX_NEW_TOKENS],
        })

    meta = {
        "schema": "steering-content-audit-e3-calibration-bank-v1",
        "timestamp": utc_now(),
        "identity_sha256": identity["identity_sha256"],
        "n_rows": MAX_ROWS,
        "n_prompts": len(prompt_rows),
        "original_prompt_count": sum(
            r["source"] == "original_eval_calibration" for r in prompt_rows),
        "extension_prompt_count": sum(
            r["source"] == "unused_train_pool_no_heldout_lexeme"
            for r in prompt_rows),
        "skipped_short_candidates": progress["skipped_short_candidates"],
        "progress_file_sha256": sha256_file(progress_path),
        "prefix_400_guard": prefix_checks,
        "prompt_rows": prompt_rows,
        "arrays": {
            key: array_record(path, array)
            for (key, path), array in zip(
                array_paths.items(), (X, base, Y))
        },
    }
    atomic_write_json(meta_path, meta)
    return (
        np.load(array_paths["X"], mmap_mode="r"),
        np.load(array_paths["base_logits"], mmap_mode="r"),
        np.load(array_paths["Y"], mmap_mode="r"),
        meta,
    )


def prepare_heldout_bank(root: Path, arm: A19.Arm, identity: dict
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    directory = root / "checkpoints" / arm.name / "heldout_bank"
    meta_path = directory / "manifest.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("heldout checkpoint identity mismatch")
        X = load_checked_array(directory, meta["arrays"]["X"])
        base = load_checked_array(directory, meta["arrays"]["base_logits"])
        Y = load_checked_array(directory, meta["arrays"]["Y"])
        return X, base, Y, meta

    base_ids: List[List[int]] = []
    base_texts: List[str] = []
    native_texts: List[str] = []
    for idx, prompt in enumerate(arm.eval_prompts):
        ids, text = A19.base_generate_ids_text(
            arm.model, arm.tokenizer, prompt, arm.max_new_tokens, arm.device)
        base_ids.append(ids)
        base_texts.append(text)
        native_texts.append(arm.native_generate(prompt))
        if (idx + 1) % 25 == 0:
            log(f"heldout base/native arm={arm.name} {idx+1}/{len(arm.eval_prompts)}")
    base_hits = [
        int(arm.hit(text, gold))
        for text, gold in zip(base_texts, arm.eval_golds)
    ]
    native_hits = [
        int(arm.hit(text, gold))
        for text, gold in zip(native_texts, arm.eval_golds)
    ]
    parity = load_json(root / "parity" / f"{arm.name}.json")
    replay_guard = {
        "base_texts_exact": base_texts == parity["base_texts"],
        "native_texts_exact": native_texts == parity["native_texts"],
        "base_hits_exact": base_hits == parity["base_hits"],
        "native_hits_exact": native_hits == parity["native_hits"],
    }
    replay_guard["pass"] = all(replay_guard.values())
    if not replay_guard["pass"]:
        raise RuntimeError(f"heldout replay drifted after Gate 0: {replay_guard}")

    X, base, spans = A19.collect_base_rows(
        arm.model, arm.tokenizer, arm.eval_prompts, base_ids, arm.device)
    Y = A19.collect_native_targets(arm, arm.eval_prompts, base_ids, base)
    X = np.asarray(X, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    array_paths = {
        "X": directory / "X.npy",
        "base_logits": directory / "base_logits.npy",
        "Y": directory / "Y.npy",
    }
    for key, array in (("X", X), ("base_logits", base), ("Y", Y)):
        atomic_save_npy(array_paths[key], array)
    meta = {
        "schema": "steering-content-audit-e3-heldout-bank-v1",
        "timestamp": utc_now(),
        "identity_sha256": identity["identity_sha256"],
        "n_prompts": len(arm.eval_prompts),
        "n_rows": len(X),
        "prompt_sha256": [
            hashlib.sha256(p.encode()).hexdigest() for p in arm.eval_prompts
        ],
        "gold_sha256": [
            hashlib.sha256(g.encode()).hexdigest() for g in arm.eval_golds
        ],
        "continuation_ids": base_ids,
        "spans": [list(x) for x in spans],
        "base_texts": base_texts,
        "native_texts": native_texts,
        "base_hits": base_hits,
        "native_hits": native_hits,
        "gate0_replay_guard": replay_guard,
        "arrays": {
            key: array_record(path, array)
            for (key, path), array in zip(
                array_paths.items(), (X, base, Y))
        },
    }
    atomic_write_json(meta_path, meta)
    return (
        np.load(array_paths["X"], mmap_mode="r"),
        np.load(array_paths["base_logits"], mmap_mode="r"),
        np.load(array_paths["Y"], mmap_mode="r"),
        meta,
    )


def derive_fidelity_arrays(native_kl_rows: Sequence[float],
                           residual_kl_rows: Sequence[float],
                           base_top1: Sequence[int],
                           native_top1: Sequence[int],
                           pred_top1: Sequence[int]) -> dict:
    n = np.asarray(native_top1)
    b = np.asarray(base_top1)
    p = np.asarray(pred_top1)
    nk = np.asarray(native_kl_rows, dtype=np.float64)
    rk = np.asarray(residual_kl_rows, dtype=np.float64)
    flips = n != b
    return {
        "coverage": 1.0 - float(rk.sum()) / max(float(nk.sum()), 1e-12),
        "mean_native_kl": float(nk.mean()),
        "mean_residual_kl": float(rk.mean()),
        "all_top1_agreement": float(np.mean(n == p)),
        "native_flip_positions": int(flips.sum()),
        "native_flip_recovered": int(np.sum((n == p) & flips)),
        "changed_top1_recovery": (
            float(np.sum((n == p) & flips)) / max(int(flips.sum()), 1)),
    }


def prompt_cluster_bootstrap(raw: dict, spans: Sequence[Sequence[int]],
                             seed: int, n_boot: int) -> dict:
    nk = np.asarray(raw["native_kl_rows"], dtype=np.float64)
    rk = np.asarray(raw["residual_kl_rows"], dtype=np.float64)
    base = np.asarray(raw["base_top1"])
    native = np.asarray(raw["native_top1"])
    pred = np.asarray(raw["pred_top1"])
    clusters = []
    for start, stop in spans:
        sl = slice(int(start), int(stop))
        flip = native[sl] != base[sl]
        clusters.append({
            "native_kl": float(nk[sl].sum()),
            "residual_kl": float(rk[sl].sum()),
            "agree": int(np.sum(native[sl] == pred[sl])),
            "n": int(stop - start),
            "flips": int(flip.sum()),
            "recovered": int(np.sum((native[sl] == pred[sl]) & flip)),
        })
    rng = np.random.default_rng(seed)
    coverage, agreement, recovery = [], [], []
    for _ in range(n_boot):
        draw = rng.integers(0, len(clusters), size=len(clusters))
        selected = [clusters[int(i)] for i in draw]
        native_sum = sum(x["native_kl"] for x in selected)
        residual_sum = sum(x["residual_kl"] for x in selected)
        coverage.append(1.0 - residual_sum / max(native_sum, 1e-12))
        agreement.append(
            sum(x["agree"] for x in selected)
            / max(sum(x["n"] for x in selected), 1))
        flips = sum(x["flips"] for x in selected)
        if flips:
            recovery.append(sum(x["recovered"] for x in selected) / flips)
    def ci(values: Sequence[float]) -> dict:
        return {
            "lo": float(np.quantile(values, 0.025)),
            "hi": float(np.quantile(values, 0.975)),
        }
    return {
        "seed": seed,
        "n_boot": n_boot,
        "resampling_unit": "heldout prompt",
        "coverage": ci(coverage),
        "all_top1_agreement": ci(agreement),
        "changed_top1_recovery": ci(recovery),
    }


def position_strata(raw: dict, spans: Sequence[Sequence[int]]) -> List[dict]:
    positions: Dict[int, List[int]] = {}
    for start, stop in spans:
        for row in range(int(start), int(stop)):
            positions.setdefault(row - int(start) + 1, []).append(row)
    rows = []
    for position, indices in sorted(positions.items()):
        subset = {
            key: [raw[key][i] for i in indices]
            for key in ("native_kl_rows", "residual_kl_rows", "base_top1",
                        "native_top1", "pred_top1")
        }
        rows.append({
            "decode_position": position,
            "n_rows": len(indices),
            **derive_fidelity_arrays(**subset),
        })
    return rows


def fidelity_raw(fit: A19.RidgeFootprint, X: np.ndarray,
                 base: np.ndarray, Y: np.ndarray,
                 spans: Sequence[Sequence[int]], seed: int,
                 n_boot: int) -> dict:
    native_kl_rows: List[float] = []
    residual_kl_rows: List[float] = []
    base_top1: List[int] = []
    native_top1: List[int] = []
    pred_top1: List[int] = []
    batch = 64
    for start in range(0, len(X), batch):
        stop = min(start + batch, len(X))
        xb = torch.from_numpy(np.asarray(X[start:stop])).to(fit.device)
        b = torch.from_numpy(np.asarray(base[start:stop])).to(fit.device).float()
        y = torch.from_numpy(np.asarray(Y[start:stop])).to(fit.device).float()
        with torch.no_grad():
            pred = fit.predict(xb).float()
            native_lp = torch.log_softmax(b + y, dim=-1)
            base_lp = torch.log_softmax(b, dim=-1)
            pred_lp = torch.log_softmax(b + pred, dim=-1)
            nk = (native_lp.exp() * (native_lp - base_lp)).sum(dim=-1)
            rk = (native_lp.exp() * (native_lp - pred_lp)).sum(dim=-1)
        native_kl_rows.extend(nk.cpu().tolist())
        residual_kl_rows.extend(rk.cpu().tolist())
        base_top1.extend(b.argmax(dim=-1).cpu().tolist())
        native_top1.extend((b + y).argmax(dim=-1).cpu().tolist())
        pred_top1.extend((b + pred).argmax(dim=-1).cpu().tolist())

    raw = {
        "native_kl_rows": native_kl_rows,
        "residual_kl_rows": residual_kl_rows,
        "base_top1": base_top1,
        "native_top1": native_top1,
        "pred_top1": pred_top1,
    }
    return {
        "raw": raw,
        "derived": derive_fidelity_arrays(**raw),
        "prompt_cluster_bootstrap_95": prompt_cluster_bootstrap(
            raw, spans, seed, n_boot),
        "by_decode_position": position_strata(raw, spans),
    }


def fit_paths(root: Path, arm: str, n_rows: int) -> Tuple[Path, Path]:
    directory = root / "checkpoints" / arm / f"fit_{n_rows}"
    return directory / "params.npz", directory / "manifest.json"


def save_fit(root: Path, arm: str, n_rows: int, fit: A19.RidgeFootprint,
             identity: dict) -> None:
    params_path, manifest_path = fit_paths(root, arm, n_rows)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = params_path.with_name(f".{params_path.name}.tmp-{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            mu=fit.mu.detach().cpu().numpy(),
            sigma=fit.sigma.detach().cpu().numpy(),
            intercept=fit.intercept.detach().cpu().numpy(),
            V=fit.V.detach().cpu().numpy(),
            shrink=fit.shrink.detach().cpu().numpy(),
            C=fit.C.detach().cpu().numpy(),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, params_path)
    manifest = {
        "schema": "steering-content-audit-e3-fit-checkpoint-v1",
        "identity_sha256": identity["identity_sha256"],
        "n_rows": n_rows,
        "params_file": params_path.name,
        "params_sha256": sha256_file(params_path),
        "fit_meta": fit.meta,
        "timestamp": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)


def load_fit(root: Path, arm: str, n_rows: int, device: str,
             identity: dict) -> A19.RidgeFootprint | None:
    params_path, manifest_path = fit_paths(root, arm, n_rows)
    if not params_path.exists() or not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    if manifest.get("identity_sha256") != identity["identity_sha256"]:
        raise RuntimeError("fit checkpoint identity mismatch")
    if sha256_file(params_path) != manifest["params_sha256"]:
        raise RuntimeError("fit checkpoint byte mismatch")
    with np.load(params_path, allow_pickle=False) as data:
        return A19.RidgeFootprint(
            data["mu"], data["sigma"], data["intercept"], data["V"],
            data["shrink"], torch.from_numpy(data["C"].copy()), device,
            manifest["fit_meta"],
        )


def gate_with_raw(texts: Sequence[str], prompts: Sequence[str], refs: dict,
                  arm: A19.Arm) -> dict:
    reps = [B.three_gram_rep_rate(t, arm.tokenizer) for t in texts]
    lengths = [len(arm.tokenizer.encode(t)) for t in texts]
    nlls = [
        B.mean_nll_under_model(
            arm.model, arm.tokenizer, p, t, device=arm.device)
        for p, t in zip(prompts, texts)
    ]
    rep = float(np.mean(reps)) if reps else 0.0
    median_len = float(np.median(lengths)) if lengths else 0.0
    nll = float(np.mean(nlls)) if nlls else 0.0
    reasons = []
    if rep > 2 * refs["rep"] + 0.1:
        reasons.append("rep")
    if median_len < 0.5 * refs["median_len"]:
        reasons.append("length")
    if nll > 3 * refs["nll"]:
        reasons.append("nll")
    return {
        "raw": {"rep_per_text": reps, "length_per_text": lengths,
                "nll_per_text": nlls},
        "refs": refs,
        "derived": {
            "tripped": bool(reasons), "rep": rep,
            "median_len": median_len, "nll": nll,
            "reason_codes": reasons,
        },
    }


def largest_frontier(root: Path, arm: A19.Arm, fit: A19.RidgeFootprint,
                     X: np.ndarray, base: np.ndarray, heldout_meta: dict,
                     identity: dict, n_boot: int) -> List[dict]:
    directory = root / "curve" / arm.name / "frontier_6400"
    directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        predicted = fit.predict(
            torch.from_numpy(np.asarray(X[:MAX_ROWS])).to(arm.device))
    budgets = []
    for target in KL_GRID:
        scalar, achieved, rows = A19.bisect_scalar(
            np.asarray(base[:MAX_ROWS]), predicted, target, arm.device)
        budgets.append({
            "target_kl": target, "scalar": scalar,
            "achieved_kl": achieved,
            "rel_error": abs(achieved - target) / target,
            "per_step_kl_rows": rows,
        })

    refs = arm.frozen["eval_baseline_refs"]
    base_hits = heldout_meta["base_hits"]
    native_hits = heldout_meta["native_hits"]
    completed = []
    for budget in budgets:
        target = budget["target_kl"]
        target_path = directory / f"kl_{target:.2f}.json"
        if target_path.exists():
            row = load_json(target_path)
            if row.get("identity_sha256") != identity["identity_sha256"]:
                raise RuntimeError("frontier checkpoint identity mismatch")
            completed.append(row)
            continue
        texts = []
        token_ids = []
        for idx, prompt in enumerate(arm.eval_prompts):
            ids, text = A19.generate_with_policy(
                arm.model, arm.tokenizer, prompt, arm.max_new_tokens,
                arm.device,
                lambda phi, s=budget["scalar"]: s * fit.predict(phi),
            )
            token_ids.append(ids)
            texts.append(text)
            if (idx + 1) % 25 == 0:
                log(f"largest frontier arm={arm.name} KL={target:.2f} {idx+1}/{len(texts) if False else len(arm.eval_prompts)}")
        hits = [
            int(arm.hit(text, gold))
            for text, gold in zip(texts, arm.eval_golds)
        ]
        row = {
            **budget,
            "schema": "steering-content-audit-e3-frontier-row-v1",
            "identity_sha256": identity["identity_sha256"],
            "n_rows": MAX_ROWS,
            "texts": texts,
            "token_ids": token_ids,
            "hits": hits,
            "rate": A19.rate_dict(
                hits, int(arm.metadata["seed"]) + 101, n_boot),
            "rho": A19.rho_dict(
                hits, native_hits, base_hits,
                int(arm.metadata["seed"]) + 102, n_boot),
            "gate": gate_with_raw(texts, arm.eval_prompts, refs, arm),
        }
        atomic_write_json(target_path, row)
        completed.append(row)
    return sorted(completed, key=lambda x: x["target_kl"])


def run_curve(root: Path, arm_name: str, n_boot: int) -> Path:
    require_both_parity(root)
    identity = ensure_identity(root, arm_name)
    final_path = root / "curve" / arm_name / "curve.json"
    if final_path.exists():
        value = load_json(final_path)
        if value.get("identity_sha256") != identity["identity_sha256"]:
            raise RuntimeError("completed curve identity mismatch")
        log(f"reuse completed curve {arm_name}")
        return final_path

    class Args:
        smoke = False
        device = execution_device()

    log(f"curve start arm={arm_name}")
    arm = A19.build_antonym_arm(arm_name, Args())
    X, base, Y, calib_meta = prepare_calibration_bank(root, arm, identity)
    Xh, Bh, Yh, heldout_meta = prepare_heldout_bank(root, arm, identity)
    oracle = A19.exact_oracle_check(arm, arm.calib_prompts, n=8)
    if not oracle["pass"]:
        raise RuntimeError(f"exact oracle failed during curve: {oracle}")

    size_rows = []
    parity_raw = load_json(root / "parity" / f"{arm_name}.json")

    # Phase 1 freezes every fit before any expanded-size heldout diagnostic is
    # materialized.  This prevents adaptive changes after seeing the curve.
    for n_rows in ROW_SIZES:
        fit = load_fit(root, arm_name, n_rows, arm.device, identity)
        if fit is None:
            fit = A19.RidgeFootprint.fit(
                np.asarray(X[:n_rows]), np.asarray(Y[:n_rows]), arm.device,
                ridge_frac=RIDGE_FRAC)
            save_fit(root, arm_name, n_rows, fit, identity)
        del fit
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 2 evaluates the byte-frozen fits in ascending N order.
    for n_rows in ROW_SIZES:
        size_path = root / "curve" / arm_name / f"size_{n_rows}.json"
        if size_path.exists():
            row = load_json(size_path)
            if row.get("identity_sha256") != identity["identity_sha256"]:
                raise RuntimeError("size checkpoint identity mismatch")
            size_rows.append(row)
            continue

        fit = load_fit(root, arm_name, n_rows, arm.device, identity)
        if fit is None:
            raise RuntimeError(f"frozen fit missing for N={n_rows}")
        fidelity = fidelity_raw(
            fit, Xh, Bh, Yh, heldout_meta["spans"],
            int(arm.metadata["seed"]) + 1000 + n_rows, n_boot)
        if n_rows == 400:
            observed = fidelity["derived"]
            expected = parity_raw["heldout_footprint_fidelity"]
            guard = {
                "coverage_abs_delta": abs(observed["coverage"] - expected["coverage"]),
                "all_top1_agreement_exact": observed["all_top1_agreement"] == expected["all_top1_agreement"],
                "changed_top1_recovery_exact": observed["changed_top1_recovery"] == expected["changed_top1_recovery"],
                "native_flip_positions_exact": observed["native_flip_positions"] == expected["native_flip_positions"],
                "native_flip_recovered_exact": observed["native_flip_recovered"] == expected["native_flip_recovered"],
            }
            guard["pass"] = (
                guard["coverage_abs_delta"]
                <= PARITY_TOLERANCE["coverage_abs"]
                and all(value for key, value in guard.items()
                        if key.endswith("_exact"))
            )
            if not guard["pass"]:
                raise RuntimeError(f"N=400 refit drifted from Gate 0: {guard}")
            fidelity["gate0_refit_guard"] = guard
        row = {
            "schema": "steering-content-audit-e3-size-v1",
            "timestamp": utc_now(),
            "identity_sha256": identity["identity_sha256"],
            "n_rows": n_rows,
            "fit": fit.meta,
            "fidelity": fidelity,
        }
        atomic_write_json(size_path, row)
        size_rows.append(row)
        del fit
        gc.collect()
        torch.cuda.empty_cache()

    fit = load_fit(root, arm_name, MAX_ROWS, arm.device, identity)
    if fit is None:
        raise RuntimeError("largest fit checkpoint unexpectedly absent")
    frontier = largest_frontier(
        root, arm, fit, X, base, heldout_meta, identity, n_boot)
    result = {
        "schema": "steering-content-audit-e3-curve-v1",
        "timestamp": utc_now(),
        "arm": arm_name,
        "identity_sha256": identity["identity_sha256"],
        "environment": environment_record(),
        "parity_decisions": require_both_parity(root),
        "calibration_bank_manifest": calib_meta,
        "heldout_bank_manifest": heldout_meta,
        "oracle_guard": oracle,
        "sizes": sorted(size_rows, key=lambda x: x["n_rows"]),
        "frontier_at_largest": frontier,
    }
    atomic_write_json(final_path, result)
    log(f"curve complete arm={arm_name}")
    return final_path


def canary_science(seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(24, 8)).astype(np.float32)
    W = rng.normal(size=(8, 13)).astype(np.float32)
    Y = (X @ W + 0.01 * rng.normal(size=(24, 13))).astype(np.float32)
    fit = A19.RidgeFootprint.fit(X, Y, "cpu", ridge_frac=RIDGE_FRAC)
    pred = fit.predict(torch.from_numpy(X)).detach().numpy()
    return {
        "X_sha256": A19.sha_array(X),
        "Y_sha256": A19.sha_array(Y),
        "pred_sha256": A19.sha_array(pred.astype(np.float32)),
        "fit": fit.meta,
    }


def checkpoint_canary(root: Path, mode: str) -> Path:
    directory = root / "checkpoint_canary"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "steering-content-audit-e3-checkpoint-canary-v1",
        "seed": 17,
        "driver_sha256": sha256_file(Path(__file__)),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    identity["identity_sha256"] = canonical_sha(identity)
    identity_path = directory / "identity.json"
    if identity_path.exists() and load_json(identity_path) != identity:
        raise RuntimeError("canary identity mismatch")
    if not identity_path.exists():
        atomic_write_json(identity_path, identity)

    if mode == "reference":
        path = directory / "reference.json"
        atomic_write_json(path, canary_science())
        return path
    bank = directory / "bank.npy"
    bank_meta = directory / "bank.json"
    if not bank.exists():
        array = np.arange(96, dtype=np.float32).reshape(12, 8)
        atomic_save_npy(bank, array)
        atomic_write_json(bank_meta, {
            "identity_sha256": identity["identity_sha256"],
            "file_sha256": sha256_file(bank),
            "array_sha256": A19.sha_array(array),
        })
    meta = load_json(bank_meta)
    if meta["identity_sha256"] != identity["identity_sha256"]:
        raise RuntimeError("canary bank identity mismatch")
    if sha256_file(bank) != meta["file_sha256"]:
        raise RuntimeError("canary bank bytes mismatch")
    if mode == "interrupt":
        raise SystemExit(75)
    if mode != "resume":
        raise ValueError(mode)
    path = directory / "resumed.json"
    atomic_write_json(path, canary_science())
    reference = load_json(directory / "reference.json")
    resumed = load_json(path)
    receipt = {
        "schema": "steering-content-audit-e3-checkpoint-canary-receipt-v1",
        "pass": reference == resumed,
        "reference_sha256": sha256_file(directory / "reference.json"),
        "resumed_sha256": sha256_file(path),
        "interruption_exit_code": 75,
        "identity_sha256": identity["identity_sha256"],
        "timestamp": utc_now(),
    }
    if not receipt["pass"]:
        raise RuntimeError("interruption/resume canary mismatch")
    receipt_path = directory / "receipt.json"
    atomic_write_json(receipt_path, receipt)
    return receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True,
        choices=("parity", "curve", "checkpoint-canary", "environment",
                 "index-manifest"))
    parser.add_argument("--arm", choices=("fv", "taskvec"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument(
        "--canary-mode", choices=("reference", "interrupt", "resume"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir = args.outdir.resolve()
    args.outdir.mkdir(parents=True, exist_ok=True)
    configure_determinism()
    if args.stage == "index-manifest":
        payload = {
            "schema": "steering-content-audit-e3-index-manifest-v1",
            "timestamp": utc_now(),
            "arms": {
                arm: calibration_index_plan(arm)
                for arm in ("fv", "taskvec")
            },
        }
        payload["manifest_sha256"] = canonical_sha(payload["arms"])
        atomic_write_json(args.outdir / "calibration_index_manifest.json", payload)
        return
    if args.stage == "checkpoint-canary":
        if args.canary_mode is None:
            raise SystemExit("--canary-mode is required")
        path = checkpoint_canary(args.outdir, args.canary_mode)
        log(f"checkpoint canary wrote {path}")
        return
    if args.stage == "environment":
        install_pinned_loader()
        atomic_write_json(args.outdir / "environment.json", environment_record())
        return
    if args.arm is None:
        raise SystemExit("--arm is required for parity/curve")
    if args.n_boot != 10000:
        raise RuntimeError("decision-bearing E3 requires n_boot=10000")
    execution_device()
    install_pinned_loader()
    hydrate_model_snapshot()
    if args.stage == "parity":
        decision = run_parity(args.outdir, args.arm, args.n_boot)
        if not load_json(decision)["pass"]:
            raise SystemExit(42)
    elif args.stage == "curve":
        run_curve(args.outdir, args.arm, args.n_boot)


if __name__ == "__main__":
    main()
