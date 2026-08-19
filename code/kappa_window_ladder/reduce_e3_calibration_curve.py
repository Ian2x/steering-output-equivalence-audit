"""Independent raw-byte reducer for the E3 calibration-size curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROW_SIZES = (400, 800, 1600, 3200, 6400)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def derive_fidelity(raw: dict) -> dict:
    native_kl = np.asarray(raw["native_kl_rows"], dtype=np.float64)
    residual_kl = np.asarray(raw["residual_kl_rows"], dtype=np.float64)
    base = np.asarray(raw["base_top1"])
    native = np.asarray(raw["native_top1"])
    pred = np.asarray(raw["pred_top1"])
    flips = native != base
    recovered = (native == pred) & flips
    return {
        "coverage": 1.0 - float(residual_kl.sum())
        / max(float(native_kl.sum()), 1e-12),
        "mean_native_kl": float(native_kl.mean()),
        "mean_residual_kl": float(residual_kl.mean()),
        "all_top1_agreement": float(np.mean(native == pred)),
        "native_flip_positions": int(flips.sum()),
        "native_flip_recovered": int(recovered.sum()),
        "changed_top1_recovery": float(recovered.sum())
        / max(int(flips.sum()), 1),
    }


def rate_dict(hits: Sequence[int], seed: int, n_boot: int) -> dict:
    values = np.asarray(hits, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[
        rng.integers(0, len(values), size=(n_boot, len(values)))
    ].mean(axis=1)
    return {
        "rate": float(values.mean()),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "count": int(values.sum()),
        "n": len(values),
    }


def rho_dict(control_hits: Sequence[int], native_hits: Sequence[int],
             base_hits: Sequence[int], seed: int, n_boot: int) -> dict:
    control = np.asarray(control_hits, dtype=float)
    native = np.asarray(native_hits, dtype=float)
    base = np.asarray(base_hits, dtype=float)
    point = (control.mean() - base.mean()) / max(
        native.mean() - base.mean(), 1e-9)
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(control), size=len(control))
        numerator = control[draw].mean() - base[draw].mean()
        denominator = native[draw].mean() - base[draw].mean()
        if abs(denominator) >= 1e-9:
            ratios.append(numerator / denominator)
    if not ratios:
        return {"point": float(point), "ci_lo": None, "ci_hi": None}
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(ratios, 2.5)),
        "ci_hi": float(np.percentile(ratios, 97.5)),
    }


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return value


def rederive_fidelity(size: dict) -> dict:
    raw = size["fidelity"]["raw"]
    derived = derive_fidelity(raw)
    recorded = size["fidelity"]["derived"]
    for key, value in derived.items():
        observed = recorded[key]
        if isinstance(value, int):
            if observed != value:
                raise RuntimeError(f"fidelity count mismatch {key}")
        elif abs(float(observed) - float(value)) > 1e-12:
            raise RuntimeError(f"fidelity scalar mismatch {key}")
    return derived


def cluster_rows(raw: dict, spans: Sequence[Sequence[int]]) -> List[dict]:
    native = np.asarray(raw["native_top1"])
    base = np.asarray(raw["base_top1"])
    pred = np.asarray(raw["pred_top1"])
    rows = []
    for start, stop in spans:
        sl = slice(int(start), int(stop))
        flips = native[sl] != base[sl]
        rows.append({
            "flips": int(flips.sum()),
            "recovered": int(np.sum((native[sl] == pred[sl]) & flips)),
        })
    return rows


def paired_recovery_difference(a: dict, b: dict,
                               spans: Sequence[Sequence[int]],
                               seed: int, n_boot: int = 10000) -> dict:
    ca = cluster_rows(a["fidelity"]["raw"], spans)
    cb = cluster_rows(b["fidelity"]["raw"], spans)
    if len(ca) != len(cb):
        raise RuntimeError("paired cluster count mismatch")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(ca), size=len(ca))
        values = []
        for clusters in (ca, cb):
            flips = sum(clusters[int(i)]["flips"] for i in draw)
            recovered = sum(clusters[int(i)]["recovered"] for i in draw)
            values.append(recovered / max(flips, 1))
        diffs.append(values[1] - values[0])
    da = rederive_fidelity(a)["changed_top1_recovery"]
    db = rederive_fidelity(b)["changed_top1_recovery"]
    return {
        "point": db - da,
        "ci_lo": float(np.quantile(diffs, 0.025)),
        "ci_hi": float(np.quantile(diffs, 0.975)),
        "seed": seed,
        "n_boot": n_boot,
        "resampling_unit": "heldout prompt",
    }


def rederive_frontier(row: dict, base_hits: Sequence[int],
                      native_hits: Sequence[int], seed: int,
                      n_boot: int) -> dict:
    hits = row["hits"]
    rate = rate_dict(hits, seed + 101, n_boot)
    rho = rho_dict(hits, native_hits, base_hits, seed + 102, n_boot)
    if rate != row["rate"] or rho != row["rho"]:
        raise RuntimeError(f"frontier rate/rho mismatch at KL={row['target_kl']}")
    raw = row["gate"]["raw"]
    refs = row["gate"]["refs"]
    rep = float(np.mean(raw["rep_per_text"]))
    med = float(np.median(raw["length_per_text"]))
    nll = float(np.mean(raw["nll_per_text"]))
    reasons = []
    if rep > 2 * refs["rep"] + 0.1:
        reasons.append("rep")
    if med < 0.5 * refs["median_len"]:
        reasons.append("length")
    if nll > 3 * refs["nll"]:
        reasons.append("nll")
    gate = {
        "tripped": bool(reasons), "rep": rep, "median_len": med,
        "nll": nll, "reason_codes": reasons,
    }
    recorded = row["gate"]["derived"]
    for key in ("tripped", "reason_codes"):
        if gate[key] != recorded[key]:
            raise RuntimeError(f"frontier gate mismatch {key}")
    for key in ("rep", "median_len", "nll"):
        if abs(gate[key] - recorded[key]) > 1e-12:
            raise RuntimeError(f"frontier gate scalar mismatch {key}")
    return {"target_kl": row["target_kl"], "rate": rate,
            "rho": rho, "gate": gate}


def reduce_arm(root: Path, arm: str) -> dict:
    curve_path = root / "curve" / arm / "curve.json"
    curve = load(curve_path)
    if curve["arm"] != arm:
        raise RuntimeError("arm mismatch")
    sizes = {int(row["n_rows"]): row for row in curve["sizes"]}
    if tuple(sorted(sizes)) != ROW_SIZES:
        raise RuntimeError("size ladder mismatch")
    rederived = {n: rederive_fidelity(row) for n, row in sizes.items()}
    spans = curve["heldout_bank_manifest"]["spans"]
    d_400_6400 = paired_recovery_difference(
        sizes[400], sizes[6400], spans,
        (20260706 if arm == "fv" else 20260707) + 6400400)
    d_3200_6400 = paired_recovery_difference(
        sizes[3200], sizes[6400], spans,
        (20260706 if arm == "fv" else 20260707) + 6403200)

    seed = 20260706 if arm == "fv" else 20260707
    base_hits = curve["heldout_bank_manifest"]["base_hits"]
    native_hits = curve["heldout_bank_manifest"]["native_hits"]
    frontier = [
        rederive_frontier(row, base_hits, native_hits, seed, 10000)
        for row in curve["frontier_at_largest"]
    ]
    all_five_clean = len(frontier) == 5 and all(
        not row["gate"]["tripped"] for row in frontier)
    low_rho = all_five_clean and all(
        row["rho"]["ci_hi"] is not None and row["rho"]["ci_hi"] <= 0.3
        for row in frontier)
    distillable = any(
        not row["gate"]["tripped"]
        and row["rho"]["ci_lo"] is not None
        and row["rho"]["ci_lo"] >= 0.9
        for row in frontier)
    largest = rederived[6400]
    material = (
        d_400_6400["point"] >= 0.10 and d_400_6400["ci_lo"] > 0.0)
    plateau = (
        d_3200_6400["point"] < 0.05
        and d_3200_6400["ci_hi"] < 0.10)
    strong_joint = (
        largest["coverage"] >= 0.80
        and largest["changed_top1_recovery"] >= 0.80)
    return {
        "arm": arm,
        "curve_sha256": sha256_file(curve_path),
        "fidelity": {str(n): row for n, row in rederived.items()},
        "paired_recovery_delta_400_to_6400": d_400_6400,
        "paired_recovery_delta_3200_to_6400": d_3200_6400,
        "material_recovery_gain": material,
        "plateau_at_largest_step": plateau,
        "strong_joint_fidelity": strong_joint,
        "frontier_6400": frontier,
        "all_five_frontier_cells_clean": all_five_clean,
        "largest_frontier_low_rho": low_rho,
        "largest_frontier_distillable": distillable,
    }


def outcome_map(arms: Dict[str, dict]) -> dict:
    values = list(arms.values())
    both_below_half_plateau = all(
        row["fidelity"]["6400"]["changed_top1_recovery"] < 0.5
        and row["plateau_at_largest_step"] for row in values)
    high_fidelity = [
        row for row in values
        if row["material_recovery_gain"] and row["strong_joint_fidelity"]
    ]
    if both_below_half_plateau:
        return {
            "row": "recovery_plateaus_below_0.5_for_both",
            "paper_implication": (
                "Strengthens only a data-saturation reading for the fixed "
                "off-policy linear ridge controller; does not establish an "
                "intrinsically target-limited interface."),
        }
    if high_fidelity and all(row["largest_frontier_low_rho"]
                             for row in high_fidelity):
        return {
            "row": "high_fidelity_and_rho_stays_low",
            "arms": [row["arm"] for row in high_fidelity],
        }
    if high_fidelity and any(row["largest_frontier_distillable"]
                             for row in high_fidelity):
        return {
            "row": "high_fidelity_and_rho_rises_to_distillable",
            "arms": [row["arm"] for row in high_fidelity
                     if row["largest_frontier_distillable"]],
        }
    return {
        "row": "unmapped_intermediate",
        "paper_implication": (
            "No director-table row fires cleanly; report the curve and do not "
            "self-adjudicate a stronger manuscript change."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.rundir = args.rundir.resolve()
    arms = {arm: reduce_arm(args.rundir, arm) for arm in ("fv", "taskvec")}
    result = {
        "schema": "steering-content-audit-e3-independent-reduction-v1",
        "arms": arms,
        "outcome_map": outcome_map(arms),
    }
    atomic_write_json(args.out.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
