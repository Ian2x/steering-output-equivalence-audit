#!/usr/bin/env python3
"""Independent raw-byte adjudication for E3 Stage A Gate 0.

This reducer deliberately does not import the E3 execution driver. It rebuilds
the frozen exact and tolerance-bearing comparisons from the two raw parity JSON
files and the shipped Amendment-19 JSON bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TOLERANCE = {
    "coverage_abs": 0.005,
    "frontier_achieved_kl_rel": 0.02,
    "oracle_max_logit_error": 1e-5,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact(name: str, observed, expected) -> dict:
    return {
        "name": name,
        "mode": "exact",
        "observed": observed,
        "expected": expected,
        "pass": observed == expected,
    }


def absolute(name: str, observed, expected, tolerance: float) -> dict:
    delta = abs(float(observed) - float(expected))
    return {
        "name": name,
        "mode": "absolute_tolerance",
        "observed": observed,
        "expected": expected,
        "abs_delta": delta,
        "tolerance": tolerance,
        "pass": delta <= tolerance,
    }


def relative(name: str, observed, expected, tolerance: float) -> dict:
    delta = abs(float(observed) - float(expected)) / max(
        abs(float(expected)), 1e-12)
    return {
        "name": name,
        "mode": "relative_tolerance",
        "observed": observed,
        "expected": expected,
        "rel_delta": delta,
        "tolerance": tolerance,
        "pass": delta <= tolerance,
    }


def reduce_arm(raw_path: Path, shipped_path: Path) -> dict:
    observed = load_json(raw_path)
    expected = load_json(shipped_path)
    checks = [
        exact("fit.n_rows", observed["fit"]["n_rows"],
              expected["fit"]["n_rows"]),
        exact("fit.feature_dim", observed["fit"]["feature_dim"],
              expected["fit"]["feature_dim"]),
        exact("fit.vocab_size", observed["fit"]["vocab_size"],
              expected["fit"]["vocab_size"]),
        exact("fit.effective_rank", observed["fit"]["effective_rank"],
              expected["fit"]["effective_rank"]),
        exact("frozen_count_guard.pass",
              observed["frozen_count_guard"]["pass"], True),
        exact("rates.base.count", observed["rates"]["base"]["count"],
              expected["rates"]["base"]["count"]),
        exact("rates.native.count", observed["rates"]["native"]["count"],
              expected["rates"]["native"]["count"]),
        exact("base_hits", observed["base_hits"], expected["base_hits"]),
        exact("native_hits", observed["native_hits"], expected["native_hits"]),
        exact("n_frontier", len(observed["frontier"]),
              len(expected["frontier"])),
        exact("fidelity.all_top1_agreement",
              observed["heldout_footprint_fidelity"]["all_top1_agreement"],
              expected["heldout_footprint_fidelity"]["all_top1_agreement"]),
        exact("fidelity.changed_top1_recovery",
              observed["heldout_footprint_fidelity"]["changed_top1_recovery"],
              expected["heldout_footprint_fidelity"]["changed_top1_recovery"]),
        exact("fidelity.native_flip_positions",
              observed["heldout_footprint_fidelity"]["native_flip_positions"],
              expected["heldout_footprint_fidelity"]["native_flip_positions"]),
        exact("fidelity.native_flip_recovered",
              observed["heldout_footprint_fidelity"]["native_flip_recovered"],
              expected["heldout_footprint_fidelity"]["native_flip_recovered"]),
        absolute("fidelity.coverage",
                 observed["heldout_footprint_fidelity"]["coverage"],
                 expected["heldout_footprint_fidelity"]["coverage"],
                 TOLERANCE["coverage_abs"]),
        exact("oracle_guard.pass", observed["oracle_guard"]["pass"], True),
    ]
    oracle_error = observed["oracle_guard"]["max_logit_error"]
    checks.append({
        "name": "oracle_guard.max_logit_error",
        "mode": "upper_bound",
        "observed": oracle_error,
        "expected": None,
        "tolerance": TOLERANCE["oracle_max_logit_error"],
        "pass": oracle_error <= TOLERANCE["oracle_max_logit_error"],
    })

    for index, (row, old) in enumerate(
            zip(observed["frontier"], expected["frontier"])):
        prefix = f"frontier[{index}]"
        checks.extend([
            exact(f"{prefix}.target_kl", float(row["target_kl"]),
                  float(old["target_kl"])),
            relative(f"{prefix}.achieved_kl", row["achieved_kl"],
                     old["achieved_kl"],
                     TOLERANCE["frontier_achieved_kl_rel"]),
            exact(f"{prefix}.hits", row["hits"], old["hits"]),
            exact(f"{prefix}.rate", row["rate"], old["rate"]),
            exact(f"{prefix}.rho", row["rho"], old["rho"]),
            exact(f"{prefix}.gate.tripped", row["gate"]["tripped"],
                  old["gate"]["tripped"]),
        ])

    tolerance_checks = [
        row for row in checks if row["mode"] != "exact"
    ]
    exact_checks = [row for row in checks if row["mode"] == "exact"]
    if all(row["pass"] for row in checks):
        outcome = "PASS"
    elif all(row["pass"] for row in tolerance_checks):
        outcome = "EXACT_ONLY_MISS"
    else:
        outcome = "BROAD_FAILURE"
    return {
        "raw_path": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "shipped_path": str(shipped_path),
        "shipped_sha256": sha256_file(shipped_path),
        "n_checks": len(checks),
        "n_exact_checks": len(exact_checks),
        "n_tolerance_checks": len(tolerance_checks),
        "all_exact_pass": all(row["pass"] for row in exact_checks),
        "all_tolerances_pass": all(row["pass"] for row in tolerance_checks),
        "pass": all(row["pass"] for row in checks),
        "outcome": outcome,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--shipped-dir", required=True, type=Path)
    args = parser.parse_args()

    arms = {
        arm: reduce_arm(
            args.outdir / "parity" / f"{arm}.json",
            args.shipped_dir / f"{arm}.json",
        )
        for arm in ("fv", "taskvec")
    }
    outcomes = {row["outcome"] for row in arms.values()}
    if outcomes == {"PASS"}:
        overall = "PASS"
    elif "BROAD_FAILURE" in outcomes:
        overall = "BROAD_FAILURE"
    else:
        overall = "EXACT_ONLY_MISS"
    payload = {
        "schema": "steering-content-audit-e3-stage-a-independent-reduction-v1",
        "tolerance": TOLERANCE,
        "overall_outcome": overall,
        "arms": arms,
    }
    target = args.outdir / "independent_stage_a_reduction.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(target)
    print(json.dumps({
        "overall_outcome": overall,
        "arms": {
            arm: {
                "outcome": row["outcome"],
                "n_checks": row["n_checks"],
                "failed": [
                    check["name"] for check in row["checks"]
                    if not check["pass"]
                ],
            }
            for arm, row in arms.items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
