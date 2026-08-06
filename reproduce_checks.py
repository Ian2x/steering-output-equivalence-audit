#!/usr/bin/env python3
"""Re-derive headline paper quantities from the shipped JSON artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TOL = 1e-9


def load(relative: str) -> dict:
    return json.loads((RESULTS / relative).read_text(encoding="utf-8"))


def close(actual: float, expected: float, label: str, tol: float = TOL) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def ratio(control: float, baseline: float, native: float) -> float:
    return (control - baseline) / (native - baseline)


def cascade(first: float, baseline: float, native: float) -> float:
    return (first - baseline) / (native - baseline)


def check_standard_row(
    name: str,
    relative: str,
    *,
    native_key: str,
    expected_rho: float,
    expected_kappa: float | None,
) -> dict:
    obj = load(relative)
    rates = obj["rates"]
    baseline = rates["baseline"]["rate"]
    native = rates[native_key]["rate"]
    control = rates["control"]["rate"]
    derived_rho = ratio(control, baseline, native)
    close(derived_rho, obj["rho"]["point"], f"{name} rho stored-vs-derived")
    close(derived_rho, expected_rho, f"{name} rho headline")

    derived_kappa = None
    if expected_kappa is not None:
        first = rates["E_first"]["rate"]
        derived_kappa = cascade(first, baseline, native)
        close(derived_kappa, obj["kappa"]["point"], f"{name} kappa stored-vs-derived")
        close(derived_kappa, expected_kappa, f"{name} kappa headline")

    return {
        "rho": derived_rho,
        "rho_ci": [obj["rho"]["ci_lo"], obj["rho"]["ci_hi"]],
        "kappa": derived_kappa,
        "kappa_ci": (
            [obj["kappa"]["ci_lo"], obj["kappa"]["ci_hi"]]
            if derived_kappa is not None
            else None
        ),
    }


def check_anchor() -> dict:
    obj = load("2026-07-06-a0-anchor-r2/results_full.json")
    rates = obj["rates"]
    baseline = rates["baseline"]["rate"]
    native = rates["E_all_steering"]["rate"]
    control = rates["control"]["rate"]
    derived = ratio(control, baseline, native)
    close(derived, 1.6153846153846152, "synthetic anchor rho")
    close(derived, obj["rho"]["point"], "synthetic anchor stored rho")
    close(obj["kappa"]["point"], 0.0, "synthetic anchor definitional kappa")
    if not obj["verdict"]["cell_valid"]:
        raise AssertionError("synthetic anchor cell is not valid")
    return {"rho": derived, "rho_ci": [obj["rho"]["ci_lo"], obj["rho"]["ci_hi"]]}


def check_function_vector() -> dict:
    obj = load("2026-07-06-a1-anchor/results_full.json")
    rates = obj["rates"]
    derived_rho = ratio(
        rates["control"]["rate"],
        rates["baseline"]["rate"],
        rates["E_all"]["rate"],
    )
    derived_kappa = cascade(
        rates["E_first"]["rate"],
        rates["baseline"]["rate"],
        rates["E_all"]["rate"],
    )
    close(derived_rho, -0.11864406779661019, "function-vector rho")
    close(derived_rho, obj["rho"]["point"], "function-vector stored rho")
    close(derived_kappa, 0.9830508474576272, "function-vector kappa")
    if not obj["degeneracy_gates"]["control"]["tripped"]:
        raise AssertionError("function-vector static control should remain void")
    return {
        "rho": derived_rho,
        "rho_ci": [obj["rho"]["ci_lo"], obj["rho"]["ci_hi"]],
        "kappa": derived_kappa,
        "static_control_void": True,
    }


def check_task_vector() -> dict:
    obj = load("2026-07-08-taskvec-7b/results_full.json")
    rates = obj["rates"]
    derived = ratio(
        rates["control"]["rate"],
        rates["baseline"]["rate"],
        rates["E_native_first"]["rate"],
    )
    close(derived, -0.1408450704225352, "task-vector static rho")
    close(derived, obj["rho"]["point"], "task-vector stored rho")
    close(obj["kappa"]["native"], 1.0, "task-vector definitional kappa")
    if not obj["degeneracy_gates"]["control"]["tripped"]:
        raise AssertionError("task-vector full-budget static control should remain void")
    return {
        "rho": derived,
        "rho_ci": [obj["rho"]["ci_lo"], obj["rho"]["ci_hi"]],
        "kappa": 1.0,
        "static_control_void": True,
    }


def check_refusal_plateau() -> dict:
    """Derive the refusal reproduction maximum under BOTH gate conventions.

    The refusal driver shipped an NLL-exempt degeneracy gate (repetition and
    length only) while every other arm counted the preregistered NLL rule. The
    manuscript reports the strict preregistered maximum 0.151 as primary and the
    NLL-exempt plateau maximum 0.259 as a documented, post-hoc, outcome-affecting
    secondary. Both are derived here under unambiguous names; neither is called
    "strict" unless the NLL rule is actually counted.
    """
    obj = load("2026-07-07-refusal-arm/dose_response_fine.json")
    adjudication = obj["adjudication"]
    baseline = obj["eval_baseline_refs"]

    # Re-apply the preregistered gate formula from the recorded per-scale
    # statistics rather than trusting the stored flags.
    rep_line = 2.0 * baseline["rep"] + 0.1
    len_line = 0.5 * baseline["median_len"]
    nll_line = 3.0 * baseline["nll"]
    for row in obj["grid"]:
        gate = row["gate"]
        rep_trip = gate["rep"] > rep_line
        len_trip = gate["median_len"] < len_line
        nll_trip = gate["nll"] > nll_line
        if rep_trip != gate["rep_trip"] or len_trip != gate["len_trip"]:
            raise AssertionError(
                f"refusal gate recomputation disagrees at frac {row['frac']}"
            )
        # nll_exempt == the convention the driver actually shipped.
        if (rep_trip or len_trip) != gate["degenerate"]:
            raise AssertionError(
                f"refusal NLL-exempt flag disagrees at frac {row['frac']}"
            )
        if (rep_trip or len_trip or nll_trip) != gate["raw_tripped"]:
            raise AssertionError(
                f"refusal strict-gate flag disagrees at frac {row['frac']}"
            )

    strict_rows = [row for row in obj["grid"] if not row["gate"]["raw_tripped"]]
    exempt_rows = [row for row in obj["grid"] if not row["gate"]["degenerate"]]

    strict_max = max(row["effect_over_native"] for row in strict_rows)
    exempt_max = max(row["effect_over_native"] for row in exempt_rows)

    # Manuscript Table 3 and Appendix H.
    close(strict_max, 0.1510791366906475, "refusal strict-gate maximum (0.151)")
    close(exempt_max, 0.2589928057553957, "refusal NLL-exempt plateau maximum (0.259)")
    close(exempt_max, adjudication["max_clean_effect_ratio"],
          "refusal NLL-exempt maximum vs shipped adjudication")
    if len(exempt_rows) != adjudication["n_clean_scales"]:
        raise AssertionError("refusal NLL-exempt clean-scale count does not match adjudication")
    if len(strict_rows) != 2:
        raise AssertionError(
            f"refusal strict-gate clean-scale count changed: {len(strict_rows)}"
        )
    # The strict convention is the manuscript's primary reading, and it does NOT
    # reach the >=3 clean-scale bar; the arm is reported as gate-sensitive.
    strict_passes_amendment2 = len(strict_rows) >= 3 and strict_max <= 0.3
    if strict_passes_amendment2:
        raise AssertionError(
            "refusal strict gate now clears Amendment 2; the manuscript's "
            "gate-sensitive reading would need revision"
        )
    if not adjudication["passes_amendment2"]:
        raise AssertionError("shipped NLL-exempt adjudication no longer passes Amendment 2")

    return {
        "gate_conventions": {
            "strict_preregistered_nll_counted": {
                "clean_fractions": [row["frac"] for row in strict_rows],
                "n_clean_scales": len(strict_rows),
                "max_effect_ratio": strict_max,
                "passes_amendment2": strict_passes_amendment2,
                "role": "manuscript primary; arm reported as gate-sensitive",
            },
            "nll_exempt_as_shipped": {
                "clean_fractions": [row["frac"] for row in exempt_rows],
                "n_clean_scales": len(exempt_rows),
                "max_effect_ratio": exempt_max,
                "passes_amendment2": adjudication["passes_amendment2"],
                "role": "documented post-hoc outcome-affecting secondary",
            },
        },
        "gate_lines": {"rep": rep_line, "median_len": len_line, "nll": nll_line},
    }


def check_refusal_cell_defects() -> dict:
    """Re-derive the refusal cell's ratios and confirm two driver defects.

    Both are disclosed in the manuscript and in KNOWN_ISSUES.md, and both are
    checkable from these bytes rather than taken on trust.

    1. Corrupt point fields. ``battery.bootstrap_ratio_ci`` clamps the
       denominator with ``max(den.mean() - base.mean(), 1e-9)``. The refusal
       effect is a suppression, so that denominator is negative and the stored
       ``rho.point`` / ``kappa.point`` are clamping artifacts of order -1e9.
       The bootstrap replicates do not use the clamp, so the intervals are
       correct. The manuscript recomputes both ratios from the recorded rates.
    2. Verdict-logic defect. ``run_refusal.py`` omits ``control_clean`` from its
       "Dissolved" branch, so it printed Dissolved while its own
       ``cell_valid`` was False and its control had degenerated into repetition
       loops. Amendment 2 forbids a degenerate control from certifying
       reproduction, so the corrected verdict for this cell is Mixed.
    """
    obj = load("2026-07-07-refusal-arm/results_full.json")
    rates = obj["rates"]
    base = rates["baseline"]["rate"]
    native = rates["E_native"]["rate"]
    first = rates["E_first"]["rate"]
    control = rates["control"]["rate"]

    rho_from_rates = (control - base) / (native - base)
    kappa_from_rates = (first - base) / (native - base)
    # Manuscript Table 3 footnote: kappa = (0.94-0.0267)/(0.94-0.0133) = 0.986.
    close(kappa_from_rates, 0.9856115107913668, "refusal kappa recomputed from rates")
    close(rho_from_rates, 0.9928057553956835, "refusal single-control rho from rates")

    # Defect 1: the stored point fields are the 1e-9-clamped artifact, not the
    # ratio above. Reproduce the clamp exactly so the claim is not hand-waved.
    clamped_rho = (control - base) / max(native - base, 1e-9)
    clamped_kappa = (first - base) / max(native - base, 1e-9)
    close(obj["rho"]["point"], clamped_rho, "refusal rho.point is the clamped artifact")
    close(obj["kappa"]["point"], clamped_kappa, "refusal kappa.point is the clamped artifact")
    if obj["rho"]["point"] > 0 or obj["kappa"]["point"] > 0:
        raise AssertionError("refusal point fields are no longer the known artifact")
    # The intervals bracket the rate-derived ratios, confirming the clamp did
    # not reach the bootstrap replicates.
    for label, value, block in (
        ("rho", rho_from_rates, obj["rho"]),
        ("kappa", kappa_from_rates, obj["kappa"]),
    ):
        if not block["ci_lo"] <= value <= block["ci_hi"]:
            raise AssertionError(
                f"refusal {label} interval no longer brackets the rate-derived value"
            )

    # Defect 2: the shipped verdict block records the contradiction directly.
    verdict = obj["verdict"]
    gates = obj["degeneracy_gates"]
    if verdict["class"] != "Dissolved":
        raise AssertionError("refusal shipped verdict class changed")
    if verdict["cell_valid"] or verdict["gate_clean_control"]:
        raise AssertionError("refusal verdict no longer records the invalid cell")
    if not gates["control"]["degenerate"] or not gates["control"]["rep_trip"]:
        raise AssertionError("refusal control is no longer flagged degenerate")
    close(gates["control"]["rep"], 0.9679331546463016, "refusal control repetition rate")

    return {
        "recomputed_from_rates": {"kappa": kappa_from_rates, "rho": rho_from_rates},
        "stored_point_fields_are_clamping_artifacts": {
            "kappa": obj["kappa"]["point"],
            "rho": obj["rho"]["point"],
        },
        "shipped_verdict": verdict["class"],
        "corrected_verdict": "Mixed",
        "control_degenerate_repetition_rate": gates["control"]["rep"],
        "note": (
            "Driver defects, both disclosed; see KNOWN_ISSUES.md. Intervals are "
            "unaffected and the decision rules read intervals. The refusal row "
            "the manuscript reports comes from the Amendment-2 dose ladder, not "
            "from this voided single-control cell."
        ),
    }


def check_distillation_anchor() -> dict:
    obj = load("2026-07-10-output-footprint-distill/synthetic.json")
    close(obj["rho"]["point"], 1.0, "full-vocabulary positive anchor rho")
    close(obj["rates"]["native"]["rate"], obj["rates"]["student"]["rate"], "anchor rates")
    if not obj["decision"]["pass"] or not obj["oracle_guard"]["pass"]:
        raise AssertionError("full-vocabulary positive anchor failed")
    return {
        "rho": obj["rho"]["point"],
        "native_rate": obj["rates"]["native"]["rate"],
        "student_rate": obj["rates"]["student"]["rate"],
        "oracle_guard": True,
    }


def check_frontier(name: str, filename: str, expected_fidelity: str) -> dict:
    obj = load(f"2026-07-10-output-footprint-distill/{filename}")
    clean = [row for row in obj["frontier"] if not row["gate"]["tripped"]]
    targets = [row["target_kl"] for row in clean]
    if targets != [0.03, 0.07, 0.11, 0.15, 0.18]:
        raise AssertionError(f"{name} clean frontier targets changed: {targets}")
    max_hi = max(row["rho"]["ci_hi"] for row in clean)
    if max_hi > 0.3 or not obj["decision"]["all_clean_rho_hi_le_0_3"]:
        raise AssertionError(f"{name} frontier exceeds the bounded 0.3 criterion")
    fidelity = obj["heldout_footprint_fidelity"]["class"]
    if fidelity != expected_fidelity:
        raise AssertionError(f"{name} fidelity class changed: {fidelity}")
    if not obj["oracle_guard"]["pass"]:
        raise AssertionError(f"{name} oracle guard failed")
    return {
        "clean_targets": targets,
        "max_clean_rho_hi": max_hi,
        "fidelity_class": fidelity,
        "decision": obj["decision"]["verdict"],
    }


def check_semantic() -> dict:
    obj = load("2026-07-10-caa-semantic-check/semantic_judged.json")
    expected = {
        "sycophancy": {
            "means": [59.85, 69.675, 77.65],
            "surface": [0.13, 0.19, 0.07],
            "difference": 7.975,
        },
        "corrigibility": {
            "means": [31.9, 45.4, 48.7],
            "surface": [0.04, 0.04, 0.06],
            "difference": 3.3,
        },
    }
    summary = {}
    for name, values in expected.items():
        behavior = obj["behaviors"][name]
        means = behavior["means"]
        observed_means = [
            means["base"]["alignment_clean"],
            means["native"]["alignment_clean"],
            means["control"]["alignment_clean"],
        ]
        observed_surface = [
            means["base"]["surface_only"],
            means["native"]["surface_only"],
            means["control"]["surface_only"],
        ]
        for index, (actual, target) in enumerate(zip(observed_means, values["means"])):
            close(actual, target, f"{name} semantic mean {index}")
        for index, (actual, target) in enumerate(zip(observed_surface, values["surface"])):
            close(actual, target, f"{name} surface-only rate {index}")
        close(
            behavior["control_minus_native_clean_alignment"]["point"],
            values["difference"],
            f"{name} control-native semantic difference",
        )
        summary[name] = {
            "clean_alignment_base_native_control": observed_means,
            "surface_only_base_native_control": observed_surface,
            "control_minus_native": behavior["control_minus_native_clean_alignment"],
            "semantic_rho_clean": behavior["semantic_rho_clean"],
            "verdict": behavior["verdict"],
        }
    return summary


def main() -> None:
    summary = {
        "synthetic_anchor": check_anchor(),
        "function_vector": check_function_vector(),
        "activation_addition": check_standard_row(
            "activation addition",
            "2026-07-06-actadd-arm/results_full.json",
            native_key="E_native",
            expected_rho=0.9586776859504132,
            expected_kappa=None,
        ),
        "refusal_ablation": check_refusal_plateau(),
        "refusal_cell_defects": check_refusal_cell_defects(),
        "task_vector": check_task_vector(),
        "sae": check_standard_row(
            "SAE",
            "2026-07-07-sae-arm/results_full.json",
            native_key="E_native",
            expected_rho=0.4161073825503356,
            expected_kappa=0.6308724832214766,
        ),
        "caa_sycophancy": check_standard_row(
            "CAA sycophancy",
            "2026-07-10-caa-semantic-check/source/sycophancy/results_full.json",
            native_key="E_native",
            expected_rho=0.8823529411764706,
            expected_kappa=0.8333333333333333,
        ),
        "caa_corrigibility": check_standard_row(
            "CAA corrigibility",
            "2026-07-10-caa-semantic-check/source/corrigibility/results_full.json",
            native_key="E_native",
            expected_rho=1.3333333333333333,
            expected_kappa=0.8333333333333331,
        ),
        "full_vocab_anchor": check_distillation_anchor(),
        "full_vocab_fv": check_frontier("FV", "fv.json", "nontrivial"),
        "full_vocab_taskvec": check_frontier("task vector", "taskvec.json", "poor"),
        "caa_semantic": check_semantic(),
    }
    print(json.dumps({"status": "PASS", "derived": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
