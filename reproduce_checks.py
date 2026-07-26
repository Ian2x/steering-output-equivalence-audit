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
    """Re-derive the refusal plateau under BOTH gate conventions.

    The shipped `degenerate` flag is the implemented gate, which exempts the
    NLL rule (repetition and length only). `raw_tripped` is the preregistered
    strict gate, which counts NLL. The paper reports the strict maximum 0.151
    as primary and the NLL-exempt plateau 0.259 as a disclosed, post-hoc,
    outcome-affecting secondary; both are recomputed here from stored rows.
    """
    obj = load("2026-07-07-refusal-arm/dose_response_fine.json")
    adjudication = obj["adjudication"]
    exempt_rows = [row for row in obj["grid"] if not row["gate"]["degenerate"]]
    if len(exempt_rows) != adjudication["n_clean_scales"]:
        raise AssertionError("refusal clean-scale count does not match adjudication")
    exempt_max = max(row["effect_over_native"] for row in exempt_rows)
    close(exempt_max, adjudication["max_clean_effect_ratio"], "refusal plateau maximum")
    if len(exempt_rows) < 3 or exempt_max > 0.3 or not adjudication["passes_amendment2"]:
        raise AssertionError("refusal NLL-exempt coherent-window adjudication failed")

    strict_rows = [row for row in obj["grid"] if not row["gate"]["raw_tripped"]]
    strict_max = max(row["effect_over_native"] for row in strict_rows)
    close(strict_max, 0.15107913669064738, "refusal strict-gate maximum")
    if len(strict_rows) >= 3:
        raise AssertionError(
            "strict gate should leave fewer than the three clean scales "
            "Amendment 2 requires; the paper's gate-sensitive reading depends on this"
        )
    return {
        "nll_exempt_clean_fractions": [row["frac"] for row in exempt_rows],
        "nll_exempt_max_effect_ratio": exempt_max,
        "nll_exempt_plateau_pass": True,
        "strict_clean_fractions": [row["frac"] for row in strict_rows],
        "strict_max_effect_ratio": strict_max,
        "strict_clean_scale_count": len(strict_rows),
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
