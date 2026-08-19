"""Focused non-model tests for the E3 calibration-size protocol."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import reduce_e3_calibration_curve as REDUCE
import run_e3_calibration_curve as E3
import run_output_footprint_distill as A19
from actlib import models as ACT_MODELS


def test_frozen_index_manifest_reconstructs_exactly() -> None:
    frozen = json.loads(
        (E3.REPO / "projects/steering-content-audit"
         / "e3_calibration_index_manifest_20260804.json").read_text()
    )
    arms = {arm: E3.calibration_index_plan(arm) for arm in ("fv", "taskvec")}
    assert E3.canonical_sha(arms) == frozen["manifest_sha256"]
    assert arms == frozen["arms"]
    assert arms["fv"]["construction_used_count"] == 1040
    assert arms["fv"]["eligible_extension_count"] == 1013
    assert arms["taskvec"]["construction_used_count"] == 880
    assert arms["taskvec"]["eligible_extension_count"] == 1167


@pytest.mark.parametrize("arm", ["fv", "taskvec"])
def test_parity_adjudication_accepts_shipped_bytes_and_rejects_drift(
        arm: str) -> None:
    shipped = E3.load_json(E3.SHIPPED_DIR / f"{arm}.json")
    assert E3.adjudicate_parity(copy.deepcopy(shipped), shipped)["pass"]

    drifted = copy.deepcopy(shipped)
    drifted["heldout_footprint_fidelity"]["coverage"] += (
        E3.PARITY_TOLERANCE["coverage_abs"] + 1e-6
    )
    decision = E3.adjudicate_parity(drifted, shipped)
    assert not decision["pass"]
    assert not next(
        row for row in decision["checks"]
        if row["name"] == "fidelity.coverage"
    )["pass"]


def test_fidelity_derivation_and_position_strata() -> None:
    raw = {
        "native_kl_rows": [2.0, 1.0, 3.0, 2.0],
        "residual_kl_rows": [1.0, 0.5, 0.0, 1.0],
        "base_top1": [0, 0, 0, 0],
        "native_top1": [1, 0, 2, 3],
        "pred_top1": [1, 0, 0, 3],
    }
    derived = E3.derive_fidelity_arrays(**raw)
    assert derived["coverage"] == pytest.approx(0.6875)
    assert derived["all_top1_agreement"] == pytest.approx(0.75)
    assert derived["native_flip_positions"] == 3
    assert derived["native_flip_recovered"] == 2
    assert derived["changed_top1_recovery"] == pytest.approx(2 / 3)

    strata = E3.position_strata(raw, [(0, 2), (2, 4)])
    assert [row["decode_position"] for row in strata] == [1, 2]
    assert [row["n_rows"] for row in strata] == [2, 2]
    assert strata[0]["changed_top1_recovery"] == pytest.approx(0.5)
    assert strata[1]["changed_top1_recovery"] == pytest.approx(1.0)


def test_prompt_cluster_bootstrap_is_deterministic() -> None:
    raw = {
        "native_kl_rows": [1.0, 2.0, 3.0, 4.0],
        "residual_kl_rows": [0.5, 1.0, 1.0, 2.0],
        "base_top1": [0, 0, 0, 0],
        "native_top1": [1, 0, 2, 3],
        "pred_top1": [1, 0, 0, 3],
    }
    first = E3.prompt_cluster_bootstrap(raw, [(0, 2), (2, 4)], 71, 100)
    second = E3.prompt_cluster_bootstrap(raw, [(0, 2), (2, 4)], 71, 100)
    assert first == second
    assert first["resampling_unit"] == "heldout prompt"


def test_independent_reducer_matches_frozen_bootstrap_definitions() -> None:
    control = [0, 1, 0, 1, 1, 0]
    native = [0, 1, 1, 1, 1, 0]
    base = [0, 0, 0, 1, 0, 0]
    assert REDUCE.rate_dict(control, 19, 1000) == A19.rate_dict(
        control, 19, 1000)
    assert REDUCE.rho_dict(control, native, base, 23, 1000) == A19.rho_dict(
        control, native, base, 23, 1000)

    raw = {
        "native_kl_rows": [2.0, 1.0],
        "residual_kl_rows": [0.5, 0.25],
        "base_top1": [0, 0],
        "native_top1": [1, 0],
        "pred_top1": [1, 0],
    }
    assert REDUCE.derive_fidelity(raw) == E3.derive_fidelity_arrays(**raw)


def test_atomic_checkpoint_canary_interrupt_resume(tmp_path: Path) -> None:
    reference = E3.checkpoint_canary(tmp_path, "reference")
    with pytest.raises(SystemExit) as interrupted:
        E3.checkpoint_canary(tmp_path, "interrupt")
    assert interrupted.value.code == 75
    receipt_path = E3.checkpoint_canary(tmp_path, "resume")
    receipt = E3.load_json(receipt_path)
    assert receipt["pass"]
    assert E3.load_json(reference) == E3.load_json(
        tmp_path / "checkpoint_canary" / "resumed.json"
    )


def test_pinned_loader_matches_original_call_trace_except_revision() -> None:
    class FakeTokenizer:
        pad_token = None
        eos_token = "<eos>"

    class FakeModel:
        config = type("Config", (), {"_commit_hash": E3.MODEL_REVISION})()

        def __init__(self, events: list) -> None:
            self.events = events

        def to(self, device: str) -> None:
            self.events.append(("to", device))

        def eval(self) -> None:
            self.events.append(("eval",))

    def trace(loader) -> list:
        events = []

        def tokenizer_load(model_id: str, **kwargs):
            events.append(("tokenizer", model_id, kwargs))
            return FakeTokenizer()

        def model_load(model_id: str, **kwargs):
            events.append(("model", model_id, kwargs))
            return FakeModel(events)

        with (
            mock.patch.object(torch, "manual_seed",
                              side_effect=lambda seed: events.append(
                                  ("manual_seed", seed))),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.backends.mps, "is_available",
                              return_value=True),
            mock.patch.object(AutoTokenizer, "from_pretrained",
                              side_effect=tokenizer_load),
            mock.patch.object(AutoModelForCausalLM, "from_pretrained",
                              side_effect=model_load),
        ):
            loader(E3.MODEL_ID, device="mps", dtype=torch.bfloat16, seed=19)
        return events

    original = ACT_MODELS.load_model
    original_trace = trace(original)
    try:
        E3.B.load_model = original
        E3.A19.B.load_model = original
        with mock.patch.object(torch.backends.mps, "is_available",
                               return_value=True):
            E3.install_pinned_loader()
        pinned_trace = trace(E3.B.load_model)
    finally:
        E3.B.load_model = original
        E3.A19.B.load_model = original

    revision = E3.MODEL_REVISION
    expected_pinned = copy.deepcopy(original_trace)
    expected_pinned[1][2]["revision"] = revision
    expected_pinned[2][2]["revision"] = revision
    assert pinned_trace == expected_pinned


def test_checkpoint_canary_fails_closed_on_byte_drift(tmp_path: Path) -> None:
    E3.checkpoint_canary(tmp_path, "reference")
    with pytest.raises(SystemExit):
        E3.checkpoint_canary(tmp_path, "interrupt")
    bank = tmp_path / "checkpoint_canary" / "bank.npy"
    with bank.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(RuntimeError, match="bank bytes mismatch"):
        E3.checkpoint_canary(tmp_path, "resume")


def test_row_ladder_never_cuts_a_prompt() -> None:
    assert all(n % E3.MAX_NEW_TOKENS == 0 for n in E3.ROW_SIZES)
    assert E3.MAX_ROWS // E3.MAX_NEW_TOKENS == 800
