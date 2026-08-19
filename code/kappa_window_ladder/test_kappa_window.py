from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import caa_steer as C
import replay_caa_semantic_check as Replay
import sae_steer as S


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, _prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[0]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return ",".join(str(int(value)) for value in ids)


class _Model(torch.nn.Module):
    """One-block deterministic model whose next token exposes hook activity."""

    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(
            h=torch.nn.ModuleList([torch.nn.Identity()]))

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        del past_key_values, use_cache
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        hidden = self.transformer.h[0](hidden)
        token_ids = hidden[..., 0].round().to(torch.long).remainder(4)
        logits = torch.full((*token_ids.shape, 4), -10.0)
        logits.scatter_(-1, token_ids.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits, past_key_values=(None,))


def _generated_activity(method):
    return [
        row["active"] for row in method.last_forward_activity
        if row["phase"] == "generated_forward"
    ]


def _exercise(method, sanity):
    selected_text = method.generate("prompt", 1.0, "first")
    assert _generated_activity(method) == [False, False, False]

    prefill_only_text = method.generate(
        "prompt", 1.0, "first_prefill_only")
    assert _generated_activity(method) == [False, False, False]

    prefill_plus1_text = method.generate(
        "prompt", 1.0, "first_prefill_plus1")
    assert _generated_activity(method) == [True, False, False]

    assert selected_text == prefill_only_text
    assert prefill_only_text != prefill_plus1_text
    assert sanity(
        method, method.tokenizer, ["prompt"], 1.0,
        first_window="prefill_only")["all_match"]
    assert sanity(
        method, method.tokenizer, ["prompt"], 1.0,
        first_window="prefill_plus1")["all_match"]


def test_caa_first_window_flag_and_sanity():
    tok = _Tokenizer()
    method = C.CAAMethod(
        _Model(), tok, layer=0, direction=torch.ones(1),
        first_window="prefill_only",
        max_new_tokens=4)
    _exercise(method, C.kv_baked_first_sanity)


def test_sae_first_window_flag_and_sanity():
    tok = _Tokenizer()
    method = S.SAESteerMethod(
        _Model(), tok, layer=0, direction=torch.ones(1),
        first_window="prefill_only",
        max_new_tokens=4)
    _exercise(method, S.kv_baked_first_sanity)


def test_first_window_is_required_and_declared_at_every_active_call_site():
    tok = _Tokenizer()
    for cls in (C.CAAMethod, S.SAESteerMethod):
        try:
            cls(_Model(), tok, layer=0, direction=torch.ones(1))
        except TypeError as exc:
            assert "first_window" in str(exc)
        else:
            raise AssertionError(f"{cls.__name__} accepted an implicit schedule")

    exp_dir = Path(__file__).resolve().parent
    missing = []
    for path in sorted(exp_dir.glob("*.py")):
        if path.name.startswith("._"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else None)
            if name not in {"CAAMethod", "SAESteerMethod"}:
                continue
            if not any(keyword.arg == "first_window"
                       for keyword in node.keywords):
                missing.append(f"{path.name}:{node.lineno}:{name}")
    assert missing == [], f"implicit first-window call sites: {missing}"


def test_advisor_precision_labels_are_fail_closed_at_the_boundaries():
    assert Replay.precision_label({"ci_lo": 0.001, "ci_hi": 0.12}) \
        == "DIRECTIONAL_PREFILL_PLUS1_HIGHER"
    assert Replay.precision_label({"ci_lo": -0.12, "ci_hi": -0.001}) \
        == "DIRECTIONAL_PREFILL_ONLY_HIGHER"
    assert Replay.precision_label({"ci_lo": -0.059, "ci_hi": 0.059}) \
        == "AGREES"
    assert Replay.precision_label({"ci_lo": -0.06, "ci_hi": 0.02}) \
        == "INCONCLUSIVE_UNDERPOWERED"
    assert Replay.precision_label({"ci_lo": -0.02, "ci_hi": 0.06}) \
        == "INCONCLUSIVE_UNDERPOWERED"


def test_checkpoint_digest_covers_nontext_gate_fields():
    record = Replay.condition_record(["I agree with you."])
    Replay.validate_checkpoint_condition(record, 1)
    record["rate"] = 0.0
    try:
        Replay.validate_checkpoint_condition(record, 1)
    except RuntimeError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("condition metadata tampering was not detected")


def test_full_study_agreement_includes_accepted_sae_raw_bytes():
    sae = Replay.accepted_sae_precision(Replay.DEFAULT_PHASE1_RAW)
    assert sae["raw_sha256"] == Replay.PHASE1_RAW_SHA256
    assert sae["gate0_pass"] and sae["gate1_pass"]
    assert sae["classification"] == "DIRECTIONAL_PREFILL_PLUS1_HIGHER"
    assert sae["delta"]["ci_lo"] > 0
    assert sae["generation_validity_pass"]


def test_production_process_interruption_resume_canary_is_byte_identical(tmp_path):
    checkpoint_path = tmp_path / "forced-interruption.json"
    command = [
        sys.executable, str(Path(Replay.__file__).resolve()),
        "--bundle-sha256", "0" * 64,
        "--out", str(checkpoint_path),
        "--checkpoint-canary",
    ]
    interrupted = subprocess.run(
        [*command, "interrupt"], text=True, capture_output=True)
    assert interrupted.returncode == 86
    assert "CHECKPOINT_CANARY_INTENTIONAL_INTERRUPT" in interrupted.stdout
    resumed = subprocess.run(
        [*command, "resume"], text=True, capture_output=True)
    assert resumed.returncode == 0, resumed.stderr
    assert "CHECKPOINT_CANARY_PASS" in resumed.stdout


def test_resume_rejects_corrected_output_before_gate0():
    control_recovery = {"fixture": True}
    Replay.seal_record(control_recovery, "control_recovery_sha256")
    condition = Replay.condition_record(["I agree with you."])
    cell = {
        "status": "RUNNING_GATE0",
        "meta": {"n_eval": 1},
        "control_recovery": control_recovery,
        "conditions": {
            "baseline": condition,
            "E_native": condition,
            "control": condition,
            "E_first_prefill_plus1": condition,
            "E_first_prefill_only": condition,
        },
    }
    Replay.seal_cell_state(cell)
    try:
        Replay.validate_cell_checkpoint_state(cell)
    except RuntimeError as exc:
        assert "corrected condition precedes passing Gate 0" in str(exc)
    else:
        raise AssertionError("premature corrected condition was accepted")


def test_resume_rejects_skipped_historical_condition():
    cell = {
        "status": "RUNNING_GATE0",
        "meta": {"n_eval": 1},
        "conditions": {"E_native": Replay.condition_record(["I agree."])},
    }
    Replay.seal_cell_state(cell)
    try:
        Replay.validate_cell_checkpoint_state(cell)
    except RuntimeError as exc:
        assert "legal prefix" in str(exc)
    else:
        raise AssertionError("skipped baseline was accepted")


def test_resume_rejects_tampered_generation_gate_record():
    condition = Replay.condition_record(["I agree with you."])
    control_recovery = {"fixture": True}
    gate0 = {"pass": True}
    gate1 = {"pass": True}
    metrics = {"fixture": True}
    baseline_refs = {"rep": 0.0, "median_len": 1.0, "nll": 1.0}
    gate = {"tripped": False, "rep": 0.0, "median_len": 1.0,
            "nll": 1.0, "reasons": []}
    Replay.seal_record(control_recovery, "control_recovery_sha256")
    Replay.seal_record(gate0, "gate0_sha256")
    Replay.seal_record(gate1, "gate1_sha256")
    Replay.seal_record(metrics, "metrics_sha256")
    Replay.seal_record(baseline_refs, "baseline_refs_sha256")
    Replay.seal_record(gate, "generation_gate_sha256")
    cell = {
        "status": "RUNNING_GATE0",
        "meta": {"n_eval": 1},
        "control_recovery": control_recovery,
        "conditions": {
            "baseline": condition, "E_native": condition,
            "control": condition, "E_first_prefill_plus1": condition,
            "E_first_prefill_only": condition,
        },
        "gate0": gate0,
        "gate1": gate1,
        "metrics": metrics,
        "degeneracy_gates": {
            "baseline_refs": baseline_refs,
            "conditions": {"E_native": gate},
        },
    }
    Replay.seal_cell_state(cell)
    cell["degeneracy_gates"]["baseline_refs"]["nll"] = 999.0
    try:
        Replay.validate_cell_checkpoint_state(cell)
    except RuntimeError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered generation-gate state was accepted")


def test_resume_rejects_gate_without_baseline_and_nonprefix_gate_order():
    condition = Replay.condition_record(["I agree with you."])
    control_recovery = {"fixture": True}
    gate0 = {"pass": True}
    gate1 = {"pass": True}
    metrics = {"fixture": True}
    baseline_refs = {"rep": 0.0, "median_len": 1.0, "nll": 1.0}
    gate = {"tripped": False, "rep": 0.0, "median_len": 1.0,
            "nll": 1.0, "reasons": []}
    Replay.seal_record(control_recovery, "control_recovery_sha256")
    Replay.seal_record(gate0, "gate0_sha256")
    Replay.seal_record(gate1, "gate1_sha256")
    Replay.seal_record(metrics, "metrics_sha256")
    Replay.seal_record(baseline_refs, "baseline_refs_sha256")
    Replay.seal_record(gate, "generation_gate_sha256")
    cell = {
        "status": "RUNNING_GATE0",
        "meta": {"n_eval": 1},
        "control_recovery": control_recovery,
        "conditions": {
            "baseline": condition, "E_native": condition,
            "control": condition, "E_first_prefill_plus1": condition,
            "E_first_prefill_only": condition,
        },
        "gate0": gate0,
        "gate1": gate1,
        "metrics": metrics,
        "degeneracy_gates": {
            "baseline_refs": None,
            "conditions": {"control": gate},
        },
    }
    Replay.seal_cell_state(cell)
    try:
        Replay.validate_cell_checkpoint_state(cell)
    except RuntimeError as exc:
        assert "precedes baseline refs" in str(exc)
    else:
        raise AssertionError("gate without baseline references was accepted")

    cell["degeneracy_gates"]["baseline_refs"] = baseline_refs
    Replay.seal_cell_state(cell)
    try:
        Replay.validate_cell_checkpoint_state(cell)
    except RuntimeError as exc:
        assert "not a legal prefix" in str(exc)
    else:
        raise AssertionError("non-prefix generation gate was accepted")


def test_resume_rederives_terminal_validity_and_claim_invariants():
    condition = Replay.condition_record(["I agree with you."])
    control_recovery = {"fixture": True}
    gate0 = {"pass": True}
    gate1 = {"pass": True}
    metrics = {"fixture": True}
    baseline_refs = {"rep": 0.0, "median_len": 1.0, "nll": 1.0}
    Replay.seal_record(control_recovery, "control_recovery_sha256")
    Replay.seal_record(gate0, "gate0_sha256")
    Replay.seal_record(gate1, "gate1_sha256")
    Replay.seal_record(metrics, "metrics_sha256")
    Replay.seal_record(baseline_refs, "baseline_refs_sha256")
    gates = {}
    for name in Replay.GENERATION_GATE_CONDITIONS:
        gate = {"tripped": False, "rep": 0.0, "median_len": 1.0,
                "nll": 1.0, "reasons": []}
        Replay.seal_record(gate, "generation_gate_sha256")
        gates[name] = gate
    validity = {
        "pass": True,
        "required_for_standardization_claim": True,
        "tripped_conditions": {},
    }
    claim = {
        "pass": True,
        "gate0_pass": True,
        "gate1_pass": True,
        "generation_validity_pass": True,
    }
    Replay.seal_record(validity, "generation_validity_sha256")
    Replay.seal_record(claim, "claim_eligibility_sha256")
    cell = {
        "status": "COMPLETE",
        "meta": {"n_eval": 1},
        "control_recovery": control_recovery,
        "conditions": {
            "baseline": condition, "E_native": condition,
            "control": condition, "E_first_prefill_plus1": condition,
            "E_first_prefill_only": condition,
        },
        "gate0": gate0,
        "gate1": gate1,
        "metrics": metrics,
        "degeneracy_gates": {
            "baseline_refs": baseline_refs, "conditions": gates},
        "generation_validity": validity,
        "claim_eligibility": claim,
        "rows": [{}],
    }
    Replay.seal_cell_state(cell)
    Replay.validate_cell_checkpoint_state(cell)

    contradictory_validity = dict(validity)
    contradictory_validity["pass"] = False
    Replay.seal_record(
        contradictory_validity, "generation_validity_sha256")
    cell["generation_validity"] = contradictory_validity
    Replay.seal_cell_state(cell)
    try:
        Replay.validate_cell_checkpoint_state(cell)
    except RuntimeError as exc:
        assert "validity contradicts" in str(exc)
    else:
        raise AssertionError("contradictory terminal validity was accepted")

    cell["generation_validity"] = validity
    contradictory_claim = dict(claim)
    contradictory_claim["pass"] = False
    Replay.seal_record(contradictory_claim, "claim_eligibility_sha256")
    cell["claim_eligibility"] = contradictory_claim
    Replay.seal_cell_state(cell)
    try:
        Replay.validate_cell_checkpoint_state(cell)
    except RuntimeError as exc:
        assert "claim eligibility contradicts" in str(exc)
    else:
        raise AssertionError("contradictory terminal claim was accepted")


def test_generation_validity_is_required_for_global_claims():
    labels = {
        "sae_feature_steering": "AGREES",
        "caa_sycophancy": "AGREES",
        "caa_corrigibility": "AGREES",
    }
    standardization_allowed = Replay.standardization_claim_allowed(
        gate0_all_pass=True,
        gate1_all_pass=True,
        generation_validity_all_pass=False,
    )
    global_agreement_allowed = Replay.global_agreement_allowed(
        labels, standardization_allowed)
    assert not standardization_allowed
    assert not global_agreement_allowed


def test_generation_gate_progress_is_condition_granular():
    cell = {
        "status": "RUNNING_GATE0",
        "meta": {"n_eval": 1},
        "conditions": {
            "baseline": Replay.condition_record(["I disagree."]),
            "E_native": Replay.condition_record(["I agree with you."]),
            "control": Replay.condition_record(["I agree with you."]),
        },
        "degeneracy_gates": {
            "baseline_refs": {"rep": 0.0, "median_len": 1.0, "nll": 1.0},
            "conditions": {
                "E_native": {"tripped": False},
            },
        },
    }
    progress = Replay.checkpoint_progress({"sycophancy": cell})["sycophancy"]
    assert progress["completed_generation_gate_conditions"] == ["E_native"]
    assert not progress["generation_gates_complete"]
    cell["degeneracy_gates"]["conditions"]["control"] = {"tripped": False}
    progress = Replay.checkpoint_progress({"sycophancy": cell})["sycophancy"]
    assert progress["generation_gates_complete"]
