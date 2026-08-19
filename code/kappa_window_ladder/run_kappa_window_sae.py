"""RunId 20260806-kappa-window-standardization: local SAE Phase 1.

Regenerates the frozen SAE cell at feature 12455 / coefficient 40 on the exact
prompt split. Gate 0 regenerates base, native, historical prefill+1, and control
before the corrected prefill-only condition is allowed to run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import battery as B  # noqa: E402
import run_sae as R  # noqa: E402
import sae_steer as S  # noqa: E402


RUN_ID = "20260806-kappa-window-standardization"
DEFAULT_OUT = (
    REPO / "runs/steering-content-audit" / RUN_ID / "phase1_sae"
)
SOURCE_RESULT = (
    REPO / "projects/steering-content-audit/paper/supplement/results"
    / "2026-07-07-sae-arm/results_full.json"
)
SOURCE_RESULT_SHA256 = (
    "4eabc2a30e3f0b095210866609267d47250886920f7d0895615ec86a32c7fab3"
)
PROMPTS_PATH = HERE / "prompts_neutral.json"
EXPECTED_COUNTS = {
    "baseline": 0,
    "E_native": 149,
    "E_first_prefill_plus1": 94,
    "control": 62,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(serialized)
    os.replace(tmp, path)


class RunLogger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        with self.path.open("a") as handle:
            handle.write(line + "\n")


def generate_condition(name, prompts, generate, log, method=None):
    texts = []
    hits = []
    activity_ok = True
    generated_forward_count = 0
    for index, prompt in enumerate(prompts):
        text = generate(prompt)
        texts.append(text)
        hits.append(int(B.wedding_topic_hit(text)))
        if method is not None:
            rows = [
                row for row in method.last_forward_activity
                if row["phase"] == "generated_forward"
            ]
            generated_forward_count += len(rows)
            activity_ok = activity_ok and all(not row["active"] for row in rows)
        if (index + 1) % 10 == 0 or index + 1 == len(prompts):
            log(f"{name}: {index + 1}/{len(prompts)}")
    return {
        "texts": texts,
        "hits": hits,
        "count": int(sum(hits)),
        "rate": float(np.mean(hits)),
        "corrected_generated_forwards_all_inactive": (
            activity_ok if method is not None else None
        ),
        "generated_forward_count": (
            generated_forward_count if method is not None else None
        ),
    }


def bootstrap_ratio_difference(
    plus1_hits, prefill_only_hits, native_hits, base_hits,
    n_boot=10000, seed=17,
):
    plus1 = np.asarray(plus1_hits, dtype=float)
    prefill_only = np.asarray(prefill_only_hits, dtype=float)
    native = np.asarray(native_hits, dtype=float)
    base = np.asarray(base_hits, dtype=float)
    denominator = native.mean() - base.mean()
    point = ((plus1.mean() - base.mean()) / denominator
             - (prefill_only.mean() - base.mean()) / denominator)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(base), size=len(base))
        den = native[idx].mean() - base[idx].mean()
        if abs(den) < 1e-9:
            continue
        values.append((plus1[idx].mean() - prefill_only[idx].mean()) / den)
    values = np.asarray(values)
    return {
        "point": float(point),
        "ci_lo": float(np.percentile(values, 2.5)),
        "ci_hi": float(np.percentile(values, 97.5)),
        "seed": seed,
        "n_boot_requested": n_boot,
        "n_boot_valid": int(values.size),
    }


def rate_record(hits, n_boot, seed=7):
    point, lo, hi = B.bootstrap_rate_ci(hits, n_boot=n_boot, seed=seed)
    return {"rate": point, "ci_lo": lo, "ci_hi": hi, "count": int(sum(hits))}


def gate_records(model, tokenizer, prompts, conditions, device, log):
    base_texts = conditions["baseline"]["texts"]
    base_rep = float(np.mean([
        B.three_gram_rep_rate(text, tokenizer) for text in base_texts
    ]))
    base_median_len = B.median_len_tokens(base_texts, tokenizer)
    base_nll = float(np.mean([
        B.mean_nll_under_model(model, tokenizer, prompt, text, device)
        for prompt, text in zip(prompts, base_texts)
    ]))
    gates = {}
    for name, condition in conditions.items():
        if name == "baseline":
            continue
        gate = B.degeneracy_gate(
            condition["texts"], prompts, base_rep, base_median_len, base_nll,
            model, tokenizer, device=device,
        )
        gates[name] = {
            "tripped": bool(gate.tripped),
            "rep": gate.rep_rate,
            "median_len": gate.median_len,
            "nll": gate.mean_nll,
            "reasons": gate.reasons,
        }
        log(f"gate {name}: tripped={gate.tripped} reasons={gate.reasons}")
    return {
        "baseline_refs": {
            "rep": base_rep, "median_len": base_median_len, "nll": base_nll,
        },
        "conditions": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=False)
    log = RunLogger(args.outdir / "run.log")
    started = time.time()
    log(f"start RunId={RUN_ID} Phase 1 SAE device={args.device}")

    observed_source_sha = sha256_file(SOURCE_RESULT)
    if observed_source_sha != SOURCE_RESULT_SHA256:
        raise RuntimeError(
            f"frozen SAE result hash mismatch: {observed_source_sha}")
    source = json.loads(SOURCE_RESULT.read_text())
    prompts_payload = json.loads(PROMPTS_PATH.read_text())
    prompts = prompts_payload["prompts"]
    assert len(prompts) == 200
    calib, eval_prompts = B.split_prompts(prompts, 50)
    assert len(calib) == 50 and len(eval_prompts) == 150

    torch.manual_seed(0)
    model, tokenizer = B.load_model("gpt2", device=args.device)
    sae, sae_meta = S.load_sae(R.SAE_RELEASE, R.SAE_ID, device=args.device)
    layer, site = S.sae_layer_from_hook(sae_meta["hook_name"])
    if (layer, site) != (7, "resid_pre"):
        raise RuntimeError(f"unexpected SAE site: {(layer, site)}")
    recon = S.reconstruction_cosine(
        sae, model, tokenizer, R.CONCEPT_TEXTS + R.NEUTRAL_TEXTS,
        layer, site, args.device,
    )
    if recon["mean_cosine_skip_pos0"] <= 0.9:
        raise RuntimeError(f"SAE reconstruction gate failed: {recon}")

    feature = 12455
    coeff = 40.0
    direction = sae.W_dec[feature].detach().cpu().float()
    method = S.SAESteerMethod(
        model, tokenizer, layer, direction, device=args.device,
        max_new_tokens=64, first_window="prefill_only",
    )
    log(
        f"loaded gpt2 + {R.SAE_RELEASE}/{R.SAE_ID}; "
        f"feature={feature} coeff={coeff} recon_skip0="
        f"{recon['mean_cosine_skip_pos0']:.12f}"
    )

    # Rebuild the frozen control exactly from the same calibration split.
    log("recover control mean delta and token set")
    mean_delta = S.position1_logit_delta(method, tokenizer, calib, coeff)
    token_ids = B.discover_token_set(mean_delta, coverage=0.90, cap=100)
    stored_token_ids = source["control_calibration"]["token_ids"]
    if token_ids != stored_token_ids:
        raise RuntimeError("SAE control token-set recovery mismatch")
    scalar = float(source["control_calibration"]["bias_scalar"])
    token_index = torch.tensor(token_ids, dtype=torch.long)
    bias = scalar * mean_delta[token_index]
    processor = B.LogitBiasProcessor(token_ids, bias)

    # Gate 0: no corrected-window generation is permitted before all four
    # historical conditions have been regenerated and checked.
    conditions = {}
    log("Gate 0 condition baseline")
    conditions["baseline"] = generate_condition(
        "baseline", eval_prompts,
        lambda prompt: B.base_generate(
            model, tokenizer, prompt, 64, args.device),
        log,
    )
    log("Gate 0 condition E_native")
    conditions["E_native"] = generate_condition(
        "E_native", eval_prompts,
        lambda prompt: method.generate(prompt, coeff, "native"), log,
    )
    log("Gate 0 condition E_first_prefill_plus1")
    conditions["E_first_prefill_plus1"] = generate_condition(
        "E_first_prefill_plus1", eval_prompts,
        lambda prompt: method.generate(
            prompt, coeff, "first_prefill_plus1"), log,
    )
    log("Gate 0 condition control")
    conditions["control"] = generate_condition(
        "control", eval_prompts,
        lambda prompt: B.control_generate(
            model, tokenizer, prompt, processor, 64, args.device),
        log,
    )

    observed_counts = {
        name: condition["count"] for name, condition in conditions.items()
    }
    sample_exact = {
        "baseline": conditions["baseline"]["texts"][:3]
        == source["samples"]["baseline"],
        "E_native": conditions["E_native"]["texts"][:3]
        == source["samples"]["E_native"],
        "E_first_prefill_plus1":
        conditions["E_first_prefill_plus1"]["texts"][:3]
        == source["samples"]["E_first"],
        "control": conditions["control"]["texts"][:3]
        == source["samples"]["control"],
    }
    gate0_pass = observed_counts == EXPECTED_COUNTS
    gate0 = {
        "pass": gate0_pass,
        "criterion": "exact banked aggregate rate/count reproduction",
        "expected_counts": EXPECTED_COUNTS,
        "observed_counts": observed_counts,
        "expected_rates": {
            name: count / 150 for name, count in EXPECTED_COUNTS.items()
        },
        "observed_rates": {
            name: condition["rate"] for name, condition in conditions.items()
        },
        "first_three_texts_exact_diagnostic_nonblocking": sample_exact,
    }
    log(f"Gate 0 pass={gate0_pass} counts={observed_counts}")

    base_payload = {
        "meta": {
            "run_id": RUN_ID,
            "phase": "Phase 1 SAE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": args.device,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "sae_lens": getattr(__import__("sae_lens"), "__version__", "unknown"),
            "model": "gpt2",
            "model_commit": getattr(model.config, "_commit_hash", None),
            "sae": sae_meta,
            "layer": layer,
            "site": site,
            "feature": feature,
            "coeff": coeff,
            "n_calib": len(calib),
            "n_eval": len(eval_prompts),
            "max_new_tokens": 64,
            "n_boot": args.n_boot,
        },
        "source": {
            "banked_result": str(SOURCE_RESULT),
            "banked_result_sha256": observed_source_sha,
            "prompts": str(PROMPTS_PATH),
            "prompts_sha256": sha256_file(PROMPTS_PATH),
        },
        "sae_reconstruction": recon,
        "control_recovery": {
            "token_ids_exact": token_ids == stored_token_ids,
            "token_ids": token_ids,
            "bias_scalar": scalar,
            "mean_delta_sha256": hashlib.sha256(
                np.ascontiguousarray(mean_delta.cpu().numpy()).tobytes()
            ).hexdigest(),
            "bias_sha256": hashlib.sha256(
                np.ascontiguousarray(bias.cpu().numpy()).tobytes()
            ).hexdigest(),
        },
        "gate0": gate0,
        "conditions": conditions,
        "spend_usd": 0.0,
    }

    if not gate0_pass:
        base_payload["status"] = "STOPPED_GATE0_FAIL"
        base_payload["runtime_sec"] = time.time() - started
        atomic_json(args.outdir / "phase1_sae_raw.json", base_payload)
        log("STOP Gate 0 failed; corrected window NOT_RUN")
        return 2

    # Gate 1 and corrected schedule are reached only after Gate 0 passes.
    sanity = {
        "prefill_only": S.kv_baked_first_sanity(
            method, tokenizer, calib[:2], coeff,
            first_window="prefill_only"),
        "prefill_plus1": S.kv_baked_first_sanity(
            method, tokenizer, calib[:2], coeff,
            first_window="prefill_plus1"),
    }
    log("condition E_first_prefill_only")
    conditions["E_first_prefill_only"] = generate_condition(
        "E_first_prefill_only", eval_prompts,
        lambda prompt: method.generate(
            prompt, coeff, "first_prefill_only"),
        log, method=method,
    )
    mutation_changed = sum(
        left != right for left, right in zip(
            conditions["E_first_prefill_plus1"]["texts"],
            conditions["E_first_prefill_only"]["texts"],
        )
    )
    gate1 = {
        "pass": bool(
            sanity["prefill_only"]["all_match"]
            and sanity["prefill_plus1"]["all_match"]
            and conditions["E_first_prefill_only"][
                "corrected_generated_forwards_all_inactive"]
            and mutation_changed > 0
        ),
        "sanity": sanity,
        "corrected_generated_forwards_all_inactive":
            conditions["E_first_prefill_only"][
                "corrected_generated_forwards_all_inactive"],
        "corrected_generated_forward_count":
            conditions["E_first_prefill_only"]["generated_forward_count"],
        "deliberate_prefill_plus1_mutation_changed_text_count": mutation_changed,
    }
    log(f"Gate 1 pass={gate1['pass']} changed_texts={mutation_changed}/150")

    rates = {
        name: rate_record(condition["hits"], args.n_boot)
        for name, condition in conditions.items()
    }
    kappa_plus1 = B.bootstrap_ratio_ci(
        conditions["E_first_prefill_plus1"]["hits"],
        conditions["E_native"]["hits"],
        conditions["baseline"]["hits"],
        n_boot=args.n_boot, seed=11,
    )
    kappa_prefill_only = B.bootstrap_ratio_ci(
        conditions["E_first_prefill_only"]["hits"],
        conditions["E_native"]["hits"],
        conditions["baseline"]["hits"],
        n_boot=args.n_boot, seed=11,
    )
    kappa_difference = bootstrap_ratio_difference(
        conditions["E_first_prefill_plus1"]["hits"],
        conditions["E_first_prefill_only"]["hits"],
        conditions["E_native"]["hits"],
        conditions["baseline"]["hits"],
        n_boot=args.n_boot, seed=17,
    )
    rho = B.bootstrap_ratio_ci(
        conditions["control"]["hits"],
        conditions["E_native"]["hits"],
        conditions["baseline"]["hits"],
        n_boot=args.n_boot, seed=13,
    )

    gates = gate_records(
        model, tokenizer, eval_prompts, conditions, args.device, log)
    base_payload.update({
        "status": "PHASE1_COMPLETE_GATE0_PASS",
        "gate1": gate1,
        "rates": rates,
        "kappa": {
            "prefill_plus1": {
                "point": kappa_plus1[0],
                "ci_lo": kappa_plus1[1], "ci_hi": kappa_plus1[2],
            },
            "prefill_only": {
                "point": kappa_prefill_only[0],
                "ci_lo": kappa_prefill_only[1], "ci_hi": kappa_prefill_only[2],
            },
            "prefill_plus1_minus_prefill_only": kappa_difference,
        },
        "rho_confirmation": {
            "point": rho[0], "ci_lo": rho[1], "ci_hi": rho[2],
            "banked_point": source["rho"]["point"],
            "unchanged_point": rho[0] == source["rho"]["point"],
        },
        "degeneracy_gates": gates,
        "runtime_sec": time.time() - started,
    })
    atomic_json(args.outdir / "phase1_sae_raw.json", base_payload)
    log(
        "complete: kappa prefill+1="
        f"{kappa_plus1[0]:.12f} prefill-only={kappa_prefill_only[0]:.12f} "
        f"difference={kappa_difference['point']:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
