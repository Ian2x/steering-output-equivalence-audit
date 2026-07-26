"""Amendment 19b guard-only algebraic oracle supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import battery as B
import run_output_footprint_distill as D


REPO = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    REPO / "runs/steering-content-audit/2026-07-10-output-footprint-distill"
    / "synthetic.json"
)
DEFAULT_OUT = DEFAULT_SOURCE.with_name("synthetic_oracle_supplement.json")


def verify(source: Path, device: str) -> dict:
    artifact = json.loads(source.read_text())
    assert artifact["meta"]["arm"] == "synthetic"
    model, tokenizer = B.load_model(
        artifact["meta"]["model"], device=device, dtype=torch.float32)

    prompts = B.build_neutral_prompts(200)[:200]
    calib_prompts, _ = B.split_prompts(prompts, artifact["meta"]["n_calib"])
    teacher = artifact["teacher"]
    rng = np.random.default_rng(artifact["meta"]["seed"])
    w = rng.normal(size=model.config.n_embd).astype(np.float32)
    w /= np.linalg.norm(w) + 1e-12
    v = np.zeros(model.config.vocab_size, dtype=np.float32)
    v[teacher["token_ids"]] = 1.0
    v -= v.mean()
    wt = torch.from_numpy(w).to(device)
    vt = torch.from_numpy(v).to(device)

    def teacher_delta(phi: torch.Tensor) -> torch.Tensor:
        z = ((torch.dot(phi.float(), wt) - teacher["q_mean"])
             / teacher["q_std"])
        return teacher["chosen_scalar"] * (1.0 + 0.05 * z) * vt

    rows = []
    for prompt in calib_prompts[:8]:
        full = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        recon_ids = []
        native_ids = []
        max_error = 0.0
        for _ in range(artifact["meta"]["max_new_tokens"]):
            base_logits, resid = D.forward_with_final_resid(model, full)
            base = base_logits[-1].float()
            native = base + teacher_delta(resid[-1].float())
            delta = native - base
            recon = base + delta
            max_error = max(
                max_error, float((recon - native).abs().max().item()))
            native_token = int(native.argmax())
            recon_token = int(recon.argmax())
            native_ids.append(native_token)
            recon_ids.append(recon_token)
            if recon_token == tokenizer.eos_token_id:
                break
            full = torch.cat([
                full, torch.tensor([[recon_token]], device=device)
            ], dim=1)
        rows.append({
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "max_logit_error": max_error,
            "token_match": recon_ids == native_ids,
            "text_match": (
                D.decode_ids(tokenizer, recon_ids).strip()
                == D.decode_ids(tokenizer, native_ids).strip()),
        })

    max_error = max(row["max_logit_error"] for row in rows)
    passed = (
        max_error <= 1e-5
        and all(row["token_match"] and row["text_match"] for row in rows)
    )
    return {
        "meta": {
            "amendment": "19b",
            "purpose": "guard-only synthetic algebraic oracle supplement",
            "source": str(source.relative_to(REPO)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "device": device,
            "n": len(rows),
        },
        "rows": rows,
        "max_logit_error": max_error,
        "all_token_match": all(row["token_match"] for row in rows),
        "all_text_match": all(row["text_match"] for row in rows),
        "pass": passed,
        "interpretation": (
            "Harness-only. Success does not upgrade anchor difficulty or "
            "teacher/student footprint fidelity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    result = verify(args.source, args.device)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload)
    print(args.out)
    print(hashlib.sha256(payload.encode()).hexdigest())
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
