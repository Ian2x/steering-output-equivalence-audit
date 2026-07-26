"""AxBench ReFT-r1 reproduce-first canary.

This is a method-breadth gate, not a rho/kappa audit cell. It uses the
released AxBench ReFT-r1 dictionary for Gemma-2-2B-IT:

    pyvene/gemma-reft-r1-2b-it-res, l20/weight.pt + l20/acts.json

and mirrors AxBench's steering-time addition semantics for LsReFT:

    h_l20 <- h_l20 + factor * max_act[concept_id] * weight[concept_id]

The default canary targets wedding concepts because the project already has a
neutral prompt set and a frozen wedding lexicon from the A0/ActAdd arms. The
gate only asks whether this released modern method family produces a clean,
measurable behavioral effect worth a later output-equivalence battery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

DEFAULT_REFT_DIR = (
    _REPO / "data/external/axbench_hf/gemma-reft-r1-2b-it-res")
DEFAULT_OUTDIR = (
    _REPO / "runs/steering-content-audit/2026-07-09-axbench-reft-wedding-canary")
DEFAULT_LEXICON_FILE = _HERE / "lexicon_wedding.json"

_STEMS = [
    "Tell me about your day.",
    "Write a short story about a walk in the park.",
    "Describe your favorite meal.",
    "What did you do last weekend?",
    "Explain how to make a cup of tea.",
    "Tell me about a book you enjoyed.",
    "Describe a city you would like to visit.",
    "What is your opinion on remote work?",
    "Write a few sentences about the weather today.",
    "Describe a hobby you find relaxing.",
    "Tell me about an interesting animal.",
    "What makes a good friend?",
    "Describe the view from a mountaintop.",
    "Write about a rainy afternoon.",
    "Explain why people enjoy music.",
    "Tell me about a memorable trip.",
    "Describe your ideal workspace.",
    "What do you think about learning new languages?",
    "Write a short note to a coworker.",
    "Describe a quiet morning routine.",
    "Tell me about a skill you would like to learn.",
    "What is your favorite season and why?",
    "Describe a garden in spring.",
    "Write about the sound of the ocean.",
    "Tell me about a hard problem you solved.",
]
_PREFIXES = [
    "", "Please ", "In a few sentences, ", "Briefly, ", "If you can, ",
    "For me, ", "Right now, ", "Honestly, ",
]
_SUFFIXES = [
    "", " Take your time.", " Keep it short.", " Be specific.",
    " Use plain language.", " Feel free to be creative.",
    " Add some detail.", " Just a couple of sentences.",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_neutral_prompts(n: int = 200, seed: int = 20260705) -> list[str]:
    combos: list[str] = []
    for stem in _STEMS:
        for pre in _PREFIXES:
            for suf in _SUFFIXES:
                if pre == "":
                    text = stem + suf
                else:
                    text = pre + stem[0].lower() + stem[1:] + suf
                combos.append(text)
    seen = set()
    uniq = []
    for c in combos:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(uniq))[:n]
    return [uniq[i] for i in idx]


def load_wedding_lexicon(path: Path = DEFAULT_LEXICON_FILE) -> list[str]:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    return list(obj["lexicon"])


WEDDING_LEXICON = load_wedding_lexicon()


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_reft_artifact(reft_dir: Path) -> dict:
    l20 = reft_dir / "l20"
    weight_path = l20 / "weight.pt"
    acts_path = l20 / "acts.json"
    meta_path = l20 / "metadata.jsonl"
    for p in (weight_path, acts_path, meta_path):
        if not p.exists():
            raise SystemExit(f"missing required ReFT artifact file: {p}")
    weights = torch.load(weight_path, map_location="cpu").float()
    with acts_path.open(encoding="utf-8") as f:
        acts = json.load(f)
    metadata = load_metadata(meta_path)
    if weights.ndim != 2:
        raise SystemExit(f"expected rank-2 ReFT weight, got {tuple(weights.shape)}")
    if len(metadata) != weights.shape[0]:
        raise SystemExit(
            f"metadata/weight concept count mismatch: {len(metadata)} vs {weights.shape[0]}")
    return {
        "weights": weights,
        "acts": acts,
        "metadata": metadata,
        "sha256": {
            "weight_pt": sha256_file(weight_path),
            "acts_json": sha256_file(acts_path),
            "metadata_jsonl": sha256_file(meta_path),
        },
    }


def wedding_hit(text: str) -> bool:
    low = text.lower()
    for w in WEDDING_LEXICON:
        if re.search(r"\b" + re.escape(w) + r"\b", low):
            return True
    return False


def hit_rate(texts: Sequence[str]) -> float:
    return float(np.mean([wedding_hit(t) for t in texts])) if texts else 0.0


def ngram_rep_rate(text: str, n: int = 3) -> float:
    toks = re.findall(r"\S+", text.lower())
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def generation_gate(texts: Sequence[str], base: dict | None = None) -> dict:
    lengths = [len(re.findall(r"\S+", t)) for t in texts]
    reps = [ngram_rep_rate(t) for t in texts]
    median_len = float(np.median(lengths)) if lengths else 0.0
    rep = float(np.mean(reps)) if reps else 0.0
    reasons: list[str] = []
    if base is not None:
        if rep > 2.0 * base["rep"] + 0.10:
            reasons.append("rep")
        if median_len < 0.5 * base["median_len"]:
            reasons.append("len")
    return {
        "rep": rep,
        "median_len": median_len,
        "degenerate": bool(reasons),
        "reasons": reasons,
    }


def render_chat(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


@contextmanager
def layer_add_hook(model, layer: int, add_vec: torch.Tensor | None):
    if add_vec is None:
        yield
        return
    module = model.model.layers[layer]

    def hook(_mod, _args, output):
        vec = add_vec
        if isinstance(output, tuple):
            hs = output[0]
            v = vec.to(device=hs.device, dtype=hs.dtype).view(1, 1, -1)
            return (hs + v,) + tuple(output[1:])
        v = vec.to(device=output.device, dtype=output.dtype).view(1, 1, -1)
        return output + v

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: Sequence[str], *,
                   add_vec: torch.Tensor | None,
                   layer: int,
                   batch_size: int,
                   max_new_tokens: int,
                   do_sample: bool,
                   temperature: float) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    out: list[str] = []
    chat_prompts = [render_chat(tokenizer, p) for p in prompts]
    for i in range(0, len(chat_prompts), batch_size):
        batch = chat_prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with layer_add_hook(model, layer, add_vec):
            ids = model.generate(**enc, **gen_kwargs)
        cont = ids[:, enc["input_ids"].shape[1]:]
        out.extend(tokenizer.batch_decode(cont, skip_special_tokens=True))
    return out


def score_condition(model, tokenizer, prompts: Sequence[str], *, layer: int,
                    add_vec: torch.Tensor | None, batch_size: int,
                    max_new_tokens: int, do_sample: bool,
                    temperature: float, base_gate: dict | None = None) -> dict:
    texts = generate_texts(
        model, tokenizer, prompts, add_vec=add_vec, layer=layer,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature)
    return {
        "rate": hit_rate(texts),
        "hits": [int(wedding_hit(t)) for t in texts],
        "texts": texts,
        "gate": generation_gate(texts, base=base_gate),
    }


def concept_record(metadata: Sequence[dict], concept_id: int, n_weight_rows: int | None = None) -> dict:
    if n_weight_rows is not None and (concept_id < 0 or concept_id >= n_weight_rows):
        raise SystemExit(f"concept_id {concept_id} out of range")
    by_id = {int(r.get("concept_id")): r for r in metadata}
    if concept_id not in by_id:
        raise SystemExit(f"concept_id {concept_id} missing from metadata")
    rec = by_id[concept_id]
    return {
        "concept_id": int(rec.get("concept_id", concept_id)),
        "concept": rec.get("concept"),
        "ref": rec.get("ref"),
    }


def write_report(outdir: Path, result: dict) -> None:
    ev = result["eval"]
    chosen = result["chosen"]
    lines = [
        "# AxBench/ReFT-r1 wedding canary",
        "",
        "This is a reproduce-first method-breadth canary, not a rho/kappa cell.",
        "",
        f"- Model: `{result['meta']['model']}`",
        f"- Artifact: `{result['meta']['artifact_repo']}`",
        f"- Layer: {result['meta']['layer']}",
        f"- Chosen concept: {chosen['concept_id']} — {chosen['concept']}",
        f"- Chosen factor: {chosen['factor']}",
        f"- Eval base topic rate: {ev['base_rate']:.3f}",
        f"- Eval steered topic rate: {ev['steered_rate']:.3f}",
        f"- Eval gain: {100 * ev['gain']:+.1f} pts",
        f"- Eval gate clean: `{not ev['steered_gate']['degenerate']}`",
        f"- Canary reproduced: `{result['verdict']['reproduced']}`",
        "",
        "Interpretation: if reproduced, run the matched-budget output-control "
        "battery before any master-table verdict. If not reproduced, do not "
        "manufacture a rho/kappa row.",
        "",
    ]
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--reft-dir", default=str(DEFAULT_REFT_DIR))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--concept-ids", default="1028,5593")
    ap.add_argument("--factors", default="0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,4.0,5.0")
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=80)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--repro-threshold", type=float, default=25.0)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reft_dir = Path(args.reft_dir)
    artifact = load_reft_artifact(reft_dir)
    weights: torch.Tensor = artifact["weights"]
    acts = artifact["acts"]
    metadata = artifact["metadata"]
    concept_ids = parse_ints(args.concept_ids)
    factors = parse_floats(args.factors)

    prompts = build_neutral_prompts(args.n_calib + args.n_eval, seed=args.seed)
    calib_prompts = prompts[:args.n_calib]
    eval_prompts = prompts[args.n_calib:args.n_calib + args.n_eval]
    concepts = [concept_record(metadata, c, weights.shape[0]) for c in concept_ids]

    dry = {
        "artifact": {
            "reft_dir": str(reft_dir),
            "weight_shape": list(weights.shape),
            "weight_dtype_loaded": str(weights.dtype),
            "sha256": artifact["sha256"],
            "acts_keys": sorted(acts.keys()),
        },
        "concepts": concepts,
        "factors": factors,
        "n_calib": len(calib_prompts),
        "n_eval": len(eval_prompts),
        "layer": args.layer,
    }
    (outdir / "dry_run.json").write_text(json.dumps(dry, indent=2), encoding="utf-8")
    log(f"loaded ReFT artifact: shape={tuple(weights.shape)} concepts={len(metadata)}")
    for c in concepts:
        log(f"concept {c['concept_id']}: {c['concept']}")
    if args.dry_run:
        log(f"dry run wrote {outdir / 'dry_run.json'}")
        return

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    log(f"loading model {args.model} on {args.device} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_calib = score_condition(
        model, tokenizer, calib_prompts, layer=args.layer, add_vec=None,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample, temperature=args.temperature)
    base_eval = score_condition(
        model, tokenizer, eval_prompts, layer=args.layer, add_vec=None,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample, temperature=args.temperature)
    log(f"base calib rate={base_calib['rate']:.3f}; eval rate={base_eval['rate']:.3f}")

    sweep = []
    thresholds = acts.get("thresholds", {})
    max_acts = acts.get("max_act", {})
    for cid in concept_ids:
        crec = concept_record(metadata, cid, weights.shape[0])
        max_act = float(max_acts.get(str(cid), thresholds.get(str(cid), 1.0)))
        direction = weights[cid].float()
        for factor in factors:
            add_vec = factor * max_act * direction
            rec = score_condition(
                model, tokenizer, calib_prompts, layer=args.layer, add_vec=add_vec,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample, temperature=args.temperature,
                base_gate=base_calib["gate"])
            gain = rec["rate"] - base_calib["rate"]
            row = {
                "concept_id": cid,
                "concept": crec["concept"],
                "factor": factor,
                "max_act": max_act,
                "calib_rate": rec["rate"],
                "calib_gain": gain,
                "gate": rec["gate"],
                "hits": rec["hits"],
                "texts": rec["texts"],
            }
            sweep.append(row)
            log(
                f"concept={cid} factor={factor:g} calib={rec['rate']:.3f} "
                f"gain={100 * gain:+.1f} clean={not rec['gate']['degenerate']}")

    clean = [r for r in sweep if not r["gate"]["degenerate"]]
    if not clean:
        chosen = max(sweep, key=lambda r: (r["calib_gain"], r["calib_rate"], -r["factor"]))
    else:
        chosen = max(clean, key=lambda r: (r["calib_gain"], r["calib_rate"], -r["factor"]))
    cid = int(chosen["concept_id"])
    max_act = float(chosen["max_act"])
    factor = float(chosen["factor"])
    add_vec = factor * max_act * weights[cid].float()
    log(f"chosen concept={cid} factor={factor:g}; scoring held-out eval")
    steered_eval = score_condition(
        model, tokenizer, eval_prompts, layer=args.layer, add_vec=add_vec,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample, temperature=args.temperature,
        base_gate=base_eval["gate"])
    eval_gain = steered_eval["rate"] - base_eval["rate"]
    clean_eval = not steered_eval["gate"]["degenerate"]
    reproduced = bool((100.0 * eval_gain >= args.repro_threshold) and clean_eval)

    result = {
        "verdict": {
            "class": "CANARY-REPRODUCED" if reproduced else "CANARY-NOT-REPRODUCED",
            "reproduced": reproduced,
            "threshold_pts": args.repro_threshold,
            "note": "Not a rho/kappa verdict; run output-equivalence controls only if reproduced.",
        },
        "meta": {
            "model": args.model,
            "artifact_repo": "pyvene/gemma-reft-r1-2b-it-res",
            "artifact_dir": str(reft_dir),
            "artifact_sha256": artifact["sha256"],
            "layer": args.layer,
            "device": str(model.device),
            "dtype": args.dtype,
            "concept_ids": concept_ids,
            "concepts": concepts,
            "factors": factors,
            "n_calib": len(calib_prompts),
            "n_eval": len(eval_prompts),
            "seed": args.seed,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "scoring": "wedding lexicon hit on full continuation",
        },
        "base": {
            "calib_rate": base_calib["rate"],
            "eval_rate": base_eval["rate"],
            "calib_gate": base_calib["gate"],
            "eval_gate": base_eval["gate"],
            "calib_hits": base_calib["hits"],
            "eval_hits": base_eval["hits"],
            "calib_texts": base_calib["texts"],
            "eval_texts": base_eval["texts"],
        },
        "calib": {
            "sweep": sweep,
            "clean_cells": sum(1 for r in sweep if not r["gate"]["degenerate"]),
        },
        "chosen": {
            "concept_id": cid,
            "concept": chosen["concept"],
            "factor": factor,
            "max_act": max_act,
            "calib_rate": chosen["calib_rate"],
            "calib_gain": chosen["calib_gain"],
            "calib_gate": chosen["gate"],
        },
        "eval": {
            "base_rate": base_eval["rate"],
            "steered_rate": steered_eval["rate"],
            "gain": eval_gain,
            "base_hits": base_eval["hits"],
            "steered_hits": steered_eval["hits"],
            "steered_texts": steered_eval["texts"],
            "steered_gate": steered_eval["gate"],
        },
    }
    (outdir / "results_canary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    write_report(outdir, result)
    log(
        f"eval base={base_eval['rate']:.3f} steered={steered_eval['rate']:.3f} "
        f"gain={100 * eval_gain:+.1f} pts clean={clean_eval} reproduced={reproduced}")


if __name__ == "__main__":
    main()
