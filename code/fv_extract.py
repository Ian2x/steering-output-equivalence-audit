"""Stage-1 function-vector extraction (A1 anchor), mirroring Todd et al.
2310.15213 (github.com/ericwtodd/function_vectors), adapted to actlib / Pythia.

Pipeline
--------
1. `build_word_pairs` / `make_icl_prompt` — the repo's exact ICL template
   (prefixes {"input":"Q:", "output":"A:"}, separators {"input":"\n",
   "output":"\n\n"}, prepend_space=True). 10-shot by default; zero-shot for
   evaluation ("Q: {x}\nA:").
2. `mean_head_activations` — for each (layer, head) average the head's slice of
   the attention `dense` INPUT (= per-head z, the o_proj input, exactly Todd's
   `get_mean_head_activations`) at the final prompt token over >=100 clean ICL
   prompts.
3. `compute_indirect_effect` — CIE head ranking: on shuffled-label ICL prompts,
   patch each head's task-mean into the `dense` input at the last token and
   measure recovery of the correct answer's first-token logprob vs the
   shuffled baseline. Search may be restricted to a layer band (logged).
4. `build_function_vector` — FV = sum over top-`k` CIE heads of
   `dense(one_hot_head_slice(mean_activation))` (Todd's
   `compute_universal_function_vector`; the pythia `dense` Linear applies its
   bias once per head, matching the reference).

Head slicing convention (GPT-NeoX / Pythia)
-------------------------------------------
`attention.dense` is `nn.Linear(hidden, hidden)` whose INPUT is the concatenated
per-head attention outputs (head h occupies columns
`h*head_dim : (h+1)*head_dim`). We capture that input with a forward-pre-hook on
`dense` (actlib's `attn_out` site is the dense OUTPUT, which is post-projection
and cannot be head-sliced), matching the reference's `retain_input=True` trace.
"""

from __future__ import annotations

import os
import sys
from contextlib import ExitStack, contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


# ---------------------------------------------------------------------------
# Model-config helpers (GPT-NeoX / Pythia)
# ---------------------------------------------------------------------------

def neox_config(model) -> Dict:
    cfg = model.config
    return {
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "resid_dim": cfg.hidden_size,
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "name_or_path": cfg.name_or_path,
    }


def dense_module(model, layer: int):
    """The attention output projection (o_proj) for a GPT-NeoX layer."""
    return model.gpt_neox.layers[layer].attention.dense


# ---------------------------------------------------------------------------
# ICL prompt construction (Todd template)
# ---------------------------------------------------------------------------

PREFIXES = {"input": "Q:", "output": "A:", "instructions": ""}
SEPARATORS = {"input": "\n", "output": "\n\n", "instructions": ""}


def make_icl_prompt(examples: Sequence[Tuple[str, str]], query: str,
                    shuffle_outputs: Optional[Sequence[str]] = None) -> str:
    """Build an ICL prompt string. ``examples`` = [(x, y), ...] demonstrations;
    ``query`` = the held-out input (its answer is NOT appended). prepend_space=True
    (a leading space on every input/output), matching the reference.

    If ``shuffle_outputs`` is given (list same length as examples), those replace
    the demonstration outputs (shuffled-label prompt for CIE).
    """
    s = ""
    for i, (x, y) in enumerate(examples):
        out = y if shuffle_outputs is None else shuffle_outputs[i]
        s += f"{PREFIXES['input']} {x}{SEPARATORS['input']}"
        s += f"{PREFIXES['output']} {out}{SEPARATORS['output']}"
    s += f"{PREFIXES['input']} {query}{SEPARATORS['input']}"
    s += f"{PREFIXES['output']}"
    return s


def zero_shot_prompt(query: str) -> str:
    """Zero-shot eval prompt: 'Q: {x}\nA:' (prepend_space on the input)."""
    return f"{PREFIXES['input']} {query}{SEPARATORS['input']}{PREFIXES['output']}"


def answer_with_space(answer: str) -> str:
    """The gold continuation as the model sees it (prepend_space=True): ' answer'.
    Its FIRST token id is the CIE target."""
    return " " + answer


# ---------------------------------------------------------------------------
# Sampling ICL prompts from a dataset
# ---------------------------------------------------------------------------

def load_pairs(path: str) -> List[Tuple[str, str]]:
    import json
    data = json.load(open(path))
    return [(d["input"], d["output"]) for d in data]


def split_pairs(pairs: Sequence[Tuple[str, str]], n_eval: int, seed: int
                ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Deterministic (train-pool, eval) split. eval = last ``n_eval`` after a
    fixed-seed permutation; train-pool = the rest (used to draw ICL shots)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    eval_idx = idx[:n_eval]
    train_idx = idx[n_eval:]
    ev = [pairs[i] for i in eval_idx]
    tr = [pairs[i] for i in train_idx]
    return tr, ev


def sample_icl_prompts(train_pool: Sequence[Tuple[str, str]], n_prompts: int,
                       n_shots: int, seed: int, shuffle_labels: bool = False):
    """Draw ``n_prompts`` ICL prompts. Each: ``n_shots`` demonstrations + one
    query, all sampled without replacement from ``train_pool`` per prompt.

    Returns list of dicts: {prompt, query, answer, examples, shuffle_outputs}.
    """
    rng = np.random.default_rng(seed)
    pool = list(train_pool)
    out = []
    for _ in range(n_prompts):
        pick = rng.choice(len(pool), size=n_shots + 1, replace=False)
        shots = [pool[i] for i in pick[:n_shots]]
        qx, qy = pool[pick[n_shots]]
        shuffle_outputs = None
        if shuffle_labels:
            outs = [y for (_, y) in shots]
            perm = rng.permutation(len(outs))
            shuffle_outputs = [outs[i] for i in perm]
        prompt = make_icl_prompt(shots, qx, shuffle_outputs=shuffle_outputs)
        out.append({"prompt": prompt, "query": qx, "answer": qy,
                    "examples": shots, "shuffle_outputs": shuffle_outputs})
    return out


# ---------------------------------------------------------------------------
# Mean head activations (dense input, head-sliced, last token)
# ---------------------------------------------------------------------------

@contextmanager
def _capture_dense_inputs(model, cfg, store: Dict[int, torch.Tensor]):
    """Forward-pre-hook every layer's `dense` to grab its INPUT (the o_proj input
    = concatenated per-head z). Stores the LAST-token row per layer: [hidden]."""
    handles = []

    def mk(layer):
        def pre_hook(mod, args):
            x = args[0]  # [batch, seq, hidden]
            store[layer] = x[:, -1, :].detach().to("cpu")  # [batch, hidden]
            return None
        return pre_hook

    for L in range(cfg["n_layers"]):
        handles.append(dense_module(model, L).register_forward_pre_hook(mk(L)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def mean_head_activations(model, tokenizer, icl_prompts, cfg, device="cpu",
                          batch_size: int = 8, log=print) -> torch.Tensor:
    """Average per-head dense-input activation at the final prompt token over
    ``icl_prompts`` (list of dicts with 'prompt'). Returns
    ``[n_layers, n_heads, head_dim]`` on CPU (float32).

    Because prompts differ in length we run them one-per-forward (batch=1) via a
    left-pad batching that keeps the last real token aligned; simplest correct
    path is per-prompt forward, which we do in mini-batches with padding + mask
    handled by taking the last non-pad token. Here prompts vary in length, so we
    forward each prompt individually (batch handled by loop) — cheap: 1 forward
    per prompt, last-token only.
    """
    n_layers, n_heads, head_dim = cfg["n_layers"], cfg["n_heads"], cfg["head_dim"]
    accum = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float32)
    count = 0
    prompts = [p["prompt"] for p in icl_prompts]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        # left-pad so the final column is the last real token for every row
        old_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        enc = tokenizer(batch, return_tensors="pt", padding=True)
        tokenizer.padding_side = old_side
        enc = {k: v.to(device) for k, v in enc.items()}
        store: Dict[int, torch.Tensor] = {}
        with torch.no_grad(), _capture_dense_inputs(model, cfg, store):
            model(**enc)
        for L in range(n_layers):
            x = store[L].float()  # [batch, hidden]
            x = x.view(x.shape[0], n_heads, head_dim)  # [batch, heads, head_dim]
            accum[L] += x.sum(0)
        count += len(batch)
        if log and (i // batch_size) % 5 == 0:
            log(f"    mean-head: {count}/{len(prompts)} prompts")
    return accum / max(count, 1)


# ---------------------------------------------------------------------------
# CIE head ranking (shuffled-label prompts, per-head patch -> logprob recovery)
# ---------------------------------------------------------------------------

@contextmanager
def _patch_one_head(model, cfg, layer: int, head: int, mean_vec: torch.Tensor,
                    device: str):
    """Replace, at the LAST token, the ``head`` slice of layer ``layer``'s dense
    INPUT with ``mean_vec`` ([head_dim]), then let dense recompute normally.

    Realised as a forward-pre-hook on `dense` that edits its input tensor.
    """
    head_dim = cfg["head_dim"]
    lo, hi = head * head_dim, (head + 1) * head_dim
    mv = mean_vec.to(device)

    def pre_hook(mod, args):
        x = args[0]
        x = x.clone()
        x[:, -1, lo:hi] = mv.to(x.dtype)
        return (x,) + tuple(args[1:])

    h = dense_module(model, layer).register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        h.remove()


def _first_answer_token_id(tokenizer, answer: str) -> int:
    ids = tokenizer.encode(answer_with_space(answer), add_special_tokens=False)
    return ids[0]


def compute_indirect_effect(model, tokenizer, shuffled_prompts, mean_acts, cfg,
                            layers: Optional[Sequence[int]] = None,
                            device="cpu", log=print) -> np.ndarray:
    """CIE per head, averaged over ``shuffled_prompts``.

    For each shuffled-label prompt we compute the baseline logprob of the correct
    answer's first token. Then for each (layer, head) in the search set we patch
    that head's task-mean at the last token and re-measure the answer's first-token
    logprob. CIE(L,H) = mean over prompts of (patched_logprob - baseline_logprob).

    Returns ``[n_layers, n_heads]`` (nan for heads outside the search band).
    """
    n_layers, n_heads = cfg["n_layers"], cfg["n_heads"]
    if layers is None:
        layers = list(range(n_layers))
    layers = list(layers)
    cie = np.full((n_layers, n_heads), np.nan, dtype=np.float64)
    # accumulate per (L,H)
    accum = {(L, H): 0.0 for L in layers for H in range(n_heads)}
    n_used = 0
    for pi, sp in enumerate(shuffled_prompts):
        tgt = _first_answer_token_id(tokenizer, sp["answer"])
        enc = tokenizer(sp["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(**enc).logits[0, -1]  # [vocab]
        base_lp = torch.log_softmax(base_logits.float(), dim=-1)[tgt].item()
        for L in layers:
            for H in range(n_heads):
                mv = mean_acts[L, H]  # [head_dim]
                with torch.no_grad(), _patch_one_head(model, cfg, L, H, mv, device):
                    logits = model(**enc).logits[0, -1]
                lp = torch.log_softmax(logits.float(), dim=-1)[tgt].item()
                accum[(L, H)] += (lp - base_lp)
        n_used += 1
        if log and (pi + 1) % 8 == 0:
            log(f"    CIE: {pi+1}/{len(shuffled_prompts)} prompts")
    for (L, H), s in accum.items():
        cie[L, H] = s / max(n_used, 1)
    return cie


def top_heads(cie: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    """Top-``k`` (layer, head, score) by CIE (nan-safe)."""
    flat = []
    for L in range(cie.shape[0]):
        for H in range(cie.shape[1]):
            v = cie[L, H]
            if not np.isnan(v):
                flat.append((L, H, float(v)))
    flat.sort(key=lambda t: t[2], reverse=True)
    return flat[:k]


# ---------------------------------------------------------------------------
# FV construction (Todd compute_universal_function_vector, pythia branch)
# ---------------------------------------------------------------------------

def build_function_vector(model, mean_acts, top, cfg, device="cpu") -> torch.Tensor:
    """FV = sum over ``top`` heads of dense(one_hot_head_slice(mean)).

    Mirrors the reference gpt-neox branch: for each (L,H) build a zero
    resid-dim vector with the head's mean placed in its slice, pass through the
    layer's `dense` (Linear, INCLUDING bias — as the reference `out_proj(x)`
    does), and sum. Returns ``[hidden]`` float32 on CPU.
    """
    resid = cfg["resid_dim"]
    head_dim = cfg["head_dim"]
    fv = torch.zeros(1, resid, device=device, dtype=model.dtype)
    for (L, H, _) in top:
        x = torch.zeros(resid, dtype=model.dtype, device=device)
        x[H * head_dim:(H + 1) * head_dim] = mean_acts[L, H].to(model.dtype).to(device)
        d = dense_module(model, L)
        with torch.no_grad():
            d_out = d(x.reshape(1, resid))
        fv = fv + d_out
    return fv.reshape(resid).detach().float().to("cpu")
