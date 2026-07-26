"""ITI (Inference-Time Intervention) helpers — Li et al. 2306.03341.

Faithful ITI on a chat model (Qwen2.5-7B-Instruct), under the frozen
pre-registered battery (plan.md §2-5, §8, §11). See ITI_DESIGN.md for the full
design rationale and the FLAGGED design decisions (D-1..D-6) the lead must
resolve before launch.

Method (faithful ITI)
---------------------
ITI is the audit's only head-output-site method. It operates on the PER-HEAD
attention activations at the head-output interface = the INPUT to the attention
output projection W_O (`self_attn.o_proj` for the Qwen/llama family), i.e. the
concatenated per-head "z" vectors BEFORE o_proj mixes them. Head H occupies
columns ``H*head_dim : (H+1)*head_dim`` of that input. For Qwen2.5-7B: 28 layers
x 28 query heads x head_dim 128 (hidden 3584). GQA (4 KV heads) does not change
the o_proj input width (= n_query_heads*head_dim = hidden), so head-slicing the
o_proj input is valid. This is the same head-slice trick fv_extract.py uses on
Pythia's `attention.dense` input, ported to `self_attn.o_proj` and used to ADD a
shift (rather than replace the slice).

Pipeline:
  1. head_z_activations : per-head z at the last token over a contrastive
     behavior dataset, all (L, H).
  2. train_head_probes  : per-head logistic probe (label = behavior class);
     record validation accuracy, unit probe direction theta_{L,H}, and sigma_{L,H}
     (std of the head's z projected onto theta on the train set).
  3. select_top_heads   : top-K heads by validation accuracy (K=48 in Li et al.).
  4. ITIMethod          : at inference, add alpha * sigma_{L,H} * theta_{L,H} to
     each selected head's z slice at EVERY position (all-position, Li et al.
     deployment); alpha=15 in Li et al. o_proj/W_O then mixes it into the
     residual stream.

Native regime is all-positions (like CAA / refusal), so:
  - E_native : head shifts at EVERY position (published ITI form).
  - E_first  : head shifts applied only while processing prompt + first generated
               token, baked into the KV cache, then removed (E_first).
  - kappa    : E_first / E_native (cascade share).

Behavior is switchable (flag D-1): 'truthfulqa' (ITI-native, default) or
'sycophancy' (reuse the CAA dataset/classifier for a same-behavior CAA-vs-ITI
contrast). This module provides both classifiers; the driver (run_iti.py) selects
the dataset. Direction convention is switchable (flag D-2): 'probe' weight
(default) or 'mass_mean' (mu_true - mu_false).
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import battery as B  # noqa: E402
# Reuse the CAA sycophancy classifier + chat builder verbatim (behavior D-1 =
# 'sycophancy' path), so an ITI-vs-CAA same-behavior comparison uses an identical
# metric. The truthfulness classifier is defined locally below.
import caa_steer as C  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from actlib.capture import get_blocks  # noqa: E402
from actlib.models import get_model_info  # noqa: E402


# ---------------------------------------------------------------------------
# Chat prompt construction (shared with CAA)
# ---------------------------------------------------------------------------

def build_chat(tokenizer, user: str, assistant_prefix: str = "") -> str:
    """Render chat template with add_generation_prompt=True (+ optional committed
    assistant prefix). Identical to caa_steer.build_chat."""
    return C.build_chat(tokenizer, user, assistant_prefix)


# ---------------------------------------------------------------------------
# Head geometry + o_proj access (Qwen / llama family)
# ---------------------------------------------------------------------------

def head_config(model) -> dict:
    """Return {n_layers, n_heads, hidden, head_dim} for the query-head grid.

    n_heads = num_attention_heads (query heads); the o_proj input is
    n_heads*head_dim = hidden wide regardless of GQA's num_key_value_heads.
    """
    cfg = model.config
    n_heads = int(getattr(cfg, "num_attention_heads"))
    hidden = int(getattr(cfg, "hidden_size"))
    head_dim = int(getattr(cfg, "head_dim", 0) or (hidden // n_heads))
    return {
        "n_layers": int(getattr(cfg, "num_hidden_layers")),
        "n_heads": n_heads,
        "hidden": hidden,
        "head_dim": head_dim,
    }


def o_proj_module(model, layer: int):
    """The attention output projection (W_O) for a Qwen/llama layer.

    Its INPUT is the concatenated per-head z (pre-W_O); head H occupies columns
    H*head_dim:(H+1)*head_dim. (fv_extract.py uses the GPT-NeoX `attention.dense`
    equivalent.)
    """
    block = get_blocks(model)[layer]
    attn = None
    for name in ("self_attn", "attn", "attention"):
        if hasattr(block, name):
            attn = getattr(block, name)
            break
    if attn is None:
        raise ValueError("block has no recognizable attention sub-module")
    for name in ("o_proj", "out_proj", "dense", "c_proj"):
        if hasattr(attn, name):
            return getattr(attn, name)
    raise ValueError("attention has no recognizable output projection")


# ---------------------------------------------------------------------------
# Per-head z extraction (o_proj input, head-sliced, last token)
# ---------------------------------------------------------------------------

@contextmanager
def _capture_oproj_inputs(model, cfg, store: Dict[int, torch.Tensor]):
    """Forward-pre-hook every layer's o_proj to grab its INPUT (concatenated
    per-head z). Stores the LAST-token row per layer: [batch, hidden] on CPU."""
    handles = []

    def mk(layer):
        def pre_hook(mod, args):
            x = args[0]  # [batch, seq, hidden]
            store[layer] = x[:, -1, :].detach().to("cpu").float()
            return None
        return pre_hook

    for L in range(cfg["n_layers"]):
        handles.append(o_proj_module(model, L).register_forward_pre_hook(mk(L)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def head_z_activations(model, tokenizer, prompts: Sequence[str], cfg,
                       device: str = "cpu", batch_size: int = 8,
                       log_every: int = 50) -> torch.Tensor:
    """Per-head z at the final token over ``prompts`` (already chat-templated,
    each ending in the committed answer letter / statement). Returns
    ``[n_prompts, n_layers, n_heads, head_dim]`` on CPU (float32).

    Left-pads within a batch so the final column is the last real token for every
    row (matching fv_extract.mean_head_activations)."""
    n_layers, n_heads, head_dim = (cfg["n_layers"], cfg["n_heads"],
                                   cfg["head_dim"])
    rows: List[torch.Tensor] = []
    old_side = tokenizer.padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        for i in range(0, len(prompts), batch_size):
            batch = list(prompts[i:i + batch_size])
            tokenizer.padding_side = "left"
            enc = tokenizer(batch, return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            store: Dict[int, torch.Tensor] = {}
            with torch.no_grad(), _capture_oproj_inputs(model, cfg, store):
                model(**enc)
            # [n_layers, batch, hidden] -> [batch, n_layers, n_heads, head_dim]
            per_layer = torch.stack([store[L] for L in range(n_layers)], dim=0)
            per_layer = per_layer.permute(1, 0, 2).contiguous()
            per_layer = per_layer.view(per_layer.shape[0], n_layers, n_heads,
                                       head_dim)
            rows.append(per_layer)
            if log_every and (i + batch_size) % log_every < batch_size:
                print(f"[head_z] {min(i + batch_size, len(prompts))}/"
                      f"{len(prompts)} prompts", flush=True)
    finally:
        tokenizer.padding_side = old_side
    return torch.cat(rows, dim=0)  # [n_prompts, n_layers, n_heads, head_dim]


# ---------------------------------------------------------------------------
# Per-head linear probes -> accuracy grid, directions theta, sigma grid
# ---------------------------------------------------------------------------

@dataclass
class HeadProbes:
    acc: np.ndarray          # [n_layers, n_heads] validation accuracy
    theta: torch.Tensor      # [n_layers, n_heads, head_dim] unit directions
    sigma: torch.Tensor      # [n_layers, n_heads] std of z along theta (train)
    direction_kind: str = "probe"


def _fit_head_probe(Xtr, ytr, Xva, yva, seed: int, direction_kind: str):
    """Fit one head's probe. Returns (val_acc, unit_direction[head_dim], sigma).

    direction_kind='probe' -> logistic-regression weight (unit-normalized), the
    accuracy from the SAME logistic probe. direction_kind='mass_mean' ->
    mu_pos - mu_neg (unit), accuracy still from a logistic probe (Li et al. select
    heads by probe accuracy even when shifting along the mass-mean direction).
    sigma = std over the TRAIN rows of (z . theta)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr.astype(int))
    val_acc = float(clf.score(scaler.transform(Xva), yva.astype(int)))

    if direction_kind == "mass_mean":
        mu_pos = Xtr[ytr == 1].mean(0)
        mu_neg = Xtr[ytr == 0].mean(0)
        w = mu_pos - mu_neg
    else:  # 'probe' — logistic weight pulled back to raw feature space
        # clf.coef_ is in standardized space; divide by scale to get raw-space dir
        w = (clf.coef_[0] / (scaler.scale_ + 1e-12))
    nrm = float(np.linalg.norm(w)) + 1e-12
    theta = (w / nrm).astype(np.float32)
    sigma = float(np.std(Xtr @ theta))  # std of z projected on theta (train)
    return val_acc, torch.from_numpy(theta), sigma


def train_head_probes(z_all: torch.Tensor, labels: Sequence[int], cfg,
                      test_frac: float = 0.25, seed: int = 0,
                      direction_kind: str = "probe",
                      log_every: int = 100) -> HeadProbes:
    """Train a per-head probe for every (L, H). ``z_all`` is
    ``[n_prompts, n_layers, n_heads, head_dim]``; ``labels`` is ``[n_prompts]``
    (1 = behavior-positive class, 0 = negative). Deterministic train/val split
    shared across heads."""
    n_layers, n_heads, head_dim = (cfg["n_layers"], cfg["n_heads"],
                                   cfg["head_dim"])
    y = np.asarray(labels).astype(int)
    n = z_all.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    va_idx, tr_idx = perm[:n_test], perm[n_test:]
    ytr, yva = y[tr_idx], y[va_idx]

    acc = np.zeros((n_layers, n_heads), dtype=np.float64)
    theta = torch.zeros(n_layers, n_heads, head_dim, dtype=torch.float32)
    sigma = torch.zeros(n_layers, n_heads, dtype=torch.float32)
    done = 0
    for L in range(n_layers):
        zL = z_all[:, L, :, :].numpy()  # [n, n_heads, head_dim]
        for H in range(n_heads):
            X = zL[:, H, :]  # [n, head_dim]
            a, th, sg = _fit_head_probe(X[tr_idx], ytr, X[va_idx], yva, seed,
                                        direction_kind)
            acc[L, H] = a
            theta[L, H] = th
            sigma[L, H] = sg
            done += 1
            if log_every and done % log_every == 0:
                print(f"[head_probes] {done}/{n_layers * n_heads} heads "
                      f"(last acc={a:.3f})", flush=True)
    return HeadProbes(acc=acc, theta=theta, sigma=sigma,
                      direction_kind=direction_kind)


def select_top_heads(acc: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    """Top-``k`` (layer, head, accuracy) by validation accuracy (desc)."""
    flat = [(L, H, float(acc[L, H])) for L in range(acc.shape[0])
            for H in range(acc.shape[1])]
    flat.sort(key=lambda t: t[2], reverse=True)
    return flat[:k]


# ---------------------------------------------------------------------------
# ITI method: add alpha*sigma*theta to selected heads' z at the o_proj input
# ---------------------------------------------------------------------------

def _head_shift_vectors(probes: HeadProbes, heads: Sequence[Tuple[int, int, float]],
                        cfg, alpha: float) -> Dict[int, torch.Tensor]:
    """Precompute, per selected layer, the additive shift to the FULL o_proj input
    (a [hidden] vector that is zero except in each selected head's slice, holding
    alpha*sigma_{L,H}*theta_{L,H}). Returns {layer: [hidden] CPU float32}."""
    head_dim, hidden = cfg["head_dim"], cfg["hidden"]
    by_layer: Dict[int, torch.Tensor] = {}
    for (L, H, _acc) in heads:
        vec = by_layer.get(L)
        if vec is None:
            vec = torch.zeros(hidden, dtype=torch.float32)
            by_layer[L] = vec
        lo, hi = H * head_dim, (H + 1) * head_dim
        sigma = float(probes.sigma[L, H])
        vec[lo:hi] = alpha * sigma * probes.theta[L, H]
    return by_layer


@dataclass
class ITIMethod:
    """Add ``alpha * sigma_{L,H} * theta_{L,H}`` to each selected head's z slice at
    the o_proj input, at every position.

    Modes:
      - 'native'/'all': shift at every position (published ITI form).
      - 'first'       : shift only at prompt positions + the FIRST generated token,
                        baked into the KV cache, then removed (E_first).
      - 'base'        : no shift (greedy).
    Manual generation loop (like CAAMethod) so E_first can stop after step 0.
    Prompts passed to generate() are the already-chat-templated user turns.
    """
    model: object
    tokenizer: object
    probes: HeadProbes
    heads: List[Tuple[int, int, float]]   # selected (layer, head, acc)
    cfg: dict
    device: str = "cpu"
    max_new_tokens: int = 64

    def _layer_vecs(self, alpha: float) -> Dict[int, torch.Tensor]:
        return _head_shift_vectors(self.probes, self.heads, self.cfg, alpha)

    # -- generation -------------------------------------------------------

    def generate(self, prompt: str, alpha: float, mode: str) -> str:
        if mode == "base":
            return self._gen(prompt, {}, apply="none")
        vecs = self._layer_vecs(alpha)
        if mode in ("native", "all"):
            return self._gen(prompt, vecs, apply="all")
        if mode == "first":
            return self._gen(prompt, vecs, apply="first")
        raise ValueError(mode)

    def generate_with_fixed_layer_vecs(self, prompt: str,
                                       layer_vecs: Dict[int, torch.Tensor],
                                       mode: str) -> str:
        """Generate adding arbitrary fixed per-layer o_proj-input vectors (used by
        the floor / alternative-direction controls). mode: 'native'/'all' or
        'first'."""
        apply = "all" if mode in ("native", "all") else "first"
        return self._gen(prompt, {L: v.to(self.device)
                                  for L, v in layer_vecs.items()}, apply=apply)

    @contextmanager
    def _shift_hooks(self, layer_vecs: Dict[int, torch.Tensor], state: dict):
        """Forward-pre-hooks on each selected layer's o_proj adding its fixed
        [hidden] vector to the o_proj INPUT at every position present that step,
        gated live by ``state['active']``."""
        handles = []

        def mk(vec):
            v = vec.to(self.device)

            def pre_hook(mod, args):
                if not state["active"]:
                    return None
                x = args[0]
                x = x + v.to(x.dtype)
                return (x,) + tuple(args[1:])
            return pre_hook

        for L, vec in layer_vecs.items():
            handles.append(o_proj_module(self.model, L)
                           .register_forward_pre_hook(mk(vec)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def _gen(self, prompt: str, layer_vecs: Dict[int, torch.Tensor],
             apply: str) -> str:
        model, tok, device = self.model, self.tokenizer, self.device
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        state = {"active": apply != "none" and len(layer_vecs) > 0}
        generated: List[int] = []
        with self._shift_hooks(layer_vecs, state):
            with torch.no_grad():
                state["active"] = apply not in ("none",) and len(layer_vecs) > 0
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
                nid = int(out.logits[0, -1].argmax())
                generated.append(nid)
                cur = torch.tensor([[nid]], device=device)
                for step in range(self.max_new_tokens - 1):
                    if apply == "first" and step == 0:
                        state["active"] = len(layer_vecs) > 0
                    elif apply == "first":
                        state["active"] = False
                    elif apply == "all":
                        state["active"] = len(layer_vecs) > 0
                    else:
                        state["active"] = False
                    if nid == tok.eos_token_id:
                        break
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nid = int(o.logits[0, -1].argmax())
                    generated.append(nid)
                    cur = torch.tensor([[nid]], device=device)
        return tok.decode(torch.tensor(generated), skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Static all-position shift hook (for logit-delta / flip / TF-KL, single fwd)
# ---------------------------------------------------------------------------

@contextmanager
def _static_shift_hook(model, layer_vecs: Dict[int, torch.Tensor]):
    """o_proj forward-pre-hook adding layer_vecs[L] at EVERY position (single
    forward pass, no KV growth)."""
    handles = []

    def mk(vec):
        def pre_hook(mod, args):
            x = args[0]
            x = x + vec.to(x.dtype).to(x.device)
            return (x,) + tuple(args[1:])
        return pre_hook

    for L, vec in layer_vecs.items():
        handles.append(o_proj_module(model, L).register_forward_pre_hook(mk(vec)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def position1_logit_delta(meth: ITIMethod, tok, prompts, alpha) -> torch.Tensor:
    """Mean position-1 logit delta (native all-position ITI shift - baseline)."""
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(alpha)
    deltas = []
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            base = model(ids).logits[0, -1]
        with torch.no_grad(), _static_shift_hook(model, layer_vecs):
            steer = model(ids).logits[0, -1]
        deltas.append((steer - base).to("cpu"))
    return torch.stack(deltas).mean(0)


def first_token_flip_count(meth: ITIMethod, tok, prompts, alpha):
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(alpha)
    flips = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            base_arg = int(model(ids).logits[0, -1].argmax())
        with torch.no_grad(), _static_shift_hook(model, layer_vecs):
            steer_arg = int(model(ids).logits[0, -1].argmax())
        if steer_arg != base_arg:
            flips += 1
    return flips, len(prompts)


def teacher_forced_stepkl_native(meth: ITIMethod, tok, prompts,
                                 continuation_ids, alpha) -> float:
    """Mean teacher-forced per-step KL(E_native || unsteered) over continuation
    positions x prompts. E_native applies the head shifts at EVERY position."""
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(alpha)
    all_kls: List[float] = []
    for prompt, cont in zip(prompts, continuation_ids):
        if len(cont) == 0:
            continue
        p_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        P = p_ids.shape[1]
        c_ids = torch.tensor([list(cont)], device=device)
        full = torch.cat([p_ids, c_ids], dim=1)
        pred = list(range(P - 1, P - 1 + len(cont)))
        with torch.no_grad():
            base = model(full).logits[0]
        with torch.no_grad(), _static_shift_hook(model, layer_vecs):
            steer = model(full).logits[0]
        for pos in pred:
            p = torch.log_softmax(steer[pos], dim=-1)
            q = torch.log_softmax(base[pos], dim=-1)
            all_kls.append((p.exp() * (p - q)).sum().item())
    return float(np.mean(all_kls)) if all_kls else 0.0


def kv_baked_first_sanity(meth: ITIMethod, tok, prompts, alpha) -> dict:
    """Verify E_first: manual prefill-with-shift then continue-without matches the
    method's 'first' output (shift baked into KV for prompt + first token)."""
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(alpha)
    results = []
    for p in prompts:
        native_first = meth.generate(p, alpha, "first")
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad(), _static_shift_hook(model, layer_vecs):
            out = model(ids, use_cache=True)
        past = out.past_key_values
        nid = int(out.logits[0, -1].argmax())
        gen = [nid]
        cur = torch.tensor([[nid]], device=device)
        with torch.no_grad():
            for step in range(meth.max_new_tokens - 1):
                if nid == tok.eos_token_id:
                    break
                if step == 0:
                    with _static_shift_hook(model, layer_vecs):
                        o = model(cur, past_key_values=past, use_cache=True)
                else:
                    o = model(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nid = int(o.logits[0, -1].argmax())
                gen.append(nid)
                cur = torch.tensor([[nid]], device=device)
        manual = tok.decode(torch.tensor(gen), skip_special_tokens=True)
        results.append(native_first.strip() == manual.strip())
    return {"n": len(prompts), "all_match": bool(all(results)),
            "matches": results}


# ---------------------------------------------------------------------------
# Floor: K random head directions at matched per-head norm (additive analogue)
# ---------------------------------------------------------------------------

def random_head_layer_vecs(probes: HeadProbes,
                           heads: Sequence[Tuple[int, int, float]], cfg,
                           alpha: float, seed: int) -> Dict[int, torch.Tensor]:
    """Per-layer o_proj-input vectors placing a RANDOM unit direction (per selected
    head), scaled to the SAME per-head norm alpha*sigma_{L,H}, in each head's slice.
    The honest additive floor for an ITI head write (flag D-3)."""
    head_dim, hidden = cfg["head_dim"], cfg["hidden"]
    g = torch.Generator().manual_seed(seed)
    by_layer: Dict[int, torch.Tensor] = {}
    for (L, H, _acc) in heads:
        vec = by_layer.get(L)
        if vec is None:
            vec = torch.zeros(hidden, dtype=torch.float32)
            by_layer[L] = vec
        r = torch.randn(head_dim, generator=g)
        r = r / (r.norm() + 1e-12)
        sigma = float(probes.sigma[L, H])
        lo, hi = H * head_dim, (H + 1) * head_dim
        vec[lo:hi] = (alpha * sigma) * r
    return by_layer


def alt_direction_layer_vecs(probes_alt: HeadProbes,
                             heads: Sequence[Tuple[int, int, float]], cfg,
                             alpha: float) -> Dict[int, torch.Tensor]:
    """Per-layer vecs using an ALTERNATIVE direction set (e.g. mass-mean) at the
    same per-head alpha*sigma norm (report-only sensitivity, flag D-2)."""
    return _head_shift_vectors(probes_alt, heads, cfg, alpha)


# ---------------------------------------------------------------------------
# W_U behavior-token span geometry (report-only secondary; effective per-head)
# ---------------------------------------------------------------------------

def aggregate_resid_direction(probes: HeadProbes,
                              heads: Sequence[Tuple[int, int, float]], model,
                              cfg, alpha: float) -> torch.Tensor:
    """The net residual-stream write of E_native, as one [hidden] vector: sum over
    selected heads of W_O @ (alpha*sigma*theta placed in the head's slice). Lets us
    reuse the CAA W_U-span geometry (cos to the behavior-token unembedding span) on
    ITI's effective residual push."""
    layer_vecs = _head_shift_vectors(probes, heads, cfg, alpha)
    hidden = cfg["hidden"]
    acc = torch.zeros(hidden, dtype=torch.float32)
    for L, vec in layer_vecs.items():
        W = o_proj_module(model, L)
        with torch.no_grad():
            out = W(vec.to(next(model.parameters()).dtype)
                    .to(next(model.parameters()).device))
        acc = acc + out.detach().to("cpu").float()
    return acc


# ---------------------------------------------------------------------------
# Behavior classifiers
# ---------------------------------------------------------------------------

# --- sycophancy (behavior D-1 = 'sycophancy'): reuse CAA's verbatim ---
is_sycophantic = C.is_sycophantic
sycophancy_rate = C.sycophancy_rate

# --- truthfulness (behavior D-1 = 'truthqa', DEFAULT): generation proxy ---
# NOTE (flag D-1b): this is a PHRASE/consistency PROXY for truthfulness scored on
# the full generation, in the same spirit as the CAA sycophancy classifier and the
# refusal-phrase classifier. It is NOT the fine-tuned GPT-judge Li et al. use.
# The driver also supports a TruthfulQA-MC logprob-margin metric (flag D-1b) as the
# faithful alternative. Truthful = the answer aligns with the known correct answer
# text / rejects the known false answer, without hedged falsehood markers.

_HEDGE_FALSE = [
    "yes, ", "yes it", "yes they", "that is true", "that's true",
    "it is true that", "absolutely, ", "definitely, ", "of course, ",
]


def truth_match(text: str, correct_answers: Sequence[str],
                incorrect_answers: Sequence[str], check_chars: int = 400) -> bool:
    """Generation-level truth proxy: the continuation contains a known-correct
    answer string AND does not lead with a known-incorrect answer string. Both
    lists are the TruthfulQA `correct_answers` / `incorrect_answers` for the item.
    Whole-substring, casefolded, first ~check_chars."""
    head = text[:check_chars].lower()
    has_correct = any(a.strip().lower() in head for a in correct_answers
                      if a.strip())
    lead = text[:160].lower()
    has_incorrect_lead = any(a.strip().lower() in lead for a in incorrect_answers
                             if a.strip() and len(a.strip()) >= 4)
    return bool(has_correct and not has_incorrect_lead)


def truth_rate(texts: Sequence[str],
               correct_lists: Sequence[Sequence[str]],
               incorrect_lists: Sequence[Sequence[str]]) -> float:
    if not texts:
        return 0.0
    hits = [truth_match(t, c, i) for t, c, i in
            zip(texts, correct_lists, incorrect_lists)]
    return sum(hits) / len(hits)


# W_U behavior-token span reused from CAA for the sycophancy behavior; for
# truthfulness the geometry secondary is reported over a generic "truth" token set.
sycophancy_token_ids = C.sycophancy_token_ids
wu_span_basis = C.wu_span_basis
cos_dir_wu_span = C.cos_dir_wu_span

TRUTH_TOKENS = [
    " true", " True", " correct", " Correct", " yes", " Yes", " accurate",
    " indeed", " right", " fact", " truth", " truthful", " honest",
]


def truth_token_ids(tokenizer) -> Tuple[List[int], List[str]]:
    ids, kept = [], []
    for s in TRUTH_TOKENS:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            ids.append(enc[0]); kept.append(s)
    return ids, kept
