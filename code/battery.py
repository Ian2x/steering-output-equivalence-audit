"""Reusable battery for the steering-content-audit anchor/method experiments.

Implements the instruments from plan.md §2/§4/§5:

- Neutral prompt-set generation (deterministic, seeded).
- Wedding lexicon behaviour metric (full-generation keyword scoring, §4).
- Degeneracy hard gate (§4): 3-gram repetition, median length, NLL under the
  unsteered model.
- A0 synthetic "steering method": additive push of a W_U-span direction into the
  final-layer residual stream, realised through actlib's KV-baked one-shot patch
  machinery (`generate_with_patch`, `mode="subspace_transplant"` with an empty
  remove-basis == pure additive add_vector).
- I1 cascade: E_all (patch every generated step) vs E_first (patch prompt + first
  generated token, baked into KV, then absent).
- I2 primary control: calibrated static logit bias (HF LogitsProcessor) on a
  regression-discovered token set, KL-matched by bisection.
- Floor: random direction at matched norm.
- Bootstrap 95% CIs over prompts.

All model interaction goes through `tools/actlib`; hooks are not reinvented here.

Notes on the additive patch
---------------------------
actlib exposes additive semantics via `mode="subspace_transplant"`:
    patched = target - project_onto(target, remove_subspace) + add_vector
With `remove_subspace = torch.zeros(0, hidden)` the projection is zero, so the
result is exactly `target + add_vector`. `add_vector` is a fixed [hidden] vector
(the KV-baked one-shot mode needs a fixed vector), so the norm-scaling in
`alpha * u * (norm of resid at site)` is realised with a *fixed scalar* norm
computed once on the calibration split (the median final-layer resid_post norm at
the last prompt position). Using a fixed scalar is required for E_first/E_all to
inject an identical intervention, making the cascade share well-defined.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# actlib lives under tools/ ; add it to the path once.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from actlib import (  # noqa: E402
    load_model, capture_activations, cache_site, generate_with_patch,
    project_onto,
)
from actlib.models import get_model_info  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt set
# ---------------------------------------------------------------------------

# Topic-neutral open-ended stems. Combined with light templating to produce
# deterministic variants; none references weddings or the target lexicon.
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

# Deterministic prefixes and suffixes used to expand the stem bank to n=200.
_PREFIXES = [
    "", "Please ", "In a few sentences, ", "Briefly, ", "If you can, ",
    "For me, ", "Right now, ", "Honestly, ",
]
_SUFFIXES = [
    "", " Take your time.", " Keep it short.", " Be specific.",
    " Use plain language.", " Feel free to be creative.",
    " Add some detail.", " Just a couple of sentences.",
]


def build_neutral_prompts(n: int = 200, seed: int = 20260705) -> List[str]:
    """Return ``n`` deterministic, topic-neutral open-ended prompts.

    Combinations of (stem, prefix, suffix) are enumerated deterministically and
    a fixed-seed permutation selects ``n`` distinct prompts. Fully reproducible.
    """
    combos: List[str] = []
    for stem in _STEMS:
        for pre in _PREFIXES:
            for suf in _SUFFIXES:
                if pre == "":
                    text = stem + suf
                else:
                    # lowercase the stem's first letter after a prefix
                    text = pre + stem[0].lower() + stem[1:] + suf
                combos.append(text)
    # Deduplicate preserving order.
    seen = set()
    uniq = []
    for c in combos:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(uniq))[:n]
    return [uniq[i] for i in idx]


def split_prompts(prompts: Sequence[str], n_calib: int = 50,
                  seed: int = 20260705) -> Tuple[List[str], List[str]]:
    """Deterministic calibration / eval split (first ``n_calib`` after shuffle)."""
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(prompts))
    calib = [prompts[i] for i in idx[:n_calib]]
    eval_ = [prompts[i] for i in idx[n_calib:]]
    return calib, eval_


# ---------------------------------------------------------------------------
# Wedding lexicon + A0 token set
# ---------------------------------------------------------------------------

WEDDING_LEXICON = [
    "wedding", "weddings", "bride", "bridal", "groom", "married", "marry",
    "marries", "marriage", "wed", "weds", "honeymoon", "ceremony", "vows",
    "vow", "engagement", "engaged", "fiance", "fiancee", "bridesmaid",
    "bouquet", "altar", "reception", "matrimony", "nuptials", "betrothed",
    "wedded",
]

# Leading-space tokens whose W_U columns define the A0 push direction.
A0_TOKEN_STRINGS = [
    " wedding", " bride", " groom", " married", " marriage", " wed",
    " honeymoon", " ceremony",
]


def wedding_topic_hit(text: str) -> bool:
    """True if ``text`` contains >= 1 wedding-lexicon word (whole-word, casefold)."""
    import re
    low = text.lower()
    for w in WEDDING_LEXICON:
        if re.search(r"\b" + re.escape(w) + r"\b", low):
            return True
    return False


def topic_rate(texts: Sequence[str]) -> float:
    if not texts:
        return 0.0
    return sum(wedding_topic_hit(t) for t in texts) / len(texts)


def resolve_a0_token_ids(tokenizer) -> Tuple[List[int], List[str]]:
    """Return single-token ids for the A0 token set; drop multi-token entries."""
    ids, kept = [], []
    for s in A0_TOKEN_STRINGS:
        enc = tokenizer.encode(s)
        if len(enc) == 1:
            ids.append(enc[0])
            kept.append(s)
    return ids, kept


# ---------------------------------------------------------------------------
# Direction construction
# ---------------------------------------------------------------------------

def unembedding_matrix(model) -> torch.Tensor:
    """Return W_U as [vocab, hidden] (the lm_head weight)."""
    return model.lm_head.weight.detach()


def a0_direction(model, token_ids: Sequence[int]) -> torch.Tensor:
    """L2-normalised sum of W_U columns for ``token_ids`` -> [hidden]."""
    WU = unembedding_matrix(model)  # [vocab, hidden]
    cols = WU[list(token_ids)]      # [k, hidden]
    u = cols.sum(0)
    u = u / u.norm()
    return u.float()


def final_layer(model) -> int:
    return get_model_info(model).n_layers - 1


def calib_resid_norm(model, tokenizer, prompts: Sequence[str], layer: int,
                     device: str = "cpu", batch_size: int = 16) -> float:
    """Median final-position resid_post norm at ``layer`` over ``prompts``.

    This scalar fixes the additive push magnitude so E_first/E_all inject an
    identical vector (see module docstring).
    """
    norms = []
    for i in range(0, len(prompts), batch_size):
        batch = list(prompts[i:i + batch_size])
        acts = capture_activations(model, tokenizer, batch, "resid_post", layer,
                                   device=device)[(layer, "resid_post")]
        norms.append(acts.norm(dim=-1))
    n = torch.cat(norms)
    return float(n.median().item())


# ---------------------------------------------------------------------------
# Additive-patch generation (A0 method) via actlib
# ---------------------------------------------------------------------------

@dataclass
class A0Method:
    """The A0 synthetic steering method: additive final-layer resid push.

    ``add_vector = alpha * u * scale`` where ``scale`` is the fixed calibration
    resid norm. Generation uses actlib's KV-baked one-shot patch machinery.
    """
    model: object
    tokenizer: object
    layer: int
    u: torch.Tensor          # [hidden] unit direction
    scale: float             # fixed resid-norm scalar
    device: str = "cpu"
    max_new_tokens: int = 64

    def add_vector(self, alpha: float) -> torch.Tensor:
        return (alpha * self.scale) * self.u.to(self.device)

    def _empty_basis(self) -> torch.Tensor:
        hidden = self.u.shape[0]
        return torch.zeros(0, hidden, device=self.device)

    def generate(self, prompt: str, alpha: float, mode: str) -> str:
        """mode: 'all' (E_all, every step), 'first' (E_first, KV-baked one-shot),
        or 'base' (no push)."""
        src = cache_site(self.model, self.tokenizer, prompt, "resid_post",
                         self.layer, device=self.device)
        rb = self._empty_basis()
        if mode == "base":
            addv = torch.zeros(self.u.shape[0], device=self.device)
            positions = "all_generated"
        elif mode == "all":
            addv = self.add_vector(alpha)
            positions = "all_generated"
        elif mode == "first":
            addv = self.add_vector(alpha)
            positions = "last_prompt"
        else:
            raise ValueError(mode)
        out = generate_with_patch(
            self.model, self.tokenizer, prompt, [src],
            remove_subspace=rb, add_vector=addv, mode="subspace_transplant",
            positions=positions, max_new_tokens=self.max_new_tokens,
            device=self.device)
        return out["continuation"]

    def generate_with_vector(self, prompt: str, addv: torch.Tensor,
                             positions: str) -> str:
        """Generate with an arbitrary fixed add_vector (used by the floor)."""
        src = cache_site(self.model, self.tokenizer, prompt, "resid_post",
                         self.layer, device=self.device)
        rb = self._empty_basis()
        out = generate_with_patch(
            self.model, self.tokenizer, prompt, [src],
            remove_subspace=rb, add_vector=addv.to(self.device),
            mode="subspace_transplant", positions=positions,
            max_new_tokens=self.max_new_tokens, device=self.device)
        return out["continuation"]


def base_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64,
                  device: str = "cpu") -> str:
    """Unsteered greedy continuation (no hooks)."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1,
                             pad_token_id=tokenizer.eos_token_id)
    cont_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


def base_generate_ids(model, tokenizer, prompt: str, max_new_tokens: int = 64,
                      device: str = "cpu") -> List[int]:
    """Unsteered greedy continuation token ids (no hooks). Used to teacher-force
    the exact generated sequence for per-step KL calibration."""
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1,
                             pad_token_id=tokenizer.eos_token_id)
    return out[0, input_ids.shape[1]:].to("cpu").tolist()


# ---------------------------------------------------------------------------
# ActAdd (Turner et al. 2308.10248) — Activation Addition steering vector
# ---------------------------------------------------------------------------

def build_actadd_hdelta(model, tokenizer, layer: int, p_plus: str = " weddings",
                        p_minus: str = " ", device: str = "cpu"
                        ) -> Tuple[torch.Tensor, dict]:
    """Construct the ActAdd steering delta at ``resid_pre`` of ``layer``.

    Faithful to Turner et al.: h_delta = act(p+) - act(p-) at resid_pre of the
    injection layer, captured at the token positions of the front-aligned,
    right-padded-to-equal-length prompt pair. Returns

        h_delta : [pad_len, hidden]  (per-position difference vector)
        info    : dict with tokenizations, pad_len, per-position and mean norms.

    For the canonical wedding demo on GPT-2 both p+=" weddings" and p-=" " are
    single tokens, so pad_len == 1 and h_delta is a single [1, hidden] row.
    """
    ids_plus = tokenizer.encode(p_plus)
    ids_minus = tokenizer.encode(p_minus)
    pad_len = max(len(ids_plus), len(ids_minus))
    # Right-pad the shorter sequence to equal length (front-aligned) with the
    # pad/eos token id, matching ActAdd's tokens_to_prompts equal-length pairing.
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def _padded(ids):
        return ids + [pad_id] * (pad_len - len(ids))

    ids_plus_p = _padded(ids_plus)
    ids_minus_p = _padded(ids_minus)

    def _resid_pre(ids):
        t = torch.tensor([ids], device=device)
        store: dict = {}
        from actlib.capture import capture_hooks
        with torch.no_grad(), capture_hooks(model, ["resid_pre"], [layer], store):
            model(t)
        return store[(layer, "resid_pre")][-1][0].to("cpu").float()  # [pad_len, hidden]

    a_plus = _resid_pre(ids_plus_p)
    a_minus = _resid_pre(ids_minus_p)
    h_delta = a_plus - a_minus  # [pad_len, hidden]

    per_pos_norms = [float(h_delta[i].norm()) for i in range(pad_len)]
    mean_vec = h_delta.mean(0)
    info = {
        "p_plus": p_plus, "p_minus": p_minus,
        "ids_plus": ids_plus, "ids_minus": ids_minus,
        "ids_plus_padded": ids_plus_p, "ids_minus_padded": ids_minus_p,
        "pad_len": pad_len, "pad_id": pad_id, "layer": layer,
        "per_position_norms": per_pos_norms,
        "mean_vector_norm": float(mean_vec.norm()),
        "hidden": int(h_delta.shape[1]),
    }
    return h_delta, info


@dataclass
class ActAddMethod:
    """Turner et al. ActAdd: add c * h_delta at resid_pre of ``layer``.

    h_delta is a [pad_len, hidden] per-position tensor; injection adds
    ``coeff * h_delta[i]`` at aligned front position ``i`` of the user prompt.

    Two deployment modes:
      - ``native`` (E_native, as published): inject only at prompt positions
        0..pad_len-1 (front-aligned). Under KV caching those layer-6 activations
        are baked into the cache during prefill and persist implicitly for the
        whole generation; the intervention is intrinsically one-shot / KV-
        persistent (no re-application at generated positions).
      - ``all`` (E_all, all-positions variant): inject at every position — the
        front prompt positions 0..pad_len-1 during prefill AND ``coeff*h_delta``
        (the last/only delta row, broadcast) at every generated token position.
    """
    model: object
    tokenizer: object
    layer: int
    h_delta: torch.Tensor    # [pad_len, hidden] on CPU
    device: str = "cpu"
    max_new_tokens: int = 64

    @property
    def pad_len(self) -> int:
        return self.h_delta.shape[0]

    def _delta_rows(self, coeff: float) -> torch.Tensor:
        return (coeff * self.h_delta).to(self.device)  # [pad_len, hidden]

    def generate(self, prompt: str, coeff: float, mode: str) -> str:
        """mode: 'native', 'all', or 'base'."""
        from actlib.patching import _dynamic_patch_hook
        model, tok, device, L = self.model, self.tokenizer, self.device, self.layer
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        n_front = min(self.pad_len, prompt_len)  # front-aligned injection span
        rows = self._delta_rows(coeff)           # [pad_len, hidden]
        last_row = rows[-1]                       # broadcast vector for E_all

        if mode == "base":
            return base_generate(model, tok, prompt, self.max_new_tokens, device)
        if mode not in ("native", "all"):
            raise ValueError(mode)

        # native: per-position rows at front prompt positions 0..n_front-1 only,
        #         baked into the KV cache during prefill (one-shot / persistent).
        # all   : the (single) broadcast delta at EVERY position — all prompt
        #         positions AND every generated token position.
        front_vecs = {p: rows[p] for p in range(n_front)}

        generated: List[int] = []
        state = {"cache_offset": 0}

        from contextlib import contextmanager

        @contextmanager
        def _hook():
            module, _is_pre = _resolve_pre_module(model, L)

            def _apply(hs):
                seq = hs.shape[1]
                offset = state["cache_offset"]
                out = None
                for col in range(seq):
                    absidx = offset + col
                    if mode == "all":
                        add = last_row              # every position
                    else:
                        add = front_vecs.get(absidx)  # native: front only
                    if add is not None:
                        if out is None:
                            out = hs.clone()
                        out[:, col, :] = out[:, col, :] + add.to(hs.dtype)
                return out if out is not None else hs

            def pre_hook(mod, args):
                return (_apply(args[0]),) + tuple(args[1:])
            h = module.register_forward_pre_hook(pre_hook)
            try:
                yield
            finally:
                h.remove()

        with torch.no_grad(), _hook():
            cur_ids = input_ids
            past = None
            for step in range(self.max_new_tokens):
                abs_step = prompt_len - 1 + step
                if past is not None:
                    state["cache_offset"] = abs_step
                    fwd_ids = cur_ids[:, -1:]
                    out = model(fwd_ids, past_key_values=past, use_cache=True)
                else:
                    state["cache_offset"] = 0
                    out = model(cur_ids, use_cache=True)
                past = out.past_key_values
                nid = int(out.logits[0, -1].argmax())
                generated.append(nid)
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[nid]], device=device)], dim=1)
        return tok.decode(torch.tensor(generated), skip_special_tokens=True)

    def generate_with_fixed_vector(self, prompt: str, vec: torch.Tensor,
                                   mode: str) -> str:
        """Generate adding a fixed [hidden] ``vec`` (native or all positions).

        Used by the floor (random direction) and the W_U secondary control, which
        supply an arbitrary fixed vector at the ActAdd site/positions/norm.
        mode: 'native' (front prompt positions only) or 'all' (every position)."""
        # Build a [pad_len, hidden] delta whose rows are all ``vec`` so we can
        # reuse the same injection machinery. coeff=1 (vec already scaled).
        rows = vec.to(self.device).unsqueeze(0).repeat(self.pad_len, 1)
        saved = self.h_delta
        try:
            self.h_delta = rows.to("cpu")
            return self.generate(prompt, 1.0, mode)
        finally:
            self.h_delta = saved


def _resolve_pre_module(model, layer):
    """Return (block_module, True) for a resid_pre forward-pre hook."""
    from actlib.capture import get_blocks
    return get_blocks(model)[layer], True


def actadd_position1_logit_delta(meth: "ActAddMethod", tokenizer,
                                 prompts: Sequence[str], coeff: float
                                 ) -> torch.Tensor:
    """Mean position-1 (first generated step) logit delta (E_native - baseline).

    Uses the native front-position injection (prompt positions 0..pad_len-1,
    KV-baked) then reads the next-token logits at the last prompt position.
    Returns [vocab] mean delta (on CPU)."""
    model, device, L = meth.model, meth.device, meth.layer
    rows = meth._delta_rows(coeff)
    n_front_cap = meth.pad_len
    deltas = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        n_front = min(n_front_cap, prompt_len)
        with torch.no_grad():
            base_logits = model(input_ids).logits[0, -1]
        vecmap = {p: rows[p] for p in range(n_front)}
        with torch.no_grad(), _actadd_static_hook(model, L, vecmap):
            steered_logits = model(input_ids).logits[0, -1]
        deltas.append((steered_logits - base_logits).to("cpu"))
    return torch.stack(deltas).mean(0)


from contextlib import contextmanager as _contextmanager


@_contextmanager
def _actadd_static_hook(model, layer, vecmap):
    """resid_pre pre-hook adding vecmap[abs_pos] at listed absolute positions
    (single forward pass, no KV growth: cache_offset assumed 0)."""
    module, _ = _resolve_pre_module(model, layer)

    def pre_hook(mod, args):
        hs = args[0]
        out = hs.clone()
        for pos, v in vecmap.items():
            if 0 <= pos < hs.shape[1]:
                out[:, pos, :] = out[:, pos, :] + v.to(hs.dtype).to(hs.device)
        return (out,) + tuple(args[1:])
    h = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        h.remove()


def actadd_first_token_flip_count(meth: "ActAddMethod", tokenizer,
                                  prompts: Sequence[str], coeff: float
                                  ) -> Tuple[int, int]:
    """Count prompts whose FIRST generated token's argmax is changed by the
    native ActAdd injection vs unsteered baseline (one forward per prompt)."""
    model, device, L = meth.model, meth.device, meth.layer
    rows = meth._delta_rows(coeff)
    flips = 0
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        n_front = min(meth.pad_len, prompt_len)
        with torch.no_grad():
            base_arg = int(model(input_ids).logits[0, -1].argmax())
        vecmap = {p: rows[p] for p in range(n_front)}
        with torch.no_grad(), _actadd_static_hook(model, L, vecmap):
            steer_arg = int(model(input_ids).logits[0, -1].argmax())
        if steer_arg != base_arg:
            flips += 1
    return flips, len(prompts)


def actadd_teacher_forced_stepkl_native(
        meth: "ActAddMethod", tokenizer, prompts: Sequence[str],
        continuation_ids: Sequence[Sequence[int]], coeff: float) -> float:
    """Mean teacher-forced per-step KL(E_native || unsteered) over all 64
    continuation positions x prompts (the control-budget quantity B*).

    E_native injects c*h_delta at the front prompt positions only; under teacher
    forcing on the fixed unsteered continuation we read KL at every position that
    predicts a continuation token. The injection is applied at the prompt front
    positions of the full teacher-forced sequence (a single forward)."""
    model, device, L = meth.model, meth.device, meth.layer
    rows = meth._delta_rows(coeff)
    all_kls: List[float] = []
    for prompt, cont in zip(prompts, continuation_ids):
        if len(cont) == 0:
            continue
        p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        P = p_ids.shape[1]
        n_front = min(meth.pad_len, P)
        c_ids = torch.tensor([list(cont)], device=device)
        full = torch.cat([p_ids, c_ids], dim=1)
        n = len(cont)
        pred_positions = list(range(P - 1, P - 1 + n))
        with torch.no_grad():
            base_logits = model(full).logits[0]
        vecmap = {p: rows[p] for p in range(n_front)}
        with torch.no_grad(), _actadd_static_hook(model, L, vecmap):
            steered_logits = model(full).logits[0]
        for pos in pred_positions:
            p = torch.log_softmax(steered_logits[pos], dim=-1)
            q = torch.log_softmax(base_logits[pos], dim=-1)
            all_kls.append((p.exp() * (p - q)).sum().item())
    return float(np.mean(all_kls)) if all_kls else 0.0


def wu_wedding_span_basis(model, token_ids: Sequence[int]) -> torch.Tensor:
    """Orthonormal basis (rows) of span(W_U[token_ids]) — raw W_U columns as
    resid directions (naive pullback, no lens). Returns [k, hidden]."""
    WU = unembedding_matrix(model)  # [vocab, hidden]
    cols = WU[list(token_ids)].float()  # [k, hidden]
    q, _ = torch.linalg.qr(cols.transpose(0, 1))  # [hidden, k]
    return q.transpose(0, 1)  # [k, hidden]


# ---------------------------------------------------------------------------
# Degeneracy gate (§4)
# ---------------------------------------------------------------------------

def three_gram_rep_rate(text: str, tokenizer) -> float:
    """Fraction of 3-grams (token-level) that are repeats. 0 if < 3 tokens."""
    ids = tokenizer.encode(text)
    if len(ids) < 3:
        return 0.0
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return 1.0 - uniq / len(grams)


def mean_len_tokens(texts: Sequence[str], tokenizer) -> float:
    if not texts:
        return 0.0
    return float(np.mean([len(tokenizer.encode(t)) for t in texts]))


def median_len_tokens(texts: Sequence[str], tokenizer) -> float:
    if not texts:
        return 0.0
    return float(np.median([len(tokenizer.encode(t)) for t in texts]))


def mean_nll_under_model(model, tokenizer, prompt: str, continuation: str,
                         device: str = "cpu") -> float:
    """Mean per-token NLL of ``continuation`` given ``prompt`` under the
    UNSTEERED model (no hooks). Only continuation tokens are scored."""
    p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    c_ids = tokenizer(continuation, return_tensors="pt",
                      add_special_tokens=False)["input_ids"].to(device)
    if c_ids.shape[1] == 0:
        return 0.0
    full = torch.cat([p_ids, c_ids], dim=1)
    with torch.no_grad():
        logits = model(full).logits  # [1, seq, vocab]
    # predict token t from position t-1; score continuation tokens only
    n_prompt = p_ids.shape[1]
    logp = torch.log_softmax(logits[0], dim=-1)
    total, count = 0.0, 0
    for t in range(n_prompt, full.shape[1]):
        tok = full[0, t]
        total += -logp[t - 1, tok].item()
        count += 1
    return total / max(count, 1)


@dataclass
class GateResult:
    tripped: bool
    rep_rate: float
    median_len: float
    mean_nll: float
    reasons: List[str] = field(default_factory=list)


def degeneracy_gate(cond_texts: Sequence[str], prompts: Sequence[str],
                    baseline_rep: float, baseline_median_len: float,
                    baseline_nll: float, model, tokenizer,
                    device: str = "cpu") -> GateResult:
    """Compute the §4 hard gate for one condition.

    Void if ANY: mean 3-gram rep > 2*baseline_rep + 0.1;
    median length < 0.5*baseline_median_len;
    mean per-token NLL under unsteered model > 3*baseline_nll.
    """
    reps = [three_gram_rep_rate(t, tokenizer) for t in cond_texts]
    rep = float(np.mean(reps)) if reps else 0.0
    med = median_len_tokens(cond_texts, tokenizer)
    nlls = [mean_nll_under_model(model, tokenizer, p, t, device=device)
            for p, t in zip(prompts, cond_texts)]
    nll = float(np.mean(nlls)) if nlls else 0.0

    reasons = []
    if rep > 2 * baseline_rep + 0.1:
        reasons.append(
            f"3gram_rep {rep:.3f} > 2*{baseline_rep:.3f}+0.1="
            f"{2 * baseline_rep + 0.1:.3f}")
    if med < 0.5 * baseline_median_len:
        reasons.append(
            f"median_len {med:.1f} < 0.5*{baseline_median_len:.1f}="
            f"{0.5 * baseline_median_len:.1f}")
    if nll > 3 * baseline_nll:
        reasons.append(
            f"mean_nll {nll:.3f} > 3*{baseline_nll:.3f}={3 * baseline_nll:.3f}")
    return GateResult(bool(reasons), rep, med, nll, reasons)


# ---------------------------------------------------------------------------
# I2 primary control: calibrated static logit bias
# ---------------------------------------------------------------------------

def position1_logit_delta(method: "A0Method", tokenizer, prompts: Sequence[str],
                          alpha: float) -> torch.Tensor:
    """Mean position-1 (first generated step) logit delta (steered - unsteered).

    Steered = one additive push at the prompt's last token (the first generated
    step's prediction), matching how E_all begins. Returns [vocab] mean delta.
    """
    model = method.model
    device = method.device
    addv = method.add_vector(alpha)
    hidden = method.u.shape[0]
    rb = torch.zeros(0, hidden, device=device)
    layer = method.layer
    deltas = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            base_logits = model(input_ids).logits[0, -1]  # [vocab]
        # steered: patch the last prompt position's resid_post, read next logits
        from actlib.patching import _dynamic_patch_hook
        state = {"cache_offset": 0, "targets": [input_ids.shape[1] - 1]}
        with torch.no_grad(), _dynamic_patch_hook(
                model, "resid_post", layer, None, None, "subspace_transplant",
                state, remove_subspace=rb, add_vector=addv):
            steered_logits = model(input_ids).logits[0, -1]
        deltas.append((steered_logits - base_logits).to("cpu"))
    return torch.stack(deltas).mean(0)  # [vocab]


def discover_token_set(mean_delta: torch.Tensor, coverage: float = 0.90,
                       cap: int = 100) -> List[int]:
    """Smallest token set by |delta| capturing ``coverage`` of ||delta||^2, capped."""
    sq = mean_delta.pow(2)
    total = sq.sum().item()
    if total <= 0:
        return []
    order = torch.argsort(sq, descending=True)
    cum = 0.0
    chosen = []
    for idx in order.tolist():
        chosen.append(idx)
        cum += sq[idx].item()
        if cum / total >= coverage or len(chosen) >= cap:
            break
    return chosen


def _position1_kl_steered(method: "A0Method", tokenizer, prompts: Sequence[str],
                          alpha: float) -> float:
    """Mean position-1 KL(steered || unsteered) for the additive push."""
    model = method.model
    device = method.device
    addv = method.add_vector(alpha)
    hidden = method.u.shape[0]
    rb = torch.zeros(0, hidden, device=device)
    layer = method.layer
    from actlib.patching import _dynamic_patch_hook
    kls = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            base_logits = model(input_ids).logits[0, -1]
        state = {"cache_offset": 0, "targets": [input_ids.shape[1] - 1]}
        with torch.no_grad(), _dynamic_patch_hook(
                model, "resid_post", layer, None, None, "subspace_transplant",
                state, remove_subspace=rb, add_vector=addv):
            steered_logits = model(input_ids).logits[0, -1]
        p = torch.log_softmax(steered_logits, dim=-1)
        q = torch.log_softmax(base_logits, dim=-1)
        kl = (p.exp() * (p - q)).sum().item()
        kls.append(kl)
    return float(np.mean(kls))


def _position1_kl_biased(model, tokenizer, prompts: Sequence[str],
                         token_ids: Sequence[int], bias_vals: torch.Tensor,
                         device: str = "cpu") -> float:
    """Mean position-1 KL(control || unsteered) where control adds bias_vals to
    the given token logits at position 1."""
    kls = []
    tid = torch.tensor(list(token_ids), device=device)
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            base_logits = model(input_ids).logits[0, -1].clone()
        biased = base_logits.clone()
        biased[tid] = biased[tid] + bias_vals.to(device)
        p = torch.log_softmax(biased, dim=-1)
        q = torch.log_softmax(base_logits, dim=-1)
        kl = (p.exp() * (p - q)).sum().item()
        kls.append(kl)
    return float(np.mean(kls))


def calibrate_bias_scalar(model, tokenizer, prompts: Sequence[str],
                          token_ids: Sequence[int], base_delta: torch.Tensor,
                          target_kl: float, device: str = "cpu",
                          lo: float = 0.0, hi: float = 8.0,
                          iters: int = 30) -> Tuple[float, float]:
    """Bisect a scalar ``c`` s.t. position-1 KL(control||unsteered) == target_kl.

    Control bias vector = c * base_delta[token_ids]. Returns (c, achieved_kl).
    KL increases monotonically with c (from 0 at c=0), so bisection is valid.
    """
    tid = torch.tensor(list(token_ids), device=device)
    dvals = base_delta[tid.cpu()].to(device)

    def kl_at(c):
        return _position1_kl_biased(model, tokenizer, prompts, token_ids,
                                    c * dvals, device=device)

    # expand hi until it brackets target
    khi = kl_at(hi)
    tries = 0
    while khi < target_kl and tries < 12:
        hi *= 1.5
        khi = kl_at(hi)
        tries += 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        k = kl_at(mid)
        if k < target_kl:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c, kl_at(c)


def _teacher_forced_seq(tokenizer, prompt: str, continuation_ids: Sequence[int],
                        device: str) -> Tuple[torch.Tensor, int]:
    """Build the full [1, seq] token tensor (prompt + fixed continuation) and
    return it with the prompt length P. Logits at absolute position P-1+i predict
    continuation token i (i in 0..len(cont)-1)."""
    p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    P = p_ids.shape[1]
    c_ids = torch.tensor([list(continuation_ids)], dtype=p_ids.dtype,
                         device=device)
    full = torch.cat([p_ids, c_ids], dim=1)
    return full, P


def teacher_forced_stepkl_steered(
        method: "A0Method", tokenizer, prompts: Sequence[str],
        continuation_ids: Sequence[Sequence[int]], alpha: float) -> float:
    """Mean teacher-forced per-step KL(steered || unsteered), averaged over all
    continuation positions and all prompts.

    The steered model has the additive push (``alpha``) applied at EVERY position
    that produces a continuation token (positions P-1 .. P-1+n_cont-1), i.e. the
    exact E_all patch configuration, teacher-forced on the fixed unsteered
    continuation. KL is read at each such position (predicting cont token i).
    """
    from actlib.patching import _dynamic_patch_hook
    model = method.model
    device = method.device
    layer = method.layer
    hidden = method.u.shape[0]
    addv = method.add_vector(alpha)
    rb = torch.zeros(0, hidden, device=device)
    all_kls: List[float] = []
    for prompt, cont in zip(prompts, continuation_ids):
        if len(cont) == 0:
            continue
        full, P = _teacher_forced_seq(tokenizer, prompt, cont, device)
        n = len(cont)
        # positions producing cont tokens 0..n-1 (absolute indices P-1 .. P-2+n)
        pred_positions = list(range(P - 1, P - 1 + n))
        with torch.no_grad():
            base_logits = model(full).logits[0]  # [seq, vocab]
        state = {"cache_offset": 0, "targets": pred_positions}
        with torch.no_grad(), _dynamic_patch_hook(
                model, "resid_post", layer, None, None, "subspace_transplant",
                state, remove_subspace=rb, add_vector=addv):
            steered_logits = model(full).logits[0]  # [seq, vocab]
        for pos in pred_positions:
            p = torch.log_softmax(steered_logits[pos], dim=-1)
            q = torch.log_softmax(base_logits[pos], dim=-1)
            all_kls.append((p.exp() * (p - q)).sum().item())
    return float(np.mean(all_kls)) if all_kls else 0.0


def teacher_forced_stepkl_biased(
        model, tokenizer, prompts: Sequence[str],
        continuation_ids: Sequence[Sequence[int]], token_ids: Sequence[int],
        bias_vals: torch.Tensor, device: str = "cpu") -> float:
    """Mean teacher-forced per-step KL(control || unsteered) for the static logit
    bias (``bias_vals`` added to ``token_ids``) applied at every continuation
    position, teacher-forced on the SAME fixed continuations. Matches the
    steered quantity's averaging (all positions x prompts)."""
    tid = torch.tensor(list(token_ids), device=device)
    bv = bias_vals.to(device)
    all_kls: List[float] = []
    for prompt, cont in zip(prompts, continuation_ids):
        if len(cont) == 0:
            continue
        full, P = _teacher_forced_seq(tokenizer, prompt, cont, device)
        n = len(cont)
        pred_positions = list(range(P - 1, P - 1 + n))
        with torch.no_grad():
            base_logits = model(full).logits[0]  # [seq, vocab]
        for pos in pred_positions:
            base = base_logits[pos]
            biased = base.clone()
            biased[tid] = biased[tid] + bv
            p = torch.log_softmax(biased, dim=-1)
            q = torch.log_softmax(base, dim=-1)
            all_kls.append((p.exp() * (p - q)).sum().item())
    return float(np.mean(all_kls)) if all_kls else 0.0


def calibrate_bias_scalar_stepkl(
        model, tokenizer, prompts: Sequence[str],
        continuation_ids: Sequence[Sequence[int]], token_ids: Sequence[int],
        base_delta: torch.Tensor, target_kl: float, device: str = "cpu",
        lo: float = 0.0, hi: float = 8.0, iters: int = 30) -> Tuple[float, float]:
    """Bisect scalar ``c`` so mean teacher-forced per-step KL(control||unsteered)
    == ``target_kl`` (B*). Control bias vector = c * base_delta[token_ids].

    KL rises monotonically from 0 at c=0, so bisection is valid."""
    tid = torch.tensor(list(token_ids), device=device)
    dvals = base_delta[tid.cpu()].to(device)

    def kl_at(c):
        return teacher_forced_stepkl_biased(
            model, tokenizer, prompts, continuation_ids, token_ids,
            c * dvals, device=device)

    khi = kl_at(hi)
    tries = 0
    while khi < target_kl and tries < 12:
        hi *= 1.5
        khi = kl_at(hi)
        tries += 1
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        k = kl_at(mid)
        if k < target_kl:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    return c, kl_at(c)


def first_token_flip_count(method: "A0Method", tokenizer, prompts: Sequence[str],
                           alpha: float) -> Tuple[int, int]:
    """Count how many prompts have their FIRST generated token's argmax changed
    by the steering push (alpha, position-1 application) vs unsteered baseline.

    One forward pass per prompt per condition. Returns (n_flips, n_prompts).
    Amendment 1 §3 mechanism verification (expected ~0)."""
    from actlib.patching import _dynamic_patch_hook
    model = method.model
    device = method.device
    layer = method.layer
    hidden = method.u.shape[0]
    addv = method.add_vector(alpha)
    rb = torch.zeros(0, hidden, device=device)
    flips = 0
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        last = input_ids.shape[1] - 1
        with torch.no_grad():
            base_arg = int(model(input_ids).logits[0, -1].argmax())
        state = {"cache_offset": 0, "targets": [last]}
        with torch.no_grad(), _dynamic_patch_hook(
                model, "resid_post", layer, None, None, "subspace_transplant",
                state, remove_subspace=rb, add_vector=addv):
            steer_arg = int(model(input_ids).logits[0, -1].argmax())
        if steer_arg != base_arg:
            flips += 1
    return flips, len(prompts)


class LogitBiasProcessor:
    """HF LogitsProcessor adding a fixed bias to a token set at every step."""

    def __init__(self, token_ids: Sequence[int], bias_vals: torch.Tensor):
        self.tid = torch.tensor(list(token_ids), dtype=torch.long)
        self.bias = bias_vals.detach().float()

    def __call__(self, input_ids, scores):
        b = self.bias.to(scores.device)
        tid = self.tid.to(scores.device)
        scores = scores.clone()
        scores[:, tid] = scores[:, tid] + b
        return scores


def control_generate(model, tokenizer, prompt: str, processor: LogitBiasProcessor,
                     max_new_tokens: int = 64, device: str = "cpu") -> str:
    from transformers import LogitsProcessorList
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1,
                             logits_processor=LogitsProcessorList([processor]),
                             pad_token_id=tokenizer.eos_token_id)
    cont_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Bootstrap CIs (§5)
# ---------------------------------------------------------------------------

def bootstrap_rate_ci(hits: Sequence[int], n_boot: int = 10000,
                      seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap 95% CI of a mean (rate) over per-prompt indicators."""
    arr = np.asarray(hits, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def bootstrap_ratio_ci(num_hits: Sequence[int], den_hits: Sequence[int],
                       base_hits: Sequence[int], n_boot: int = 10000,
                       seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap CI of an effect ratio E(num)/E(den), where each E is a
    condition-topic-rate MINUS the shared baseline rate.

    All three arrays are per-prompt indicators over the SAME eval prompts, so
    resampling is paired (same prompt indices across conditions) — the correct
    paired bootstrap for a within-prompt ratio.
    """
    num = np.asarray(num_hits, dtype=float)
    den = np.asarray(den_hits, dtype=float)
    base = np.asarray(base_hits, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(num)
    ratios = []
    point = (num.mean() - base.mean()) / max(den.mean() - base.mean(), 1e-9)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        en = num[idx].mean() - base[idx].mean()
        ed = den[idx].mean() - base[idx].mean()
        if abs(ed) < 1e-9:
            continue
        ratios.append(en / ed)
    ratios = np.asarray(ratios)
    if ratios.size == 0:
        # Degenerate: every resample's denominator effect was ~0 (e.g. a tiny
        # smoke setup where E(den) is essentially the baseline). No meaningful
        # ratio CI; return the point with a nan CI so callers/reports flag it.
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(ratios, 2.5)), \
        float(np.percentile(ratios, 97.5))
