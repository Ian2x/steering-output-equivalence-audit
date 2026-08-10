"""SAE feature-steering helpers for the steering-content-audit SAE arm.

Faithful Templeton-style SAE feature steering on GPT-2-small with a RELEASED
open SAE (Joseph Bloom `gpt2-small-res-jb`, blocks.7.hook_resid_pre; 24576
features, d_in=768, no activation normalization, decoder rows already unit-norm).

Method
------
Steering = add ``c * W_dec[f]`` (unit decoder direction of feature ``f``) to the
residual stream at the SAE's layer/site (resid_pre of layer 7), at every position
(prompt + each generated token) — the native all-position deployment of SAE
feature steering. The "clamp feature to k x max" variant is an equivalent scaled
add; we use the scaled-add form and report ``c``. Because the native regime is
all-positions, ``kappa = E_first / E_native`` is informative (as for refusal).
The standardized ``first`` mode is prefill-only; the shipped prefill+1 schedule
remains available explicitly for window-sensitivity runs:

  - E_native = c*W_dec[f] added at EVERY position (native SAE-steering form).
  - E_first  = c*W_dec[f] added only while processing the prompt (prefill-only).
  - E_first_prefill_plus1 = the historical schedule, which also applies while
                            processing the first generated token.
  - kappa    = E_first / E_native (cascade share).

Feature discovery is programmatic (no Neuronpedia): encode concept-bearing vs
neutral texts at the SAE site, pick the feature with the largest mean activation
gap (concept - neutral), and report its top max-activating tokens as evidence.

All hooks operate on actlib's resid_pre site of the SAE layer (numerically the
same tensor TransformerLens exposes as blocks.L.hook_resid_pre — confirmed by the
>0.9 reconstruction cosine gate). No TransformerLens model is used for generation;
we drive the plain HF gpt2 model exactly as the other arms do.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import battery as B  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from actlib.capture import capture_hooks, get_blocks  # noqa: E402


# ---------------------------------------------------------------------------
# SAE loading + reconstruction gate
# ---------------------------------------------------------------------------

def load_sae(release: str, sae_id: str, device: str = "cpu"):
    """Load a released SAE via sae_lens; return (sae, meta dict). Decoder rows
    for this family are unit-norm, so W_dec[f] is already the unit direction."""
    from sae_lens import SAE
    sae = SAE.from_pretrained(release, sae_id, device=device)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(torch.float32)
    cfg = sae.cfg
    meta = {
        "release": release, "sae_id": sae_id,
        "d_in": int(cfg.d_in), "d_sae": int(cfg.d_sae),
        "normalize_activations": str(cfg.normalize_activations),
        "apply_b_dec_to_input": bool(cfg.apply_b_dec_to_input),
        "hook_name": getattr(getattr(cfg, "metadata", None), "hook_name", sae_id),
        "wdec_row_norm_mean": float(sae.W_dec.detach().norm(dim=-1).mean()),
    }
    return sae, meta


def sae_layer_from_hook(hook_name: str) -> Tuple[int, str]:
    """Map a TransformerLens hook name 'blocks.L.hook_resid_pre' -> (L, site).
    Only resid_pre / resid_post supported (the SAE families we use)."""
    parts = hook_name.split(".")
    L = int(parts[1])
    if "resid_pre" in hook_name:
        return L, "resid_pre"
    if "resid_post" in hook_name:
        return L, "resid_post"
    raise ValueError(f"unsupported hook site: {hook_name}")


def capture_resid(model, tok, prompt: str, layer: int, site: str,
                  device: str = "cpu") -> torch.Tensor:
    """resid at (layer, site) for a single prompt -> [seq, hidden] on CPU float."""
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    store: dict = {}
    with torch.no_grad(), capture_hooks(model, [site], [layer], store):
        model(ids)
    return store[(layer, site)][-1][0].to("cpu").float()


def reconstruction_cosine(sae, model, tok, prompts: Sequence[str], layer: int,
                          site: str, device: str = "cpu",
                          skip_pos0: bool = True) -> dict:
    """Mean per-token reconstruction cosine (resid vs sae.decode(sae.encode(resid))).
    GPT-2's position-0 attention-sink token has a huge-norm residual the SAE does
    not reconstruct; skip_pos0 excludes it (report both)."""
    cos_all, cos_skip = [], []
    for p in prompts:
        resid = capture_resid(model, tok, p, layer, site, device)
        with torch.no_grad():
            recon = sae.decode(sae.encode(resid.to(device))).to("cpu")
        c = torch.nn.functional.cosine_similarity(resid, recon, dim=-1)
        cos_all.extend(c.tolist())
        cos_skip.extend(c[1:].tolist())
    return {
        "mean_cosine_all_positions": float(np.mean(cos_all)),
        "mean_cosine_skip_pos0": float(np.mean(cos_skip)),
        "n_tokens": len(cos_all),
    }


# ---------------------------------------------------------------------------
# Feature discovery (programmatic, no Neuronpedia)
# ---------------------------------------------------------------------------

def mean_feature_activation(sae, model, tok, texts: Sequence[str], layer: int,
                            site: str, device: str = "cpu",
                            skip_pos0: bool = True) -> torch.Tensor:
    """Mean SAE feature activation vector over all (non-pos0) tokens of ``texts``.
    Returns [d_sae] on CPU."""
    accs = []
    for t in texts:
        resid = capture_resid(model, tok, t, layer, site, device)
        with torch.no_grad():
            f = sae.encode(resid.to(device)).to("cpu")
        f = f[1:] if (skip_pos0 and f.shape[0] > 1) else f
        accs.append(f.mean(0))
    return torch.stack(accs).mean(0)


def discover_concept_feature(sae, model, tok, concept_texts, neutral_texts,
                             layer, site, device="cpu", topk=8) -> dict:
    """Pick the feature with the largest mean activation gap (concept - neutral).
    Returns the winner id plus the ranked shortlist with per-set activations."""
    mc = mean_feature_activation(sae, model, tok, concept_texts, layer, site, device)
    mn = mean_feature_activation(sae, model, tok, neutral_texts, layer, site, device)
    gap = mc - mn
    order = torch.argsort(gap, descending=True)[:topk].tolist()
    shortlist = [{
        "feature": int(i), "gap": float(gap[i]),
        "concept_act": float(mc[i]), "neutral_act": float(mn[i]),
    } for i in order]
    return {"winner": int(order[0]), "shortlist": shortlist}


def max_activating_tokens(sae, model, tok, feature: int, corpus: Sequence[str],
                          layer: int, site: str, device: str = "cpu",
                          topn: int = 15) -> List[dict]:
    """Top-activating (token, activation) pairs for ``feature`` over ``corpus``
    (skipping position 0). Evidence the feature encodes the concept."""
    rows = []
    for t in corpus:
        ids = tok(t, return_tensors="pt")["input_ids"][0]
        resid = capture_resid(model, tok, t, layer, site, device)
        with torch.no_grad():
            f = sae.encode(resid.to(device)).to("cpu")[:, feature]
        for j in range(1, len(ids)):
            rows.append({"act": float(f[j]), "token": tok.decode([int(ids[j])])})
    rows.sort(key=lambda r: r["act"], reverse=True)
    # dedup by token string keeping the max
    seen, out = set(), []
    for r in rows:
        if r["token"] in seen:
            continue
        seen.add(r["token"])
        out.append(r)
        if len(out) >= topn:
            break
    return out


# ---------------------------------------------------------------------------
# SAE feature steering method (add c*W_dec[f] at resid_pre of the SAE layer)
# ---------------------------------------------------------------------------

@dataclass
class SAESteerMethod:
    """Add ``coeff * dir`` (dir = unit decoder direction W_dec[f]) at resid_pre of
    ``layer``. Native regime = ALL positions (prompt + every generated token).

    Modes:
      - 'native'/'all': add at every position (the published SAE-steering form).
      - 'first'       : use the explicitly selected ``first_window``.
      - 'first_prefill_only': add only during prompt prefill.
      - 'first_prefill_plus1': historical schedule; also add while processing
                               the first generated token.
      - 'base'        : no steering.
    The hook is a resid_pre forward-pre hook driven by a manual generation loop
    so the intervention window is explicit.
    """
    model: object
    tokenizer: object
    layer: int
    direction: torch.Tensor  # [hidden] unit vector on CPU
    first_window: str
    device: str = "cpu"
    max_new_tokens: int = 64

    def __post_init__(self):
        if self.first_window not in {"prefill_only", "prefill_plus1"}:
            raise ValueError(
                "first_window must be 'prefill_only' or 'prefill_plus1'")
        self.last_forward_activity = []

    def _first_apply(self, mode: str) -> str:
        if mode == "first":
            return self.first_window
        if mode == "first_prefill_only":
            return "prefill_only"
        if mode == "first_prefill_plus1":
            return "prefill_plus1"
        raise ValueError(mode)

    def _vec(self, coeff: float) -> torch.Tensor:
        return (coeff * self.direction).to(self.device)

    def generate(self, prompt: str, coeff: float, mode: str) -> str:
        if mode == "base":
            return B.base_generate(self.model, self.tokenizer, prompt,
                                   self.max_new_tokens, self.device)
        if mode in ("native", "all"):
            return self._gen(prompt, self._vec(coeff), apply="all")
        if mode.startswith("first"):
            return self._gen(prompt, self._vec(coeff),
                             apply=self._first_apply(mode))
        raise ValueError(mode)

    def generate_with_fixed_vector(self, prompt: str, vec: torch.Tensor,
                                   mode: str) -> str:
        apply = ("all" if mode in ("native", "all")
                 else self._first_apply(mode))
        return self._gen(prompt, vec.to(self.device), apply=apply)

    def _gen(self, prompt: str, add_vec: torch.Tensor, apply: str) -> str:
        model, tok, device, L = (self.model, self.tokenizer, self.device,
                                 self.layer)
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        module = get_blocks(model)[L]
        # state: absolute index of the first token in the current forward, plus a
        # flag for whether steering is active in the selected window.
        state = {"offset": 0, "active": True}

        def pre_hook(mod, args):
            if not state["active"]:
                return None
            hs = args[0]
            out = hs.clone()
            out = out + add_vec.to(hs.dtype)  # broadcast add at every column
            return (out,) + tuple(args[1:])

        h = module.register_forward_pre_hook(pre_hook)
        generated: List[int] = []
        self.last_forward_activity = []
        try:
            with torch.no_grad():
                # prefill (prompt): steering ON for both modes
                state["offset"] = 0
                state["active"] = True
                self.last_forward_activity.append({
                    "phase": "prefill", "active": True
                })
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
                nid = int(out.logits[0, -1].argmax())
                generated.append(nid)
                cur = torch.tensor([[nid]], device=device)
                # The first token was selected from prefill logits above. The
                # selected window controls whether its processing forward is
                # intervened (prefill+1) or clean (prefill-only).
                for step in range(self.max_new_tokens - 1):
                    if apply == "prefill_plus1":
                        state["active"] = step == 0
                    elif apply == "prefill_only":
                        state["active"] = False
                    else:
                        state["active"] = True
                    state["offset"] = prompt_len + step
                    self.last_forward_activity.append({
                        "phase": "generated_forward", "step": step,
                        "active": bool(state["active"]),
                    })
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nid = int(o.logits[0, -1].argmax())
                    generated.append(nid)
                    cur = torch.tensor([[nid]], device=device)
        finally:
            h.remove()
        return tok.decode(torch.tensor(generated), skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Mechanism / control helpers (mirror the ActAdd/refusal versions)
# ---------------------------------------------------------------------------

@contextmanager
def _residpre_static_hook(model, layer, add_vec):
    """resid_pre pre-hook adding ``add_vec`` at EVERY position (single forward)."""
    module = get_blocks(model)[layer]

    def pre_hook(mod, args):
        hs = args[0]
        return (hs + add_vec.to(hs.dtype).to(hs.device),) + tuple(args[1:])
    h = module.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        h.remove()


def position1_logit_delta(meth: SAESteerMethod, tok, prompts, coeff) -> torch.Tensor:
    """Mean position-1 logit delta (native all-position steering - baseline).
    All-position add during the prompt forward, read next-token logits at the last
    prompt position. Returns [vocab] on CPU."""
    model, device, L = meth.model, meth.device, meth.layer
    add_vec = meth._vec(coeff)
    deltas = []
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            base = model(ids).logits[0, -1]
        with torch.no_grad(), _residpre_static_hook(model, L, add_vec):
            steer = model(ids).logits[0, -1]
        deltas.append((steer - base).to("cpu"))
    return torch.stack(deltas).mean(0)


def first_token_flip_count(meth: SAESteerMethod, tok, prompts, coeff):
    """Count prompts whose first generated token's argmax flips under native
    steering vs baseline."""
    model, device, L = meth.model, meth.device, meth.layer
    add_vec = meth._vec(coeff)
    flips = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            base_arg = int(model(ids).logits[0, -1].argmax())
        with torch.no_grad(), _residpre_static_hook(model, L, add_vec):
            steer_arg = int(model(ids).logits[0, -1].argmax())
        if steer_arg != base_arg:
            flips += 1
    return flips, len(prompts)


def teacher_forced_stepkl_native(meth: SAESteerMethod, tok, prompts,
                                 continuation_ids, coeff) -> float:
    """Mean teacher-forced per-step KL(E_native || unsteered) over all
    continuation positions x prompts. E_native adds c*dir at EVERY position; under
    teacher forcing on the fixed unsteered continuation we read KL at each position
    that predicts a continuation token (single forward per prompt)."""
    model, device, L = meth.model, meth.device, meth.layer
    add_vec = meth._vec(coeff)
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
        with torch.no_grad(), _residpre_static_hook(model, L, add_vec):
            steer = model(full).logits[0]
        for pos in pred:
            p = torch.log_softmax(steer[pos], dim=-1)
            q = torch.log_softmax(base[pos], dim=-1)
            all_kls.append((p.exp() * (p - q)).sum().item())
    return float(np.mean(all_kls)) if all_kls else 0.0


def kv_baked_first_sanity(meth: SAESteerMethod, tok, prompts, coeff,
                          first_window: str | None = None) -> dict:
    """Verify the selected E_first window against an independent manual loop."""
    model, device, L = meth.model, meth.device, meth.layer
    add_vec = meth._vec(coeff)
    window = first_window or meth.first_window
    if window not in {"prefill_only", "prefill_plus1"}:
        raise ValueError(window)
    mode = f"first_{window}"
    results = []
    for p in prompts:
        native_first = meth.generate(p, coeff, mode)
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad(), _residpre_static_hook(model, L, add_vec):
            out = model(ids, use_cache=True)
        past = out.past_key_values
        nid = int(out.logits[0, -1].argmax())
        gen = [nid]
        cur = torch.tensor([[nid]], device=device)
        with torch.no_grad():
            for step in range(meth.max_new_tokens - 1):
                if window == "prefill_plus1" and step == 0:
                    with _residpre_static_hook(model, L, add_vec):
                        o = model(cur, past_key_values=past, use_cache=True)
                else:
                    o = model(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nid = int(o.logits[0, -1].argmax())
                gen.append(nid)
                cur = torch.tensor([[nid]], device=device)
        manual = tok.decode(torch.tensor(gen), skip_special_tokens=True)
        results.append(native_first.strip() == manual.strip())
    return {
        "window": window, "n": len(prompts),
        "all_match": bool(all(results)), "matches": results,
    }
