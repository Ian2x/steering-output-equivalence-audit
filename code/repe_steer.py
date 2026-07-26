"""RepE reading-vector steering helpers — Zou et al. 2310.01405 ("Representation
Engineering"; the LAT / reading-vector method).

Faithful RepE reading-vector steering on a chat model (Qwen2.5-7B-Instruct),
sycophancy behavior, under the frozen pre-registered battery (plan.md §2-5, §8,
§11; Amendments 1-3). See REPE_DESIGN.md for the full design rationale and the
FLAGGED design decisions (D-1..D-7) the lead must resolve before launch.

Method (faithful RepE reading vector)
-------------------------------------
RepE extracts a *reading vector* via Linear Artificial Tomography (LAT): collect
residual-stream reps over a contrastive stimulus set, form the paired
differences, run PCA, and take the TOP principal component as the reading
direction v_hat_L. Steering ("reading-vector control" / RepControl) then ADDS
c * v_hat_L to the residual stream -- mechanically the SAME additive residual-
stream steer as CAA (Rimsky). The ONLY substantive difference from the CAA arm is
the DIRECTION EXTRACTION: CAA uses the mean difference of the contrastive reps;
RepE uses the top PCA component (LAT). Everything downstream -- the additive
resid_post hook, base/native/first modes, KV-baked E_first, the battery, the
control family, the verdict machinery -- is REUSED from the CAA arm so CAA and
RepE differ ONLY in mean-diff-vs-PCA.

Extraction (LAT), per layer L (`read_vectors_pca`):
  1. Over the SAME Rimsky sycophancy A/B pairs CAA uses, read resid_post at the
     IDENTICAL site/token CAA reads (the last token of the chat prompt ending in
     the committed answer letter "(A"/"(B" -- the answer-letter token) for both
     the sycophantic and the non-sycophantic letter.
  2. Form the paired differences d_i = resid(sycophantic) - resid(non_syc)
     (LAT's contrastive difference construction; flag D-6 covers stacked-reps as
     an alternative).
  3. (Optionally, flag D-5) mean-center the differences, then run PCA (SVD) and
     take the TOP principal component as the reading direction.
  4. SIGN-ALIGN v_hat so projecting the reps onto it correlates POSITIVELY with
     the sycophancy label (the reading vector must push TOWARD sycophancy-
     positive -- a flipped sign spuriously fails the +25 gate). We align by the
     sign of mean(d_i . v_hat): the difference vectors point syc-minus-nonsyc, so
     a positive mean projection means +v_hat is the pro-sycophancy direction.
  5. Unit-normalize v_hat.

Steering = ADD c * v_hat at resid_post of layer L (coeff = added-vector norm,
exactly like CAA's c * v_hat). Two injection patterns (flag D-1, `--inject-layers`):
  - single (default): add at ONE chosen layer, a clean drop-in CAA contrast
    (E_native = E_all at that layer, kappa machinery identical to CAA).
  - all : add each layer's OWN reading vector at EVERY swept layer simultaneously
    (more faithful to Zou et al., but a DIFFERENT injection pattern that muddies
    the CAA contrast and complicates E_first/kappa).

Native regime is all-positions (like CAA / refusal / SAE), so kappa = E_first /
E_native is informative:
  - E_native : c*v_hat added at EVERY position (RepControl deployment).
  - E_first  : c*v_hat added only while processing prompt + first generated token,
               baked into the KV cache, then removed.
  - kappa    : E_first / E_native (cascade share).

Sycophancy behavior + dataset + classifier + eval prompts are REUSED VERBATIM
from the CAA arm (imported from caa_steer): the audit's controlled variable is the
direction-DERIVATION (mean-diff vs PCA vs per-head-probe), so behavior + model +
injection-family are held fixed across CAA/RepE/ITI for a clean three-way
contrast. RepE's reading-vector extraction is behavior-agnostic (PCA on any
contrastive stimulus set), so sycophancy is a legitimate application.
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import battery as B  # noqa: E402
# Reuse the CAA sycophancy classifier, chat builder, W_U geometry, and A/B
# extraction conventions VERBATIM. RepE differs from CAA ONLY in the direction
# extraction (read_vectors_pca below); everything else is CAA's, imported.
import caa_steer as C  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from actlib.capture import capture_hooks, get_blocks  # noqa: E402


# ---------------------------------------------------------------------------
# Chat prompt construction + sycophancy classifier (reuse CAA verbatim)
# ---------------------------------------------------------------------------
# Re-export CAA's helpers under this module so run_repe.py can call C.* or the
# local names interchangeably and a reader sees the reuse explicitly.
build_chat = C.build_chat
is_sycophantic = C.is_sycophantic
sycophancy_rate = C.sycophancy_rate
sycophancy_token_ids = C.sycophancy_token_ids
wu_span_basis = C.wu_span_basis
cos_dir_wu_span = C.cos_dir_wu_span
SYCOPHANCY_TOKENS = C.SYCOPHANCY_TOKENS


# ---------------------------------------------------------------------------
# RepE reading-vector extraction (LAT: PCA top component over contrastive reps)
# ---------------------------------------------------------------------------
# Read site/token IDENTICAL to CAA's caa_vector: resid_post at the LAST token of
# the chat prompt ending in the committed answer letter. We reuse CAA's private
# reader so CAA and RepE read the EXACT same activation; RepE only replaces the
# mean over the differences with the top PCA component.

_last_token_resid_post = C._last_token_resid_post


def _extraction_reps(model, tokenizer, pairs: Sequence[dict], layer: int,
                     device: str, log_every: int = 50) -> Tuple[torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor]:
    """Collect the contrastive reps CAA reads, at ``layer``, at the answer-letter
    token. Returns (H_syc, H_non, diffs), each [n_pairs, hidden] CPU f32, where
    diffs = H_syc - H_non (the LAT paired differences, syc-minus-nonsyc)."""
    h_syc, h_non = [], []
    for i, p in enumerate(pairs):
        q = p["question"]
        syc = p["answer_matching"]  # 'A' or 'B'
        non = "B" if syc == "A" else "A"
        user = f"{q}\n\n(A) {p['text_A']}\n(B) {p['text_B']}"
        prompt_syc = C.build_chat(tokenizer, user, assistant_prefix=f"({syc}")
        prompt_non = C.build_chat(tokenizer, user, assistant_prefix=f"({non}")
        hs = _last_token_resid_post(model, tokenizer, prompt_syc, layer, device)
        hn = _last_token_resid_post(model, tokenizer, prompt_non, layer, device)
        h_syc.append(hs)
        h_non.append(hn)
        if log_every and (i + 1) % log_every == 0:
            print(f"[repe reps] L={layer} {i+1}/{len(pairs)} pairs", flush=True)
    H_syc = torch.stack(h_syc)
    H_non = torch.stack(h_non)
    return H_syc, H_non, (H_syc - H_non)


def read_vectors_pca(model, tokenizer, pairs: Sequence[dict], layer: int,
                     device: str = "cpu", n_components: int = 1,
                     mean_center: bool = True, log_every: int = 50) -> dict:
    """LAT reading vector at ``layer`` (Zou et al. 2310.01405).

    Faithful LAT: read resid_post at the answer-letter token over the contrastive
    sycophancy A/B pairs (IDENTICAL site/token to CAA's caa_vector), form the
    paired differences d_i = resid(syc) - resid(non_syc), (optionally mean-center)
    and take the TOP principal component (n_components=1) as the reading direction
    v_hat. Sign-align so +v_hat is the pro-sycophancy direction, then unit-
    normalize.

    ``pairs`` schema is CAA's: {"question", "answer_matching" ('A'|'B'),
    "text_A", "text_B"} with answer_matching = the SYCOPHANTIC letter.

    Returns the SAME dict shape as C.caa_vector so run_repe drops it straight into
    the CAA stage1/stage2 flow: {"layer", "v_hat", "raw_norm", "n_pairs", ...}.
    ``raw_norm`` here = the mean paired-difference norm (the natural analogue of
    CAA's raw diff norm; the PCA direction is unit by construction, and the
    coefficient c is the added-vector norm exactly as in CAA -- v_hat is unit in
    both arms so the coeff sweep is directly comparable).
    """
    H_syc, H_non, diffs = _extraction_reps(model, tokenizer, pairs, layer,
                                           device, log_every=log_every)
    n = diffs.shape[0]
    X = diffs.clone()
    diff_mean = diffs.mean(0)
    if mean_center:
        X = X - diff_mean
    # PCA via SVD on the (centered) paired differences. Rows are samples; the top
    # right-singular vector is the top principal component (leading rep direction).
    # SVD is deterministic up to sign; we fix the sign below by the label.
    try:
        U, S, Vh = torch.linalg.svd(X.float(), full_matrices=False)
        comps = Vh  # [k, hidden], rows = principal components (unit)
        svals = S
    except Exception:  # pragma: no cover - numerical fallback
        # Fall back to the covariance eigen-decomposition if SVD fails.
        cov = X.float().t() @ X.float() / max(n - 1, 1)
        evals, evecs = torch.linalg.eigh(cov)
        order = torch.argsort(evals, descending=True)
        comps = evecs[:, order].t()
        svals = evals[order].clamp_min(0).sqrt()
    top = comps[0]  # [hidden]
    # Sign-align: +v_hat must push TOWARD sycophancy-positive. The difference
    # vectors point syc-minus-nonsyc, so align to the sign of their mean
    # projection onto the component (equivalently the sign of diff_mean . top).
    proj_on_diffs = (diffs.float() @ top)          # [n]
    align = float(proj_on_diffs.mean())
    if align < 0:
        top = -top
        proj_on_diffs = -proj_on_diffs
    v_hat = top / (top.norm() + 1e-12)
    # Diagnostics for REPE_DESIGN sign-alignment audit + geometry.
    mean_diff_dir = diff_mean / (diff_mean.norm() + 1e-12)
    cos_to_caa = float(torch.dot(v_hat.float(), mean_diff_dir.float()))
    total_var = float((svals ** 2).sum().item())
    top_var_frac = (float((svals[0] ** 2).item()) / total_var
                    if total_var > 0 else float("nan"))
    label_corr = float(np.sign(proj_on_diffs.numpy()).mean())  # frac aligned
    return {
        "layer": layer,
        "v_hat": v_hat,
        "raw_norm": float(diffs.mean(0).norm()),  # mean paired-diff norm
        "n_pairs": n,
        "n_components": n_components,
        "mean_center": bool(mean_center),
        "top_var_frac": top_var_frac,
        "cos_to_meandiff": cos_to_caa,     # PCA-dir vs CAA mean-diff direction
        "sign_align_mean_proj": align,     # pre-flip mean projection (audit)
        "label_frac_aligned": label_corr,  # frac of diffs on the +v_hat side
    }


# ---------------------------------------------------------------------------
# RepE steering method: add c*v_hat at resid_post, single- or multi-layer
# ---------------------------------------------------------------------------
# For inject_layers="single" this is a byte-for-byte behavioral copy of CAAMethod
# (one layer, one vector) so the CAA-vs-RepE contrast differs ONLY in the
# direction. For inject_layers="all" it adds each layer's OWN reading vector at
# every listed layer (Zou et al.'s multi-layer RepControl). kappa is only
# CAA-comparable in the single-layer mode (flag D-1).

@contextmanager
def _multi_residpost_static_hook(model, layer_vecs: Dict[int, torch.Tensor]):
    """resid_post forward hooks adding layer_vecs[L] at EVERY position, on each
    listed layer, for a SINGLE forward pass (mirrors C._residpost_static_hook but
    over a set of layers)."""
    blocks = get_blocks(model)
    with ExitStack() as stack:
        for L, add_vec in layer_vecs.items():
            module = blocks[L]

            def make_hook(v):
                def hook(mod, args, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                        return (hs + v.to(hs.dtype).to(hs.device),) \
                            + tuple(output[1:])
                    return output + v.to(output.dtype).to(output.device)
                return hook
            stack.enter_context(_hook_ctx(module, make_hook(add_vec)))
        yield


@contextmanager
def _hook_ctx(module, hook):
    h = module.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@dataclass
class RepEMethod:
    """Add ``coeff * v_hat`` at resid_post at every position.

    Two injection patterns (``inject_layers``):
      - 'single': add at ONE layer (``layer`` / ``directions[layer]``). Byte-for-
        byte the CAA injection -> clean CAA contrast, kappa comparable.
      - 'all'   : add EACH layer's own reading vector at EVERY layer in
        ``directions`` simultaneously (Zou et al. multi-layer RepControl).

    Modes (identical to CAAMethod):
      - 'native'/'all': add at every position (RepControl deployment).
      - 'first'       : add through prompt + FIRST generated token (KV-baked),
                        then removed (E_first).
      - 'base'        : no steering (greedy).
    Manual generation loop (like CAAMethod) so E_first can stop after step 0.
    ``directions`` maps layer -> unit reading vector (CPU). For single mode only
    ``directions[layer]`` is used; ``layer`` is the reported / kappa layer.
    """
    model: object
    tokenizer: object
    layer: int
    directions: Dict[int, torch.Tensor]   # layer -> [hidden] unit vector, CPU
    device: str = "cpu"
    max_new_tokens: int = 64
    inject_layers: str = "single"         # "single" | "all"

    @property
    def direction(self) -> torch.Tensor:
        """The single-layer reading vector (kappa layer). Mirrors CAAMethod."""
        return self.directions[self.layer]

    def _active_layers(self) -> List[int]:
        if self.inject_layers == "all":
            return sorted(self.directions.keys())
        return [self.layer]

    def _vec(self, coeff: float) -> torch.Tensor:
        """Single-layer add vector at the kappa layer (parity with CAAMethod)."""
        return (coeff * self.direction).to(self.device)

    def _layer_vecs(self, coeff: float) -> Dict[int, torch.Tensor]:
        """coeff * v_hat per active layer (single -> one entry)."""
        return {L: (coeff * self.directions[L]).to(self.device)
                for L in self._active_layers()}

    def generate(self, prompt: str, coeff: float, mode: str) -> str:
        if mode == "base":
            return self._gen(prompt, None, apply="none")
        if mode in ("native", "all"):
            return self._gen(prompt, self._layer_vecs(coeff), apply="all")
        if mode == "first":
            return self._gen(prompt, self._layer_vecs(coeff), apply="first")
        raise ValueError(mode)

    def generate_with_fixed_vector(self, prompt: str, vec: torch.Tensor,
                                   mode: str) -> str:
        """Generate adding a fixed [hidden] ``vec`` (floor / alt-direction
        control). For inject_layers='all' the SAME fixed vector is added at every
        active layer (matched to the multi-layer deployment); for 'single' at the
        one layer. Mirrors CAAMethod.generate_with_fixed_vector."""
        apply = "all" if mode in ("native", "all") else "first"
        layer_vecs = {L: vec.to(self.device) for L in self._active_layers()}
        return self._gen(prompt, layer_vecs, apply=apply)

    def _gen(self, prompt: str, layer_vecs, apply: str) -> str:
        model, tok, device = self.model, self.tokenizer, self.device
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        blocks = get_blocks(model)
        active_layers = self._active_layers()
        state = {"active": apply != "none"}

        def make_hook(L):
            def hook(mod, args, output):
                if not state["active"] or layer_vecs is None:
                    return output
                add_vec = layer_vecs.get(L)
                if add_vec is None:
                    return output
                if isinstance(output, tuple):
                    hs = output[0]
                    return (hs + add_vec.to(hs.dtype),) + tuple(output[1:])
                return output + add_vec.to(output.dtype)
            return hook

        handles = [blocks[L].register_forward_hook(make_hook(L))
                   for L in active_layers]
        generated: List[int] = []
        try:
            with torch.no_grad():
                state["active"] = apply != "none"
                out = model(input_ids, use_cache=True)
                past = out.past_key_values
                nid = int(out.logits[0, -1].argmax())
                generated.append(nid)
                cur = torch.tensor([[nid]], device=device)
                for step in range(self.max_new_tokens - 1):
                    if apply == "first" and step == 0:
                        state["active"] = True
                    elif apply == "first":
                        state["active"] = False
                    elif apply == "all":
                        state["active"] = True
                    else:
                        state["active"] = False
                    if nid == tok.eos_token_id:
                        break
                    o = model(cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nid = int(o.logits[0, -1].argmax())
                    generated.append(nid)
                    cur = torch.tensor([[nid]], device=device)
        finally:
            for h in handles:
                h.remove()
        return tok.decode(torch.tensor(generated), skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Mechanism / control helpers (mirror CAA versions; dispatch over active layers)
# ---------------------------------------------------------------------------
# For inject_layers="single" each of these is behaviorally identical to the CAA
# helper (one layer). For "all" they apply the per-layer reading vectors on every
# active layer in the SAME single forward pass, so pos-1 deltas / TF-KL / flips
# reflect the multi-layer deployment.

def position1_logit_delta(meth: RepEMethod, tok, prompts, coeff) -> torch.Tensor:
    """Mean position-1 logit delta (native all-position RepE add - baseline)."""
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(coeff)
    deltas = []
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            base = model(ids).logits[0, -1]
        with torch.no_grad(), _multi_residpost_static_hook(model, layer_vecs):
            steer = model(ids).logits[0, -1]
        deltas.append((steer - base).to("cpu"))
    return torch.stack(deltas).mean(0)


def first_token_flip_count(meth: RepEMethod, tok, prompts, coeff):
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(coeff)
    flips = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            base_arg = int(model(ids).logits[0, -1].argmax())
        with torch.no_grad(), _multi_residpost_static_hook(model, layer_vecs):
            steer_arg = int(model(ids).logits[0, -1].argmax())
        if steer_arg != base_arg:
            flips += 1
    return flips, len(prompts)


def teacher_forced_stepkl_native(meth: RepEMethod, tok, prompts,
                                 continuation_ids, coeff) -> float:
    """Mean teacher-forced per-step KL(E_native || unsteered) over continuation
    positions x prompts. E_native adds c*v at EVERY position (each active layer)."""
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(coeff)
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
        with torch.no_grad(), _multi_residpost_static_hook(model, layer_vecs):
            steer = model(full).logits[0]
        for pos in pred:
            p = torch.log_softmax(steer[pos], dim=-1)
            q = torch.log_softmax(base[pos], dim=-1)
            all_kls.append((p.exp() * (p - q)).sum().item())
    return float(np.mean(all_kls)) if all_kls else 0.0


def kv_baked_first_sanity(meth: RepEMethod, tok, prompts, coeff) -> dict:
    """Verify E_first: manual prefill-with-steering then continue-without matches
    the method's 'first' output (steering baked into KV for prompt+first token).
    Mirrors C.kv_baked_first_sanity; applies the shift on all active layers."""
    model, device = meth.model, meth.device
    layer_vecs = meth._layer_vecs(coeff)
    results = []
    for p in prompts:
        native_first = meth.generate(p, coeff, "first")
        ids = tok(p, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad(), _multi_residpost_static_hook(model, layer_vecs):
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
                    with _multi_residpost_static_hook(model, layer_vecs):
                        o = model(cur, past_key_values=past, use_cache=True)
                else:
                    o = model(cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                nid = int(o.logits[0, -1].argmax())
                gen.append(nid)
                cur = torch.tensor([[nid]], device=device)
        manual = tok.decode(torch.tensor(gen), skip_special_tokens=True)
        results.append(native_first.strip() == manual.strip())
    return {"n": len(prompts), "all_match": bool(all(results)), "matches": results}
