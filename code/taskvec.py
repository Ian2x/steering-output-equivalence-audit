"""In-context task-vector extraction + REPLACE injection (Hendel et al.
2310.15916, "In-Context Learning Creates Task Vectors" — the single-residual-
vector variant), for the steering-content-audit task-vectors arm.

Distinct from the A1 function-vectors arm (Todd et al., sum of top-CIE attention
heads, ADD at layer L/3). Hendel's method:

  1. Task vector `theta` = MEAN over N>=100 ICL demonstration contexts of the
     FULL residual-stream activation (`resid_post` at layer L) at the LAST token
     of the demonstration context — the separator/"->" position immediately
     before the query answer would be produced. In the Todd ICL template used by
     A1 (which we reuse for split discipline) the ICL prompt ends with the
     query's answer-prefix `A:`, so the "last demo-context token" is the final
     token of the full ICL prompt (the `:` of `A:`), i.e. exactly the position
     whose next-token prediction is the query answer.
  2. Injection = REPLACE (Hendel patches, not adds): overwrite `resid_post` at
     layer L at the zero-shot query's FINAL token position with `theta`. The
     zero-shot prompt has NO demonstrations ("Q: {x}\nA:").

Layer sweep L in {8,12,14,16,18,20} of 32; reproduction gate = zero-shot
accuracy gain >= +25 pts at some L (Hendel report medium-high recovery). REPLACE
is primary; ADD is a sensitivity fallback reported if REPLACE fails.

This module supplies:
  - `mean_task_vector` : theta at a layer over ICL demo contexts.
  - `TaskVecMethod`     : REPLACE (or ADD) injection with actlib KV-baked E_first
                          semantics, mirroring battery.A0Method / FVMethod so the
                          whole battery (TF-KL control calib, kappa, degeneracy,
                          bootstrap, floor) is reused verbatim.

All hooks go through the same resid_post site the battery patches.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import List, Sequence

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_TOOLS = os.path.join(_REPO, "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

import fv_extract as FV  # noqa: E402  (ICL template, dataset splits)
from actlib.capture import capture_hooks, get_blocks  # noqa: E402
from actlib.patching import _resolve_target_module     # noqa: E402


# ---------------------------------------------------------------------------
# Task-vector extraction: mean resid_post at layer L, last ICL-context token
# ---------------------------------------------------------------------------

def resid_post_last_token(model, tokenizer, prompt: str, layer: int,
                          device: str = "cpu") -> torch.Tensor:
    """resid_post[layer] at the LAST token of ``prompt`` -> [hidden] (cpu float32)."""
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    store: dict = {}
    with torch.no_grad(), capture_hooks(model, ["resid_post"], [layer], store):
        model(ids)
    return store[(layer, "resid_post")][-1][0, -1].to("cpu").float()  # [hidden]


def mean_task_vector(model, tokenizer, icl_prompts, layer: int,
                     device: str = "cpu", log=print) -> torch.Tensor:
    """Hendel theta at ``layer`` = mean resid_post at the last token of each ICL
    demonstration context, over ``icl_prompts`` (list of dicts with 'prompt').
    Returns [hidden] (cpu float32)."""
    acc = None
    n = 0
    for i, p in enumerate(icl_prompts):
        v = resid_post_last_token(model, tokenizer, p["prompt"], layer, device)
        acc = v if acc is None else acc + v
        n += 1
        if log and (i + 1) % 25 == 0:
            log(f"    task-vec[L={layer}]: {i+1}/{len(icl_prompts)} contexts")
    return acc / max(n, 1)


# ---------------------------------------------------------------------------
# REPLACE / ADD injection at resid_post[layer], query final token
# ---------------------------------------------------------------------------

class TaskVecMethod:
    """Hendel injection of a fixed task vector ``theta`` at ``resid_post[layer]``.

    op = 'replace'  -> overwrite the target position's resid with theta (Hendel).
    op = 'add'      -> theta added to the target position's resid (sensitivity).

    Deployment regimes (position sets), matching the battery's E_all/E_first:
      - 'all'   (E_all / native for a single-position patch used every step): the
        REPLACE/ADD is applied at every generated position AND the final prompt
        token. For REPLACE this pins the resid to theta at each step.
      - 'first' (E_first, KV-baked one-shot): the patch is applied only while
        processing the prompt + first generated token, baked into the KV cache,
        then removed. This is the native regime for Hendel (a single write at the
        query final position that then propagates through the sampling loop).

    NOTE ON kappa (plan §2 I1, Amendment 3): task-vector injection is a single-
    position write at the query's final prompt token whose effect propagates
    through the model's own sampling loop — a prompt-injection-family method like
    ActAdd. So E_native = E_first (the KV-baked single write), and kappa =
    E_first/E_native ~= 1. E_all (the forced broadcast at every generated
    position) is a DIAGNOSTIC only, never the kappa denominator. We report
    kappa = E_first/E_first as the native-regime coordinate and also list the
    diagnostic E_first/E_all.
    """

    def __init__(self, model, tokenizer, layer, theta, op="replace",
                 device="cpu", max_new_tokens=8):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.theta = theta.to(device).float()
        self.op = op
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.norm = float(self.theta.norm().item())
        self._module, self._is_pre = _resolve_target_module(
            model, "resid_post", layer)

    # -- core injected-generation loop (KV cache aware) --
    def _generate(self, prompt: str, vec: torch.Tensor, op: str, regime: str
                  ) -> str:
        """Greedy generate with ``vec`` written (op) into resid_post[layer].

        regime 'all'   : write at query final prompt token AND every generated
                         position.
        regime 'first' : write only during prefill (query final token) + the
                         first generated step, baked into the KV cache, then the
                         hook is removed for the remaining steps (E_first).
        """
        model, tok, device = self.model, self.tokenizer, self.device
        v = vec.to(device)
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        prompt_len = input_ids.shape[1]
        state = {"cache_offset": 0, "targets": [prompt_len - 1], "active": True}

        def _apply(hs):
            if not state["active"]:
                return hs
            seq = hs.shape[1]
            offset = state["cache_offset"]
            cols = [p - offset for p in state["targets"]
                    if 0 <= (p - offset) < seq]
            if not cols:
                return hs
            out = hs.clone()
            vv = v.to(hs.dtype)
            for col in cols:
                if op == "replace":
                    out[:, col, :] = vv
                elif op == "add":
                    out[:, col, :] = out[:, col, :] + vv
                else:
                    raise ValueError(op)
            return out

        if self._is_pre:
            def hook(mod, args):
                return (_apply(args[0]),) + tuple(args[1:])
            handle = self._module.register_forward_pre_hook(hook)
        else:
            def hook(mod, args, output):
                if isinstance(output, tuple):
                    return (_apply(output[0]),) + tuple(output[1:])
                return _apply(output)
            handle = self._module.register_forward_hook(hook)

        generated: List[int] = []
        try:
            with torch.no_grad():
                cur = input_ids
                past = None
                for step in range(self.max_new_tokens):
                    abs_step = prompt_len - 1 + step  # position producing this tok
                    if regime == "all":
                        # write at final prompt token (step 0) and each generated
                        # position (the position that predicts the next token).
                        state["targets"] = [abs_step]
                        state["active"] = True
                    elif regime == "first":
                        # write only at the query final prompt token (prefill).
                        # After the prefill forward it is baked into the KV cache;
                        # remove the hook for all subsequent steps.
                        state["targets"] = [prompt_len - 1]
                        state["active"] = (step == 0)
                    else:
                        raise ValueError(regime)
                    if past is not None:
                        state["cache_offset"] = abs_step
                        fwd = cur[:, -1:]
                        out = model(fwd, past_key_values=past, use_cache=True)
                    else:
                        state["cache_offset"] = 0
                        out = model(cur, use_cache=True)
                    past = out.past_key_values
                    nid = int(out.logits[0, -1].argmax())
                    generated.append(nid)
                    cur = torch.cat(
                        [cur, torch.tensor([[nid]], device=device)], dim=1)
        finally:
            handle.remove()
        return tok.decode(torch.tensor(generated), skip_special_tokens=True)

    def generate(self, prompt: str, mode: str) -> str:
        """mode: 'base' | 'all' | 'first' (uses self.op = replace/add)."""
        if mode == "base":
            return base_generate(self.model, self.tokenizer, prompt,
                                 self.max_new_tokens, self.device)
        if mode == "all":
            return self._generate(prompt, self.theta, self.op, "all")
        if mode == "first":
            return self._generate(prompt, self.theta, self.op, "first")
        raise ValueError(mode)

    def generate_with_vector(self, prompt: str, vec: torch.Tensor, op: str,
                             regime: str) -> str:
        """Generate writing an arbitrary fixed ``vec`` (floor/random-dir control).
        op = 'replace'|'add'; regime = 'all'|'first'."""
        return self._generate(prompt, vec, op, regime)

    # -- position-1 logit delta at the query final token (control token-set) --
    def position1_logit_delta(self, prompts) -> torch.Tensor:
        """Mean position-1 logit delta (theta written at the query final prompt
        token) over ``prompts``. Returns [vocab] (cpu). Uses op = self.op."""
        model, tok, device = self.model, self.tokenizer, self.device
        v = self.theta.to(device)
        deltas = []
        for p in prompts:
            enc = tok(p, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            with torch.no_grad():
                base = model(ids).logits[0, -1]
            steered = self._single_forward_last_token(ids, v)
            deltas.append((steered - base).to("cpu"))
        return torch.stack(deltas).mean(0)

    def _single_forward_last_token(self, ids, v):
        """One forward with theta written (self.op) at the last prompt token;
        return last-position logits [vocab]."""
        last = ids.shape[1] - 1
        state = {"targets": [last], "active": True}

        def _apply(hs):
            out = hs.clone()
            vv = v.to(hs.dtype)
            if self.op == "replace":
                out[:, last, :] = vv
            else:
                out[:, last, :] = out[:, last, :] + vv
            return out

        if self._is_pre:
            def hook(mod, args):
                return (_apply(args[0]),) + tuple(args[1:])
            h = self._module.register_forward_pre_hook(hook)
        else:
            def hook(mod, args, output):
                if isinstance(output, tuple):
                    return (_apply(output[0]),) + tuple(output[1:])
                return _apply(output)
            h = self._module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = self.model(ids).logits[0, -1]
        finally:
            h.remove()
        return logits

    def first_token_flip_count(self, prompts):
        """Count prompts whose first generated token argmax changes under the
        single-write theta injection vs baseline. Returns (n_flips, n)."""
        flips = 0
        v = self.theta.to(self.device)
        for p in prompts:
            enc = self.tokenizer(p, return_tensors="pt")
            ids = enc["input_ids"].to(self.device)
            with torch.no_grad():
                base_arg = int(self.model(ids).logits[0, -1].argmax())
            steer = int(self._single_forward_last_token(ids, v).argmax())
            if steer != base_arg:
                flips += 1
        return flips, len(prompts)

    def teacher_forced_stepkl(self, prompts, continuation_ids) -> float:
        """Mean teacher-forced per-step KL(steered || unsteered) over all
        continuation positions x prompts, where 'steered' writes theta (self.op)
        at EVERY continuation-predicting position (the E_all patch config). This
        is the additive-family control budget B* (Amendment 1)."""
        model, tok, device = self.model, self.tokenizer, self.device
        v = self.theta.to(device)
        all_kls: List[float] = []
        for prompt, cont in zip(prompts, continuation_ids):
            if len(cont) == 0:
                continue
            p_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
            P = p_ids.shape[1]
            c_ids = torch.tensor([list(cont)], device=device)
            full = torch.cat([p_ids, c_ids], dim=1)
            n = len(cont)
            pred_positions = list(range(P - 1, P - 1 + n))
            with torch.no_grad():
                base_logits = model(full).logits[0]
            steered_logits = self._forward_write_positions(full, v, pred_positions)
            for pos in pred_positions:
                p = torch.log_softmax(steered_logits[pos], dim=-1)
                q = torch.log_softmax(base_logits[pos], dim=-1)
                all_kls.append((p.exp() * (p - q)).sum().item())
        return float(np.mean(all_kls)) if all_kls else 0.0

    def _forward_write_positions(self, full_ids, v, positions):
        """One forward writing theta (self.op) at the given absolute positions of
        the (teacher-forced) sequence. Returns logits [seq, vocab]."""
        posset = set(positions)

        def _apply(hs):
            out = hs.clone()
            vv = v.to(hs.dtype)
            for pos in posset:
                if 0 <= pos < hs.shape[1]:
                    if self.op == "replace":
                        out[:, pos, :] = vv
                    else:
                        out[:, pos, :] = out[:, pos, :] + vv
            return out

        if self._is_pre:
            def hook(mod, args):
                return (_apply(args[0]),) + tuple(args[1:])
            h = self._module.register_forward_pre_hook(hook)
        else:
            def hook(mod, args, output):
                if isinstance(output, tuple):
                    return (_apply(output[0]),) + tuple(output[1:])
                return _apply(output)
            h = self._module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = self.model(full_ids).logits[0]
        finally:
            h.remove()
        return logits


# ---------------------------------------------------------------------------
# Base (unsteered) generation — mirror battery.base_generate for consistency
# ---------------------------------------------------------------------------

def base_generate(model, tokenizer, prompt: str, max_new_tokens: int = 8,
                  device: str = "cpu") -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             do_sample=False, num_beams=1,
                             pad_token_id=tokenizer.eos_token_id)
    cont_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)
