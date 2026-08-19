"""Activation caching and (subspace-projected) patching via forward hooks.

Some callers need to replace *only the projection onto a subspace* of an
activation at a site during a target forward pass. This module provides:

- :func:`cache_site`      : run a source prompt and cache a site's activation.
- :func:`patch_and_run`   : run a target prompt while overwriting a site's
  activation (fully, or only its component in a given subspace).
- :func:`project_onto`    : project vectors onto an orthonormal basis.

Shapes
------
Cached activations are ``[seq, hidden]`` (single prompt). Patches are applied at
the block output (``resid_post``) or sub-module outputs, at chosen positions.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional, Sequence

import torch

from .capture import _get_submodule, _module_output_tensor, get_blocks


def project_onto(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project rows of ``x`` onto the subspace spanned by ``basis``.

    Parameters
    ----------
    x : ``[..., hidden]``.
    basis : ``[k, hidden]`` orthonormal rows (an orthonormal basis of a
        k-dim subspace). If not orthonormal, results use ``basis`` as-is via
        ``(x @ basis.T) @ basis`` (correct only for orthonormal bases).

    Returns
    -------
    ``[..., hidden]`` : the component of ``x`` lying in the subspace.
    """
    coeffs = x @ basis.transpose(-1, -2)  # [..., k]
    return coeffs @ basis                  # [..., hidden]


def _resolve_target_module(model, site, layer):
    """Return (module, is_pre) for hooking a given site/layer."""
    block = get_blocks(model)[layer]
    if site == "resid_post":
        return block, False
    if site == "resid_pre":
        return block, True
    if site == "mlp_out":
        return _get_submodule(block, "mlp"), False
    if site == "attn_out":
        return _get_submodule(block, "attn"), False
    raise ValueError(f"Unsupported site {site!r} for patching.")


def cache_site(model, tokenizer, prompt: str, site: str, layer: int,
               device: str = "cpu") -> dict:
    """Run ``prompt`` and cache the activation at ``(layer, site)``.

    Returns
    -------
    dict with:
        ``activation`` : ``[seq, hidden]`` cached tensor (CPU, detached).
        ``input_ids``  : ``[seq]`` token ids.
        ``site``, ``layer`` : echoed for convenience.
    """
    from .capture import capture_hooks

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    store: dict = {}
    with torch.no_grad(), capture_hooks(model, [site], [layer], store):
        model(**enc)
    act = store[(layer, site)][-1].to("cpu")  # [1, seq, hidden]
    return {
        "activation": act[0],
        "input_ids": enc["input_ids"][0].to("cpu"),
        "site": site,
        "layer": layer,
    }


def _apply_patch_to_row(target, patch_row, subspace, mode,
                        remove_subspace=None, add_vector=None):
    """Return the patched value for a single ``[batch, hidden]`` slice.

    ``mode`` is one of:
      - ``"replace"`` / ``"subspace_swap"``: semantics depend on whether
        ``subspace`` is given (matching :func:`patch_and_run`). With a subspace,
        the target's component in ``subspace`` is replaced by the source's
        component in that SAME subspace.
      - ``"project_out"``: ablation — remove the ``subspace`` component of the
        target entirely (no source used).
      - ``"subspace_transplant"``: two-basis composition (plan §10 v2). Project
        OUT the target's component in ``remove_subspace`` (B's own basis), then
        ADD ``add_vector`` (A's precomputed component, e.g. A's mean activation
        projected onto A's OWN basis). ``add_vector`` is a fixed ``[hidden]``
        vector supplied by the caller; ``patch_row`` and ``subspace`` are unused.
        This lets the remove-basis and the add-component live in different
        (per-fact) bases, which single-basis ``subspace_swap`` cannot express.
    """
    if mode == "subspace_transplant":
        rb = remove_subspace.to(target.dtype).to(target.device)
        av = add_vector.to(target.dtype).to(target.device)
        target_perp = target - project_onto(target, rb)
        return target_perp + av  # av broadcasts over the batch dim
    if mode == "project_out":
        # Ablation: drop the subspace component of the target entirely.
        b = subspace.to(target.dtype).to(target.device)
        return target - project_onto(target, b)
    if subspace is None:
        # Full replacement with the source activation.
        return patch_row.to(target.dtype).to(target.device).expand_as(target)
    b = subspace.to(target.dtype).to(target.device)
    target_perp = target - project_onto(target, b)
    patch_par = project_onto(
        patch_row.to(target.dtype).to(target.device).unsqueeze(0), b).squeeze(0)
    return target_perp + patch_par


@contextmanager
def _patch_hook(model, site, layer, patch_value, positions, subspace,
                mode="replace"):
    """Install a hook that overwrites a site's activation during forward.

    ``patch_value`` is ``[len(positions), hidden]`` (or broadcastable). If
    ``subspace`` (``[k, hidden]`` orthonormal) is given, only the component of
    the activation in that subspace is replaced by the patch's component. For
    ``mode="project_out"`` the subspace component is removed (no source needed).
    """
    module, is_pre = _resolve_target_module(model, site, layer)

    def _apply(hs: torch.Tensor) -> torch.Tensor:
        # hs: [batch, seq, hidden]
        out = hs.clone()
        pv = None if patch_value is None else patch_value.to(hs.dtype).to(hs.device)
        for i, pos in enumerate(positions):
            target = out[:, pos, :]                # [batch, hidden]
            patch_row = None if pv is None else pv[i]
            out[:, pos, :] = _apply_patch_to_row(
                target, patch_row, subspace, mode)
        return out

    handles = []
    if is_pre:
        def pre_hook(mod, args):
            hs = args[0]
            new = _apply(hs)
            return (new,) + tuple(args[1:])
        handles.append(module.register_forward_pre_hook(pre_hook))
    else:
        def hook(mod, args, output):
            if isinstance(output, tuple):
                new = _apply(output[0])
                return (new,) + tuple(output[1:])
            return _apply(output)
        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def patch_and_run(model, tokenizer, target_prompt: str, patches: Sequence[dict],
                  subspace: Optional[torch.Tensor] = None,
                  positions: Optional[Sequence[int]] = None,
                  device: str = "cpu") -> dict:
    """Run ``target_prompt`` while patching cached activations into sites.

    Parameters
    ----------
    target_prompt : the prompt to run.
    patches : sequence of dicts, each from :func:`cache_site` (must contain
        ``activation`` ``[src_seq, hidden]``, ``site``, ``layer``). Each patch
        overwrites its site at the target's positions.
    subspace : optional ``[k, hidden]`` orthonormal basis. If given, only the
        projection of the activation onto this subspace is replaced
        (subspace-patching); the orthogonal component of the target is preserved.
    positions : target positions to patch. Defaults to the last position. Length
        must be <= the cached source length; cached rows are aligned by order
        from the *end* of the source (so "last-to-last" patching is the default).
    device : forward device.

    Returns
    -------
    dict with:
        ``logits`` : ``[seq, vocab]`` logits for the target forward pass.
        ``next_token_logits`` : ``[vocab]`` logits at the last target position.
        ``next_token_probs``  : softmax of the above.
    """
    enc = tokenizer(target_prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    seq_len = enc["input_ids"].shape[1]
    if positions is None:
        positions = [seq_len - 1]
    positions = list(positions)

    from contextlib import ExitStack
    with torch.no_grad(), ExitStack() as stack:
        for patch in patches:
            src = patch["activation"]  # [src_seq, hidden]
            # Align cached rows to target positions from the end.
            rows = src[-len(positions):]
            if rows.shape[0] < len(positions):
                raise ValueError(
                    "cached activation shorter than number of patch positions")
            stack.enter_context(
                _patch_hook(model, patch["site"], patch["layer"], rows,
                            positions, subspace))
        out = model(**enc)

    logits = out.logits[0].to("cpu")  # [seq, vocab]
    next_logits = logits[-1]
    return {
        "logits": logits,
        "next_token_logits": next_logits,
        "next_token_probs": torch.softmax(next_logits, dim=-1),
    }


# ---------------------------------------------------------------------------
# Free-generation-under-patching
# ---------------------------------------------------------------------------


@contextmanager
def _dynamic_patch_hook(model, site, layer, patch_row, subspace, mode, state,
                        remove_subspace=None, add_vector=None):
    """Install a patch hook whose target positions are resolved per forward.

    Unlike :func:`_patch_hook` (fixed absolute positions), this hook consults a
    mutable ``state`` on every forward call to decide which *columns* of the
    current hidden-state tensor to patch. This is what makes patching work under
    KV caching, where forward calls after the first pass a single new token and
    absolute positions no longer index the tensor directly.

    ``state`` fields (updated by the generation loop before each forward):
        ``cache_offset`` : absolute index of column 0 of the current forward's
            hidden state (0 on the prefill pass; == past length when cached).
        ``targets`` : iterable of absolute positions requested for patching.
    ``patch_row`` is a single ``[hidden]`` source vector (or None for
    ``project_out``); the same source is applied at every patched position.
    """
    module, is_pre = _resolve_target_module(model, site, layer)

    def _apply(hs: torch.Tensor) -> torch.Tensor:
        seq = hs.shape[1]
        offset = state["cache_offset"]
        # Map requested absolute positions into columns present this forward.
        cols = [p - offset for p in state["targets"]
                if 0 <= (p - offset) < seq]
        if not cols:
            return hs
        out = hs.clone()
        for col in cols:
            target = out[:, col, :]
            out[:, col, :] = _apply_patch_to_row(
                target, patch_row, subspace, mode,
                remove_subspace=remove_subspace, add_vector=add_vector)
        return out

    handles = []
    if is_pre:
        def pre_hook(mod, args):
            hs = args[0]
            return (_apply(hs),) + tuple(args[1:])
        handles.append(module.register_forward_pre_hook(pre_hook))
    else:
        def hook(mod, args, output):
            if isinstance(output, tuple):
                return (_apply(output[0]),) + tuple(output[1:])
            return _apply(output)
        handles.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _resolve_generation_positions(positions, prompt_len):
    """Return a callable ``targets(abs_step) -> list[int]`` of absolute indices.

    ``abs_step`` is the absolute index of the token being *produced this step*
    (i.e. the position whose activation feeds the next-token prediction).

    Semantics
    ---------
    - ``"last_prompt"`` : patch only the prompt's last token (absolute index
      ``prompt_len - 1``). With ``use_cache=True`` this position is only present
      in the prefill forward; its patched layer output is baked into the KV
      cache and therefore persists implicitly for all later steps.
    - ``"all_generated"`` : patch the current token position at every step
      (the newest column of each forward), starting from the prompt's last token.
    - explicit ``list``/sequence of ints : those absolute positions, patched
      whenever they appear in a forward's hidden state.
    """
    if positions == "last_prompt":
        fixed = [prompt_len - 1]
        return lambda abs_step: fixed
    if positions == "all_generated":
        # Patch every position from the last prompt token onward.
        return lambda abs_step: [abs_step]
    if isinstance(positions, (list, tuple)):
        fixed = list(positions)
        return lambda abs_step: fixed
    raise ValueError(
        "positions must be 'last_prompt', 'all_generated', or a list of ints; "
        f"got {positions!r}")


def generate_with_patch(model, tokenizer, prompt: str,
                        sources: Sequence[dict],
                        *,
                        subspace: Optional[torch.Tensor] = None,
                        remove_subspace: Optional[torch.Tensor] = None,
                        add_vector: Optional[torch.Tensor] = None,
                        positions="last_prompt",
                        max_new_tokens: int = 10,
                        mode: str = "replace",
                        use_cache: bool = True,
                        do_sample: bool = False,
                        temperature: float = 1.0,
                        seed: Optional[int] = None,
                        device: str = "cpu") -> dict:
    """Greedily (or by sampling) generate under a persistent activation patch.

    The patch is (re)applied at every forward step, at the positions selected by
    ``positions``, so free-generation causal tests see the intervention persist
    across the whole continuation rather than a single forward pass.

    Parameters
    ----------
    model, tokenizer : from :func:`actlib.models.load_model`.
    prompt : the prompt to condition on.
    sources : sequence of cached-site dicts (from :func:`cache_site`); the
        last row of each source's ``activation`` is used as the ``[hidden]``
        source vector for its ``(layer, site)``. Ignored for ``mode="project_out"``
        (pass ``[]``). Multiple sources patch multiple sites simultaneously.
    subspace : optional ``[k, hidden]`` orthonormal basis. Semantics match
        :func:`patch_and_run` (replace only the subspace component). Required for
        ``mode="subspace_swap"`` and ``mode="project_out"``.
    positions : "last_prompt", "all_generated", or an explicit list of absolute
        positions. See :func:`_resolve_generation_positions` for how each
        interacts with KV caching.
    max_new_tokens : number of tokens to generate.
    mode : "replace" (full or subspace replacement from source), "subspace_swap"
        (alias for replace when a subspace is given), or "project_out" (ablate
        the subspace component; no source needed).
    use_cache : pass-through to the forward (KV caching). Default True. Set False
        to re-run the full sequence each step (used by the consistency test).
    do_sample : if True, temperature-sample instead of greedy argmax.
    temperature : sampling temperature (only used when ``do_sample``).
    seed : optional torch seed for reproducible sampling.
    device : forward device.

    Returns
    -------
    dict with:
        ``generated_ids``   : ``[max_new_tokens]`` LongTensor of new token ids.
        ``continuation``    : decoded text of the generated ids.
        ``full_text``       : prompt + continuation decoded together.
        ``first_step_top5`` : dict with ``ids`` ``[5]`` and ``probs`` ``[5]`` for
            the first generated step's next-token distribution.
    """
    if mode not in ("replace", "subspace_swap", "project_out",
                    "subspace_transplant"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "subspace_swap" and subspace is None:
        raise ValueError("mode='subspace_swap' requires a subspace basis")
    if mode == "project_out" and subspace is None:
        raise ValueError("mode='project_out' requires a subspace basis")
    if mode == "subspace_transplant":
        if remove_subspace is None or add_vector is None:
            raise ValueError(
                "mode='subspace_transplant' requires remove_subspace [k,hidden] "
                "and add_vector [hidden]")
        if not sources:
            raise ValueError(
                "subspace_transplant needs source dicts to name (layer, site) "
                "targets; pass a cache_site output (its activation is unused, "
                "the add_vector carries the source component)")
    if mode == "project_out":
        # Ablation uses no source; but we still need to know the target sites.
        if not sources:
            raise ValueError(
                "project_out still needs source dicts to name (layer, site) "
                "targets; pass cache_site outputs (their activations are unused)")

    if seed is not None:
        torch.manual_seed(seed)

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)              # [1, prompt_len]
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_len = input_ids.shape[1]

    targets_fn = _resolve_generation_positions(positions, prompt_len)

    # One mutable state + hook per source site.
    site_states = []
    for src in sources:
        row = None
        if mode not in ("project_out", "subspace_transplant"):
            row = src["activation"][-1]                    # [hidden]
        site_states.append({
            "site": src["site"], "layer": src["layer"],
            "row": row,
            "state": {"cache_offset": 0, "targets": []},
        })

    generated = []
    first_step_top5 = None

    from contextlib import ExitStack
    with torch.no_grad(), ExitStack() as stack:
        for s in site_states:
            stack.enter_context(_dynamic_patch_hook(
                model, s["site"], s["layer"], s["row"], subspace, mode,
                s["state"], remove_subspace=remove_subspace,
                add_vector=add_vector))

        cur_ids = input_ids
        cur_mask = attention_mask
        past = None
        for step in range(max_new_tokens):
            # Absolute index of the token producing this step's prediction.
            abs_step = prompt_len - 1 + step
            step_targets = targets_fn(abs_step)

            if use_cache and past is not None:
                # Cached path: this forward only sees the single newest token,
                # whose column-0 absolute index is (current length - 1).
                cache_offset = abs_step
                forward_ids = cur_ids[:, -1:]
                fwd_mask = cur_mask
            else:
                cache_offset = 0
                forward_ids = cur_ids
                fwd_mask = cur_mask

            for s in site_states:
                s["state"]["cache_offset"] = cache_offset
                s["state"]["targets"] = step_targets

            fkwargs = {"use_cache": use_cache}
            if fwd_mask is not None:
                fkwargs["attention_mask"] = fwd_mask
            if use_cache and past is not None:
                fkwargs["past_key_values"] = past
            out = model(forward_ids, **fkwargs)

            next_logits = out.logits[0, -1]                # [vocab]
            if step == 0:
                probs0 = torch.softmax(next_logits, dim=-1)
                top_p, top_i = probs0.topk(5)
                first_step_top5 = {
                    "ids": top_i.to("cpu"),
                    "probs": top_p.to("cpu"),
                }

            if do_sample:
                probs = torch.softmax(next_logits / max(temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = int(next_logits.argmax().item())
            generated.append(next_id)

            next_tok = torch.tensor([[next_id]], device=device)
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            if cur_mask is not None:
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype,
                                          device=device)], dim=1)
            if use_cache:
                past = out.past_key_values

    gen_ids = torch.tensor(generated, dtype=torch.long)
    continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
    full_text = tokenizer.decode(
        torch.cat([input_ids[0].to("cpu"), gen_ids]), skip_special_tokens=True)
    return {
        "generated_ids": gen_ids,
        "continuation": continuation,
        "full_text": full_text,
        "first_step_top5": first_step_top5,
    }
