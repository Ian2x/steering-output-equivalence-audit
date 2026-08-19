"""Forward-hook activation capture across HF architecture families.

Sites
-----
- ``resid_pre``  : residual stream at block *input*  (hidden_states into block i)
- ``resid_post`` : residual stream at block *output* (hidden_states out of block i)
- ``mlp_out``    : output of the block's MLP sub-module
- ``attn_out``   : output of the block's attention sub-module
- ``mlp_hidden`` : MLP hidden state post-nonlinearity (the input to the MLP's
  down-projection). Width is the MLP intermediate size (e.g. 4x hidden), not
  the residual width. This is the site where a privileged per-channel basis
  is expected (Elhage et al. 2022) and the site Yona et al. 2026 rank neurons
  at.

All captured tensors are detached, moved to CPU, and (optionally) downcast.
Hooks are always removed via a context manager so repeated calls do not leak.

Shapes & units
--------------
A capture result is a dict keyed by ``(layer, site)`` mapping to a tensor.
- positions="last": tensor shape ``[n_prompts, hidden]`` (last non-pad token).
- positions="all" : tensor shape ``[n_prompts, seq, hidden]`` (padded to the
  batch max; padding positions are zeroed and an attention mask is returned).
- positions=int   : tensor shape ``[n_prompts, hidden]`` (that absolute index).

Values are raw activations in the model's units (residual-stream / sub-module
outputs), float32 by default.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Sequence, Tuple, Union

import torch

from .models import _detect_family

SUPPORTED_SITES = ("resid_pre", "resid_post", "mlp_out", "attn_out",
                   "mlp_hidden")


# ---------------------------------------------------------------------------
# Architecture adapter
# ---------------------------------------------------------------------------

def get_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the ModuleList of transformer blocks for a supported model.

    Raises
    ------
    ValueError if the architecture family is not recognized.
    """
    family = _detect_family(model)
    if family == "gpt2":
        return model.transformer.h
    if family == "gpt_neox":
        return model.gpt_neox.layers
    # llama / qwen / gemma2
    return model.model.layers


def _get_submodule(block: torch.nn.Module, kind: str) -> torch.nn.Module:
    """Return the attention or mlp sub-module of a block across families.

    ``kind`` is "attn" or "mlp".
    """
    if kind == "mlp":
        if hasattr(block, "mlp"):
            return block.mlp
        raise ValueError("Block has no recognizable MLP sub-module.")
    # attention
    for name in ("attn", "attention", "self_attn"):
        if hasattr(block, name):
            return getattr(block, name)
    raise ValueError("Block has no recognizable attention sub-module.")


def get_mlp_down_proj(block: torch.nn.Module) -> torch.nn.Module:
    """Return the MLP down-projection sub-module of a block across families.

    Its *input* is the post-nonlinearity MLP hidden state (``mlp_hidden``):
    gpt2 ``mlp.c_proj``, gpt_neox ``mlp.dense_4h_to_h``, llama/qwen/gemma2
    ``mlp.down_proj`` (for gated MLPs this is ``act(gate) * up``, the standard
    per-channel-analysis site).
    """
    mlp = _get_submodule(block, "mlp")
    for name in ("c_proj", "dense_4h_to_h", "down_proj"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError("MLP has no recognizable down-projection sub-module.")


def _module_output_tensor(output) -> torch.Tensor:
    """Extract the primary hidden-state tensor from a hook output.

    HF sub-modules commonly return either a Tensor or a tuple whose first
    element is the hidden state.
    """
    if isinstance(output, tuple):
        return output[0]
    return output


# ---------------------------------------------------------------------------
# Hook registration context manager
# ---------------------------------------------------------------------------

@contextmanager
def capture_hooks(model: torch.nn.Module, sites: Sequence[str],
                  layers: Sequence[int], store: Dict[Tuple[int, str], list]):
    """Register forward hooks for the requested (layer, site) pairs.

    Captured tensors are appended (per forward call) into ``store`` lists keyed
    by ``(layer, site)``. Hooks are removed on exit even if an error occurs.

    ``resid_pre`` uses a *forward_pre* hook on the block (its input hidden
    state). ``resid_post`` uses a forward hook on the block. ``mlp_out`` /
    ``attn_out`` use forward hooks on the respective sub-modules.
    """
    for s in sites:
        if s not in SUPPORTED_SITES:
            raise ValueError(
                f"Unsupported site {s!r}. Supported: {SUPPORTED_SITES}")
    blocks = get_blocks(model)
    n = len(blocks)
    handles = []

    def _make_block_pre_hook(layer):
        def hook(module, args):
            # args[0] is hidden_states into the block.
            hs = args[0] if isinstance(args, tuple) else args
            store[(layer, "resid_pre")].append(hs.detach())
        return hook

    def _make_block_hook(layer):
        def hook(module, args, output):
            hs = _module_output_tensor(output)
            store[(layer, "resid_post")].append(hs.detach())
        return hook

    def _make_sub_hook(layer, site):
        def hook(module, args, output):
            hs = _module_output_tensor(output)
            store[(layer, site)].append(hs.detach())
        return hook

    def _make_sub_pre_hook(layer, site):
        def hook(module, args):
            hs = args[0] if isinstance(args, tuple) else args
            store[(layer, site)].append(hs.detach())
        return hook

    try:
        for layer in layers:
            if layer < 0 or layer >= n:
                raise IndexError(
                    f"layer {layer} out of range for model with {n} blocks")
            block = blocks[layer]
            for site in sites:
                store.setdefault((layer, site), [])
                if site == "resid_pre":
                    handles.append(block.register_forward_pre_hook(
                        _make_block_pre_hook(layer)))
                elif site == "resid_post":
                    handles.append(block.register_forward_hook(
                        _make_block_hook(layer)))
                elif site == "mlp_out":
                    sub = _get_submodule(block, "mlp")
                    handles.append(sub.register_forward_hook(
                        _make_sub_hook(layer, site)))
                elif site == "attn_out":
                    sub = _get_submodule(block, "attn")
                    handles.append(sub.register_forward_hook(
                        _make_sub_hook(layer, site)))
                elif site == "mlp_hidden":
                    sub = get_mlp_down_proj(block)
                    handles.append(sub.register_forward_pre_hook(
                        _make_sub_pre_hook(layer, site)))
        yield store
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Position selection with padding awareness
# ---------------------------------------------------------------------------

def _last_nonpad_index(attention_mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    """Return the index of the last non-pad token per row.

    ``attention_mask`` is ``[batch, seq]`` (1 = real token). For right padding
    this is ``mask.sum(1) - 1``; for left padding it is always ``seq - 1``.
    """
    seq = attention_mask.shape[1]
    if padding_side == "left":
        return torch.full((attention_mask.shape[0],), seq - 1, dtype=torch.long)
    return attention_mask.long().sum(dim=1) - 1


def _select_positions(batch_act: torch.Tensor, attention_mask: torch.Tensor,
                      positions, padding_side: str):
    """Select requested positions from a ``[batch, seq, hidden]`` tensor.

    Returns either ``[batch, hidden]`` (last / int) or the full
    ``[batch, seq, hidden]`` with padding positions zeroed (all).
    """
    if positions == "last":
        idx = _last_nonpad_index(attention_mask, padding_side)
        gather = idx.view(-1, 1, 1).expand(-1, 1, batch_act.shape[-1])
        return batch_act.gather(1, gather).squeeze(1)
    if positions == "all":
        masked = batch_act * attention_mask.unsqueeze(-1).to(batch_act.dtype)
        return masked
    if isinstance(positions, int):
        return batch_act[:, positions, :]
    raise ValueError(
        f"positions must be 'last', 'all', or an int; got {positions!r}")


# ---------------------------------------------------------------------------
# Public capture API
# ---------------------------------------------------------------------------

def capture_activations(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    sites: Union[str, Sequence[str]],
    layers: Union[int, Sequence[int]],
    positions="last",
    batch_size: int = 8,
    device: str = "cpu",
    dtype: torch.dtype = None,
    return_mask: bool = False,
):
    """Capture activations at named sites/layers over ``prompts``.

    Parameters
    ----------
    model, tokenizer : from :func:`actlib.models.load_model`.
    prompts : sequence of str.
    sites : one or more of ``resid_pre``, ``resid_post``, ``mlp_out``,
        ``attn_out``.
    layers : one or more 0-based block indices.
    positions : "last" (last non-pad token), "all" (full padded sequence with
        pad positions zeroed), or an int absolute index.
    batch_size : forward-pass batch size.
    device : device to run the forward passes on ("cpu"/"mps").
    dtype : optional downcast for stored activations (e.g. torch.float16) to
        save memory. Values are always detached and moved to CPU regardless.
    return_mask : if True and positions=="all", also return the attention mask.

    Returns
    -------
    dict keyed by ``(layer, site)`` -> tensor on CPU. See module docstring for
    shapes. If ``return_mask`` and positions=="all", returns
    ``(result, mask)`` where mask is ``[n_prompts, seq]``.

    Notes
    -----
    Padding side is taken from ``tokenizer.padding_side``; "last" resolves to
    the last non-pad token using the attention mask, so results are invariant
    to left vs right padding.
    """
    if isinstance(sites, str):
        sites = [sites]
    if isinstance(layers, int):
        layers = [layers]
    layers = list(layers)

    padding_side = tokenizer.padding_side
    # Accumulate per (layer, site).
    collected: Dict[Tuple[int, str], List[torch.Tensor]] = {
        (l, s): [] for l in layers for s in sites}
    masks_all: List[torch.Tensor] = []

    prompts = list(prompts)
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        attn = enc["attention_mask"]

        # Padding-aware position_ids: real tokens should be positioned 0..L-1
        # regardless of padding side. Absolute-position models (e.g. GPT-2)
        # otherwise give different activations under left vs right padding.
        # (RoPE models ignore position_ids passed this way but are unaffected
        # since HF handles the mask; passing correct ids is harmless.)
        pos_ids = (attn.long().cumsum(dim=1) - 1).clamp_min(0)
        pos_ids = pos_ids * attn.long()

        store: Dict[Tuple[int, str], list] = {}
        forward_kwargs = dict(enc)
        try:
            with torch.no_grad(), capture_hooks(model, sites, layers, store):
                model(**forward_kwargs, position_ids=pos_ids)
        except (TypeError, ValueError):
            # Some architectures don't accept an explicit position_ids kwarg;
            # fall back to default positioning.
            store.clear()
            with torch.no_grad(), capture_hooks(model, sites, layers, store):
                model(**forward_kwargs)

        attn_cpu = attn.detach().to("cpu")
        if positions == "all":
            masks_all.append(attn_cpu)

        for key, tensors in store.items():
            # One forward call -> exactly one captured tensor per key.
            act = tensors[-1].to("cpu")  # [batch, seq, hidden]
            sel = _select_positions(act, attn_cpu, positions, padding_side)
            if dtype is not None:
                sel = sel.to(dtype)
            collected[key].append(sel)

    result: Dict[Tuple[int, str], torch.Tensor] = {}
    for key, parts in collected.items():
        if positions == "all":
            # Sequences may differ in length across batches; pad to global max.
            maxlen = max(p.shape[1] for p in parts)
            padded = []
            for p in parts:
                if p.shape[1] < maxlen:
                    pad = torch.zeros(p.shape[0], maxlen - p.shape[1],
                                      p.shape[2], dtype=p.dtype)
                    p = torch.cat([p, pad], dim=1)
                padded.append(p)
            result[key] = torch.cat(padded, dim=0)
        else:
            result[key] = torch.cat(parts, dim=0)

    if return_mask and positions == "all":
        maxlen = max(m.shape[1] for m in masks_all)
        mpad = []
        for m in masks_all:
            if m.shape[1] < maxlen:
                m = torch.cat([m, torch.zeros(m.shape[0], maxlen - m.shape[1],
                                              dtype=m.dtype)], dim=1)
            mpad.append(m)
        return result, torch.cat(mpad, dim=0)
    return result
