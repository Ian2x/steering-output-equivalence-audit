"""Extract per-layer normalization gain (weight) vectors across families.

Gain-weighted channel ranking needs the layer's norm gain, so we expose the
LayerNorm/RMSNorm weight vector feeding each block plus the final norm.
"""

from __future__ import annotations

from typing import Dict

import torch

from .capture import get_blocks
from .models import _detect_family, _norm_type_of


def _block_input_norm(block: torch.nn.Module):
    """Return the norm module applied to a block's residual-stream input."""
    for name in ("ln_1", "input_layernorm"):
        if hasattr(block, name):
            return getattr(block, name)
    raise ValueError("Block has no recognizable input norm.")


def _final_norm(model: torch.nn.Module):
    """Return the model's final norm module across families."""
    family = _detect_family(model)
    if family == "gpt2":
        return model.transformer.ln_f
    if family == "gpt_neox":
        return model.gpt_neox.final_layer_norm
    # llama / qwen / gemma2
    return model.model.norm


def extract_norm_gains(model: torch.nn.Module) -> Dict:
    """Extract per-layer pre-block norm gains and the final norm gain.

    Returns
    -------
    dict with keys:
        ``norm_type`` : "LayerNorm" or "RMSNorm".
        ``gains`` : tensor ``[n_layers, hidden]`` of per-layer input-norm
            weight (gain) vectors, on CPU, detached, float32.
        ``final_gain`` : tensor ``[hidden]`` of the final norm's gain.

    Notes
    -----
    Gains are the multiplicative ``weight`` parameter of the norm (units:
    dimensionless per-channel scale). LayerNorm biases are not returned; the
    gain-weighted ranking uses the gain magnitude only.
    """
    blocks = get_blocks(model)
    gains = []
    norm_type = None
    for block in blocks:
        norm = _block_input_norm(block)
        if norm_type is None:
            norm_type = _norm_type_of(norm)
        w = norm.weight.detach().to("cpu").float()
        gains.append(w)
    final = _final_norm(model)
    return {
        "norm_type": norm_type,
        "gains": torch.stack(gains, dim=0),
        "final_gain": final.weight.detach().to("cpu").float(),
    }
