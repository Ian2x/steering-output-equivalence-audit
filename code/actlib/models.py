"""Model loading and architecture introspection for actlib.

Loads HuggingFace causal-LM models in a deterministic, eval-mode, CPU-default
configuration and exposes a small architecture adapter so downstream code can
locate transformer blocks and norms across differing HF families.

Conventions
-----------
- CPU is the default device for reproducibility. MPS is opt-in and guarded:
  if requested but unavailable we warn and fall back to CPU.
- All loaded models are put in ``.eval()`` mode with gradients disabled at the
  parameter level is NOT done here (callers may want grads); we only set eval.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps short aliases -> full HF ids. Unknown names are passed through to HF.
_ALIASES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "gemma-2-2b": "google/gemma-2-2b",
}

# Documents the wave's intended models and their role. ``auth`` marks models
# that typically require HuggingFace authentication / license acceptance and so
# must NOT be relied on in unit tests.
MODEL_REGISTRY = {
    # gpt2 family: pipeline development, gateless, tiny.
    "gpt2": {"hf_id": "gpt2", "role": "pipeline-dev", "auth": False,
             "norm": "LayerNorm", "family": "gpt2"},
    "gpt2-medium": {"hf_id": "gpt2-medium", "role": "pipeline-dev", "auth": False,
                    "norm": "LayerNorm", "family": "gpt2"},
    # pythia suite: checkpoints + cross-scale, gateless.
    "pythia-70m": {"hf_id": "EleutherAI/pythia-70m", "role": "cross-scale",
                   "auth": False, "norm": "LayerNorm", "family": "gpt_neox"},
    "pythia-160m": {"hf_id": "EleutherAI/pythia-160m", "role": "cross-scale",
                    "auth": False, "norm": "LayerNorm", "family": "gpt_neox"},
    "pythia-410m": {"hf_id": "EleutherAI/pythia-410m", "role": "cross-scale",
                    "auth": False, "norm": "LayerNorm", "family": "gpt_neox"},
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "role": "cross-scale",
                  "auth": False, "norm": "LayerNorm", "family": "gpt_neox"},
    # headline models: RMSNorm-based. Qwen2.5 repos are Apache-2.0 and NOT
    # gated (verified by anonymous download 2026-07-05); Llama/Gemma remain
    # license-gated.
    "qwen2.5-0.5b": {"hf_id": "Qwen/Qwen2.5-0.5B", "role": "headline",
                     "auth": False, "norm": "RMSNorm", "family": "llama"},
    "qwen2.5-1.5b": {"hf_id": "Qwen/Qwen2.5-1.5B", "role": "headline",
                     "auth": False, "norm": "RMSNorm", "family": "llama"},
    "llama-3.2-1b": {"hf_id": "meta-llama/Llama-3.2-1B", "role": "headline",
                     "auth": True, "norm": "RMSNorm", "family": "llama"},
    "gemma-2-2b": {"hf_id": "google/gemma-2-2b", "role": "headline",
                   "auth": True, "norm": "RMSNorm", "family": "gemma2"},
}


def resolve_name(name: str) -> str:
    """Resolve a short alias to a full HF id; pass through unknown ids."""
    key = name.lower()
    return _ALIASES.get(key, name)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    """Return a usable device string, guarding MPS availability."""
    if device == "mps":
        if not torch.backends.mps.is_available():
            warnings.warn("MPS requested but not available; falling back to CPU.")
            return "cpu"
        return "mps"
    if device == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return "cuda"
    return "cpu"


def load_model(
    name: str,
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
    seed: int = 0,
) -> Tuple[torch.nn.Module, "object"]:
    """Load an HF causal-LM model and tokenizer, deterministically.

    Parameters
    ----------
    name : str
        Short alias (e.g. ``"gpt2"``, ``"pythia-160m"``) or a full HF id.
    device : {"cpu", "mps", "cuda"}
        Target device. CPU is default. MPS/CUDA are guarded and fall back to
        CPU with a warning if unavailable.
    dtype : torch.dtype, optional
        Parameter dtype. Defaults to float32 (recommended on CPU/MPS for
        numerical parity in interpretability work).
    seed : int
        Seed applied to torch (and CUDA if present) for determinism.

    Returns
    -------
    (model, tokenizer)
        ``model`` is in ``.eval()`` mode on ``device``. The tokenizer has a
        ``pad_token`` set (falls back to ``eos_token`` if the model has none).

    Notes
    -----
    Models are downloaded once to the HF cache (set ``HF_HOME`` /
    ``TRANSFORMERS_CACHE`` to control location). No network needed afterwards.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = _resolve_device(device)
    hf_id = resolve_name(name)
    if dtype is None:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        # Many causal LMs (gpt2, llama) ship without a pad token.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
    model.to(dev)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Architecture introspection
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Summary of a model's shape and norm type.

    Attributes
    ----------
    family : str
        One of the supported adapter families ("gpt2", "gpt_neox", "llama",
        "gemma2").
    n_layers : int
        Number of transformer blocks.
    hidden_size : int
        Residual-stream width (channels).
    norm_type : str
        "LayerNorm" or "RMSNorm".
    """

    family: str
    n_layers: int
    hidden_size: int
    norm_type: str


def _detect_family(model: torch.nn.Module) -> str:
    """Return the adapter family string for a model, or raise if unsupported."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2"
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return "gpt_neox"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama, Qwen2, Mistral, Gemma-2 all use model.layers. Distinguish
        # gemma2 by config for the norm placement / naming where relevant.
        arch = (getattr(model.config, "model_type", "") or "").lower()
        if "gemma" in arch:
            return "gemma2"
        return "llama"
    raise ValueError(
        "Unsupported model architecture: "
        f"{type(model).__name__}. Supported families are GPT-2 "
        "(transformer.h), GPT-NeoX/Pythia (gpt_neox.layers), and "
        "Llama/Qwen/Gemma-2 (model.layers)."
    )


def _norm_type_of(module: torch.nn.Module) -> str:
    """Best-effort norm-type name from a norm module."""
    cls = type(module).__name__
    if "RMS" in cls:
        return "RMSNorm"
    if "LayerNorm" in cls or cls == "FusedLayerNorm":
        return "LayerNorm"
    # Gemma / Llama RMSNorm variants all contain "RMS"; fall back to class name.
    return cls


def get_model_info(model: torch.nn.Module) -> ModelInfo:
    """Introspect layer count, hidden size, and norm type of ``model``.

    Returns
    -------
    ModelInfo

    Raises
    ------
    ValueError
        If the architecture is not one of the supported families.
    """
    family = _detect_family(model)
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer")
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")

    # Grab a representative block-level norm to classify the family's norm type.
    from .capture import get_blocks  # local import to avoid cycle at import time

    blocks = get_blocks(model)
    example_block = blocks[0]
    norm_module = None
    for cand in ("ln_1", "input_layernorm"):
        if hasattr(example_block, cand):
            norm_module = getattr(example_block, cand)
            break
    norm_type = _norm_type_of(norm_module) if norm_module is not None else "unknown"

    return ModelInfo(
        family=family,
        n_layers=int(n_layers),
        hidden_size=int(hidden),
        norm_type=norm_type,
    )
