"""actlib — shared activation-capture library.

Public surface:

Models / introspection
    load_model, MODEL_REGISTRY, get_model_info, ModelInfo, resolve_name

Capture
    capture_activations, capture_hooks, get_blocks, SUPPORTED_SITES

Streaming stats / z-scoring
    ActivationStats, zscore

Norm gains
    extract_norm_gains

Channel ranking
    rank_channels, METHODS

Patching (subspace-projected)
    cache_site, patch_and_run, generate_with_patch, project_onto

Probing
    train_probe, eval_probe, train_test_split_idx

Supports GPT-2, GPT-NeoX/Pythia, Llama, Qwen2, and Gemma block layouts.
CPU is the default device; MPS is opt-in and guarded.
"""

from .models import (
    MODEL_REGISTRY,
    ModelInfo,
    get_model_info,
    load_model,
    resolve_name,
)
from .capture import (
    SUPPORTED_SITES,
    capture_activations,
    capture_hooks,
    get_blocks,
)
from .stats import ActivationStats, zscore
from .norms import extract_norm_gains
from .ranking import (METHODS, rank_channels, sink_channels,
                      StreamingChannelScores)
from .patching import (cache_site, generate_with_patch, patch_and_run,
                       project_onto)
from .probing import eval_probe, train_probe, train_test_split_idx
from .similarity import linear_cka

__all__ = [
    "MODEL_REGISTRY", "ModelInfo", "get_model_info", "load_model", "resolve_name",
    "SUPPORTED_SITES", "capture_activations", "capture_hooks", "get_blocks",
    "ActivationStats", "zscore",
    "extract_norm_gains",
    "METHODS", "rank_channels",
    "cache_site", "patch_and_run", "generate_with_patch", "project_onto",
    "eval_probe", "train_probe", "train_test_split_idx",
    "linear_cka",
]
