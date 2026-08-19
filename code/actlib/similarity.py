"""Representation-similarity helpers.

Currently exposes ``linear_cka``: linear Centered Kernel Alignment between two
activation matrices ``X, Y`` of shape ``[n, d1]`` and ``[n, d2]`` (same n rows,
paired by row). CKA is invariant to orthogonal transforms and isotropic scaling
of the features, which makes it a standard tool for comparing representations
across layers/models/prompt-sets.

Kept minimal and dependency-free (numpy/torch only).
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def linear_cka(X, Y) -> float:
    """Linear CKA between paired representation matrices.

    Parameters
    ----------
    X : ``[n, d1]`` activations.
    Y : ``[n, d2]`` activations (same ``n`` rows, paired by row index).

    Returns
    -------
    float in ``[0, 1]`` (1 = identical up to orthogonal transform + isotropic
    scaling). Uses the feature-space (linear-kernel) closed form:

        CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    with both X and Y column-centered first. Returns 0.0 if either matrix has
    zero variance (degenerate).
    """
    Xn = _to_numpy(X).astype(np.float64)
    Yn = _to_numpy(Y).astype(np.float64)
    if Xn.shape[0] != Yn.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of rows; got {Xn.shape[0]} "
            f"and {Yn.shape[0]}")
    # Column-center.
    Xc = Xn - Xn.mean(axis=0, keepdims=True)
    Yc = Yn - Yn.mean(axis=0, keepdims=True)
    # ||Y^T X||_F^2 = ||Xc^T Yc||_F^2.
    cross = Xc.T @ Yc                      # [d1, d2]
    hsic_xy = float((cross ** 2).sum())
    xx = Xc.T @ Xc
    yy = Yc.T @ Yc
    norm_x = float(np.sqrt((xx ** 2).sum()))
    norm_y = float(np.sqrt((yy ** 2).sum()))
    denom = norm_x * norm_y
    if denom == 0.0:
        return 0.0
    return hsic_xy / denom
