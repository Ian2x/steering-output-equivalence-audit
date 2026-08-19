"""Linear probes over activations (logistic / ridge) via scikit-learn.

Returns learned directions (weight vectors) so callers can build subspaces for
:mod:`actlib.patching`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def train_test_split_idx(n: int, test_frac: float = 0.25, seed: int = 0):
    """Return (train_idx, test_idx) numpy arrays for a shuffled split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    return perm[n_test:], perm[:n_test]


def train_probe(X, y, kind: str = "logistic", standardize: bool = True,
                seed: int = 0, C: float = 1.0, alpha: float = 1.0):
    """Train a linear probe on ``(X, y)``.

    Parameters
    ----------
    X : ``[n, d]`` features (activations).
    y : ``[n]`` labels. Binary/int for ``logistic``; continuous for ``ridge``.
    kind : "logistic" (classification) or "ridge" (regression).
    standardize : z-score features using training statistics (recommended).
    seed : RNG seed for the solver.
    C, alpha : regularization strengths (C for logistic, alpha for ridge).

    Returns
    -------
    dict with:
        ``model`` : the fitted sklearn estimator.
        ``scaler``: the fitted StandardScaler or None.
        ``direction`` : ``[d]`` weight vector in the ORIGINAL feature space
            (unit-normalized). For subspace construction, use this direction.
        ``kind`` : echoed.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler

    Xn = _to_numpy(X)
    yn = _to_numpy(y)

    scaler = None
    Xt = Xn
    if standardize:
        scaler = StandardScaler().fit(Xn)
        Xt = scaler.transform(Xn)

    if kind == "logistic":
        est = LogisticRegression(C=C, max_iter=2000, random_state=seed)
        est.fit(Xt, yn.astype(int))
        # coef_ is [1, d] for binary, [n_classes, d] for multiclass.
        coef = est.coef_
    elif kind == "ridge":
        est = Ridge(alpha=alpha, random_state=seed)
        est.fit(Xt, yn)
        coef = np.atleast_2d(est.coef_)
    else:
        raise ValueError(f"kind must be 'logistic' or 'ridge'; got {kind!r}")

    # Map weight back to original feature space if standardized: a change of
    # dx in original = dx/scale in scaled space, so original-space direction is
    # w / scale (broadcast over rows for the multiclass case).
    if scaler is not None:
        coef_orig = coef / scaler.scale_.reshape(1, -1)
    else:
        coef_orig = coef
    # Unit-normalize each row (one direction per class). For binary/single-row
    # problems we return a [d] vector for backward compatibility; multiclass
    # returns a [n_classes, d] matrix of per-class directions.
    norms = np.linalg.norm(coef_orig, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dir_mat = coef_orig / norms
    direction = dir_mat[0] if dir_mat.shape[0] == 1 else dir_mat

    return {
        "model": est,
        "scaler": scaler,
        "direction": torch.from_numpy(direction).float(),
        "kind": kind,
    }


def eval_probe(probe: dict, X, y) -> dict:
    """Evaluate a fitted probe from :func:`train_probe`.

    Returns
    -------
    dict:
        classification: ``accuracy`` and ``auroc`` (binary only, else None).
        regression: ``r2``.
    """
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score

    est = probe["model"]
    scaler = probe["scaler"]
    Xn = _to_numpy(X)
    yn = _to_numpy(y)
    Xt = scaler.transform(Xn) if scaler is not None else Xn

    if probe["kind"] == "logistic":
        pred = est.predict(Xt)
        out = {"accuracy": float(accuracy_score(yn.astype(int), pred))}
        try:
            if len(np.unique(yn)) == 2:
                prob = est.predict_proba(Xt)[:, 1]
                out["auroc"] = float(roc_auc_score(yn.astype(int), prob))
            else:
                out["auroc"] = None
        except Exception:
            out["auroc"] = None
        return out
    pred = est.predict(Xt)
    return {"r2": float(r2_score(yn, pred))}
