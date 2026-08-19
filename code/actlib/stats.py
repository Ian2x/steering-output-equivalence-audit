"""Streaming per-channel activation statistics (Welford) + z-scoring.

Memory model
------------
:class:`ActivationStats` holds only running moments of shape ``[n_features]``
(mean, M2) plus an optional bounded reservoir for robust stats. It therefore
scales to an arbitrarily large corpus without holding all activations.

- Mean / var / std: exact one-pass Welford, O(n_features) memory.
- Median / MAD (robust): estimated from a bounded random reservoir of at most
  ``reservoir_size`` rows (default 4096). This is an approximation whose memory
  is bounded by ``reservoir_size * n_features``. For exact medians, disable the
  reservoir and do a second pass over stored data yourself.

All inputs are treated as ``[n_rows, n_features]`` (rows = tokens/prompts,
features = channels). Higher-rank tensors are flattened over leading dims.
"""

from __future__ import annotations

from typing import Optional

import torch


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    """Flatten leading dims so ``x`` is ``[n_rows, n_features]``."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    return x.reshape(-1, x.shape[-1])


class ActivationStats:
    """Online per-channel statistics accumulator (Welford).

    Parameters
    ----------
    n_features : int, optional
        Channel count. Inferred from the first ``update`` if omitted.
    reservoir_size : int
        Max rows retained for robust (median/MAD) estimation. Set to 0 to
        disable robust stats and save memory.
    seed : int
        Seed for reservoir sampling reproducibility.

    Notes
    -----
    ``update`` accepts any tensor whose last dim is the channel dim; leading
    dims are flattened into rows.
    """

    def __init__(self, n_features: Optional[int] = None,
                 reservoir_size: int = 4096, seed: int = 0):
        self.n_features = n_features
        self.count = 0
        self._mean = None  # [n_features]
        self._M2 = None    # [n_features] sum of squared deviations
        self.reservoir_size = reservoir_size
        self._reservoir = None  # [<=reservoir_size, n_features]
        self._seen = 0  # total rows seen (for reservoir sampling)
        self._gen = torch.Generator().manual_seed(seed)

    def _init(self, n_features: int, dtype, device):
        self.n_features = n_features
        self._mean = torch.zeros(n_features, dtype=torch.float64)
        self._M2 = torch.zeros(n_features, dtype=torch.float64)
        if self.reservoir_size > 0:
            self._reservoir = torch.empty(0, n_features, dtype=torch.float32)

    def update(self, batch: torch.Tensor) -> "ActivationStats":
        """Fold a batch of rows into the running statistics.

        ``batch`` is ``[..., n_features]``; leading dims become rows.
        """
        x = _as_2d(batch).to(torch.float64)
        if self._mean is None:
            self._init(x.shape[1], x.dtype, x.device)
        elif x.shape[1] != self.n_features:
            raise ValueError(
                f"feature dim {x.shape[1]} != expected {self.n_features}")

        # Vectorized Welford (batch update, Chan et al. parallel form).
        n_b = x.shape[0]
        if n_b == 0:
            return self
        mean_b = x.mean(dim=0)
        M2_b = ((x - mean_b) ** 2).sum(dim=0)
        delta = mean_b - self._mean
        total = self.count + n_b
        self._mean = self._mean + delta * (n_b / total)
        self._M2 = self._M2 + M2_b + delta ** 2 * (self.count * n_b / total)
        self.count = total

        if self.reservoir_size > 0:
            self._update_reservoir(_as_2d(batch).to(torch.float32))
        return self

    def _update_reservoir(self, x: torch.Tensor):
        """Reservoir-sample rows for robust stats (bounded memory)."""
        for i in range(x.shape[0]):
            self._seen += 1
            if self._reservoir.shape[0] < self.reservoir_size:
                self._reservoir = torch.cat([self._reservoir, x[i:i + 1]], dim=0)
            else:
                j = int(torch.randint(0, self._seen, (1,), generator=self._gen))
                if j < self.reservoir_size:
                    self._reservoir[j] = x[i]

    def finalize(self) -> dict:
        """Return computed statistics.

        Returns
        -------
        dict with keys:
            ``count`` (int), ``mean`` [n_features], ``var`` [n_features]
            (sample variance, ddof=1 when count>1), ``std`` [n_features].
            If a reservoir was kept: ``median`` and ``mad`` (median absolute
            deviation, unscaled) [n_features].

        All returned tensors are float32.
        """
        if self._mean is None or self.count == 0:
            raise ValueError("No data accumulated; call update() first.")
        var = self._M2 / (self.count - 1) if self.count > 1 else torch.zeros_like(self._M2)
        out = {
            "count": self.count,
            "mean": self._mean.to(torch.float32),
            "var": var.to(torch.float32),
            "std": var.clamp_min(0).sqrt().to(torch.float32),
        }
        if self.reservoir_size > 0 and self._reservoir is not None and \
                self._reservoir.shape[0] > 0:
            res = self._reservoir
            median = res.median(dim=0).values
            mad = (res - median).abs().median(dim=0).values
            out["median"] = median.to(torch.float32)
            out["mad"] = mad.to(torch.float32)
        return out


def zscore(x: torch.Tensor, stats: dict, kind: str = "standard",
           var_floor: float = 1e-6) -> torch.Tensor:
    """Z-score ``x`` against precomputed ``stats``.

    Parameters
    ----------
    x : ``[..., n_features]`` tensor.
    stats : dict from :meth:`ActivationStats.finalize`.
    kind : "standard" uses (x - mean) / std; "robust" uses
        (x - median) / (1.4826 * MAD), the MAD scaled to a Gaussian-consistent
        estimate of std.
    var_floor : minimum variance. **Mandatory flooring**: the denominator std
        is clamped to ``sqrt(var_floor)`` so near-constant channels do not
        produce infinite z-scores.

    Returns
    -------
    Tensor same shape as ``x``.
    """
    floor_std = float(var_floor) ** 0.5
    if kind == "standard":
        mean = stats["mean"]
        std = stats["std"].clamp_min(floor_std)
        return (x - mean) / std
    if kind == "robust":
        if "median" not in stats:
            raise ValueError("robust z-score needs median/MAD; enable reservoir.")
        median = stats["median"]
        scale = (1.4826 * stats["mad"]).clamp_min(floor_std)
        return (x - median) / scale
    raise ValueError(f"kind must be 'standard' or 'robust'; got {kind!r}")
