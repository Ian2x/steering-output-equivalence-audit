"""Channel ranking by magnitude and z-score methods.

Given activations (or precomputed stats) for a single site/layer, rank channels
by how "salient" they are under several methods, and optionally exclude
massive-activation / attention-sink channels that dominate raw magnitude but
carry little input-dependent signal.

The premise is that *raw magnitude* and *(gain-weighted)
z-score* rank channels differently: a high-variance, high-mean channel can
dominate raw magnitude while a low-magnitude but highly input-selective channel
dominates z-score. This module makes that comparison mechanical.
"""

from __future__ import annotations

from typing import Optional

import torch

from .stats import ActivationStats

METHODS = (
    "raw_magnitude",
    "abs_zscore",
    "robust_zscore",
    "gain_weighted_zscore",
    "outlier_frequency",
)


def _stats_from(activations_or_stats):
    """Accept a ``[n_rows, hidden]`` tensor or a finalized stats dict.

    Returns ``(stats_dict, activations_or_None)``. When given raw activations,
    stats are computed here (with a robust reservoir).
    """
    if isinstance(activations_or_stats, dict):
        return activations_or_stats, None
    x = activations_or_stats
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    acc = ActivationStats(reservoir_size=min(4096, x.shape[0]))
    acc.update(x)
    return acc.finalize(), x


def _sink_mask(stats: dict, activations: Optional[torch.Tensor],
               mag_percentile: float = 99.0,
               cv_threshold: float = 0.05,
               var_floor: float = 1e-6) -> torch.Tensor:
    """Boolean mask (True = sink channel to exclude).

    Criterion (documented control): a channel is treated
    as a massive-activation / attention-sink channel when BOTH

      1. its mean absolute magnitude exceeds the ``mag_percentile`` percentile
         across channels (it is "massive"), AND
      2. it is near-constant across inputs: coefficient of variation
         ``std / (|mean| + eps) < cv_threshold`` (it barely responds to input).

    Requiring both conditions avoids excluding channels that are merely large
    but still input-selective. Uses precomputed mean/std; ``activations`` is
    unused here but accepted for API symmetry.
    """
    mean = stats["mean"]
    std = stats["std"]
    abs_mean = mean.abs()
    thresh = torch.quantile(abs_mean, mag_percentile / 100.0)
    massive = abs_mean >= thresh
    cv = std / (abs_mean + var_floor ** 0.5)
    near_constant = cv < cv_threshold
    return massive & near_constant


def sink_channels(stats: dict, mag_percentile: float = 99.0,
                  cv_threshold: float = 0.05,
                  var_floor: float = 1e-6) -> torch.Tensor:
    """Public wrapper around the sink criterion: indices of channels that are
    BOTH >``mag_percentile``-pct mean-|magnitude| AND near-constant
    (CoV < ``cv_threshold``). Use to apply a *corpus-level* sink mask to a
    ranking computed on a small prompt set (where per-set sink stats would be
    noisy)."""
    mask = _sink_mask(stats, None, mag_percentile=mag_percentile,
                      cv_threshold=cv_threshold, var_floor=var_floor)
    return mask.nonzero(as_tuple=False).flatten()


class StreamingChannelScores:
    """One-pass streaming ranking scores over a corpus.

    Holding all activations to call :func:`rank_channels` is infeasible over a
    multi-100k-token corpus; this accumulates the per-channel score ingredients
    online. Z-score methods require an external ``ref_stats`` baseline (the
    reference-baseline setup); self-referential z-scoring would need two passes
    — use :func:`rank_channels` on a finalized stats dict for that.

    Accumulated per channel: E|x| (``raw_magnitude``), E|z| against ``ref_stats``
    (``abs_zscore``; scaled by ``|gains|`` for ``gain_weighted_zscore``),
    P(|z| > ``outlier_z``) (``outlier_frequency``), E|z_robust| against the ref
    median/MAD (``robust_zscore``, if ref has a reservoir), plus full
    :class:`ActivationStats` of the test stream (used for the sink mask).
    """

    def __init__(self, ref_stats: Optional[dict] = None,
                 var_floor: float = 1e-6, outlier_z: float = 4.0,
                 reservoir_size: int = 0, seed: int = 0):
        self.ref = ref_stats
        self.var_floor = var_floor
        self.outlier_z = outlier_z
        self.stats = ActivationStats(reservoir_size=reservoir_size, seed=seed)
        self._n = 0
        self._abs_sum = None
        self._absz_sum = None
        self._absrz_sum = None
        self._outlier_count = None
        if ref_stats is not None:
            floor_std = float(var_floor) ** 0.5
            self._ref_mean = ref_stats["mean"].to(torch.float64)
            self._ref_std = ref_stats["std"].to(torch.float64).clamp_min(floor_std)
            if "median" in ref_stats:
                self._ref_med = ref_stats["median"].to(torch.float64)
                self._ref_scale = (1.4826 * ref_stats["mad"].to(torch.float64)
                                   ).clamp_min(floor_std)
            else:
                self._ref_med = None

    def update(self, batch: torch.Tensor) -> "StreamingChannelScores":
        x = batch.reshape(-1, batch.shape[-1]).to(torch.float64)
        if x.shape[0] == 0:
            return self
        self.stats.update(x)
        if self._abs_sum is None:
            h = x.shape[1]
            self._abs_sum = torch.zeros(h, dtype=torch.float64)
            self._absz_sum = torch.zeros(h, dtype=torch.float64)
            self._absrz_sum = torch.zeros(h, dtype=torch.float64)
            self._outlier_count = torch.zeros(h, dtype=torch.float64)
        self._abs_sum += x.abs().sum(dim=0)
        if self.ref is not None:
            z = (x - self._ref_mean) / self._ref_std
            self._absz_sum += z.abs().sum(dim=0)
            self._outlier_count += (z.abs() > self.outlier_z).sum(dim=0)
            if self._ref_med is not None:
                rz = (x - self._ref_med) / self._ref_scale
                self._absrz_sum += rz.abs().sum(dim=0)
        self._n += x.shape[0]
        return self

    def rank(self, method: str, gains: Optional[torch.Tensor] = None,
             exclude_sinks: bool = True, top_k: Optional[int] = None) -> dict:
        """Rank channels from the accumulated scores (same output shape as
        :func:`rank_channels`)."""
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}. Methods: {METHODS}")
        if self._n == 0:
            raise ValueError("No data accumulated; call update() first.")
        needs_ref = method in ("abs_zscore", "robust_zscore",
                               "gain_weighted_zscore", "outlier_frequency")
        if needs_ref and self.ref is None:
            raise ValueError(
                f"{method} requires ref_stats in streaming mode (one-pass).")
        if method == "raw_magnitude":
            scores = self._abs_sum / self._n
        elif method == "abs_zscore":
            scores = self._absz_sum / self._n
        elif method == "robust_zscore":
            if self._ref_med is None:
                raise ValueError("robust_zscore needs ref median/MAD.")
            scores = self._absrz_sum / self._n
        elif method == "gain_weighted_zscore":
            if gains is None:
                raise ValueError("gain_weighted_zscore requires `gains`.")
            scores = (self._absz_sum / self._n) * gains.abs().to(torch.float64)
        elif method == "outlier_frequency":
            scores = self._outlier_count / self._n
        scores = scores.float().clone()

        excluded = torch.empty(0, dtype=torch.long)
        if exclude_sinks:
            mask = _sink_mask(self.stats.finalize(), None,
                              var_floor=self.var_floor)
            excluded = mask.nonzero(as_tuple=False).flatten()
            scores[mask] = float("-inf")
        order = torch.argsort(scores, descending=True)
        if excluded.numel() > 0:
            excluded_set = set(excluded.tolist())
            order = torch.tensor([i for i in order.tolist()
                                  if i not in excluded_set], dtype=torch.long)
        if top_k is not None:
            order = order[:top_k]
        return {"indices": order, "scores": scores[order], "method": method,
                "excluded": excluded}


def rank_channels(
    activations_or_stats,
    method: str,
    gains: Optional[torch.Tensor] = None,
    exclude_sinks: bool = True,
    var_floor: float = 1e-6,
    top_k: Optional[int] = None,
    outlier_z: float = 4.0,
    ref_stats: Optional[dict] = None,
):
    """Rank channels by salience under ``method``.

    Parameters
    ----------
    activations_or_stats : either a ``[n_rows, hidden]`` activation tensor for a
        single (layer, site), or a finalized stats dict from
        :meth:`ActivationStats.finalize`. Methods needing per-row data
        (``outlier_frequency``) require the raw tensor.
    method : one of :data:`METHODS`.
        - ``raw_magnitude``      : mean absolute activation per channel.
        - ``abs_zscore``         : mean |standard z-score| across rows
          (needs raw activations) OR |mean|/std (from stats only).
        - ``robust_zscore``      : mean |robust (MAD) z-score| across rows.
        - ``gain_weighted_zscore``: abs_zscore scaled by the norm gain magnitude
          (``gains`` required: a ``[hidden]`` vector).
        - ``outlier_frequency``  : fraction of rows where |standard z| exceeds
          ``outlier_z`` (needs raw activations).
    gains : ``[hidden]`` norm-gain vector, required for ``gain_weighted_zscore``.
    exclude_sinks : drop massive-activation/sink channels (see :func:`_sink_mask`)
        before ranking; their scores are set to ``-inf`` so they never rank.
    var_floor : mandatory variance floor for z-scoring (prevents inf).
    top_k : if set, return only the top-k channels.
    outlier_z : z threshold for ``outlier_frequency``.
    ref_stats : optional finalized stats dict used as the *reference* baseline
        for centering/scaling all z-score methods (standard, robust,
        gain-weighted, outlier-frequency). The core use case:
        rank a *test* batch's channels by how far they deviate from a benign
        reference corpus. When omitted, each channel is z-scored against its own
        stats (self-referential), so a low-magnitude channel that is merely
        Gaussian will NOT rank high. Pass ``ref_stats`` to surface channels that
        shift strongly from baseline despite small raw magnitude.

    Returns
    -------
    dict with:
        ``indices`` : LongTensor of channel indices, best first.
        ``scores``  : FloatTensor of the corresponding scores (same order).
        ``method``  : the method string.
        ``excluded``: LongTensor of channel indices excluded as sinks.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Methods: {METHODS}")

    stats, acts = _stats_from(activations_or_stats)
    hidden = stats["mean"].shape[0]
    floor_std = float(var_floor) ** 0.5

    # Reference stats used for z-score centering/scaling: external baseline if
    # provided, else the channels' own stats (self-referential).
    ref = ref_stats if ref_stats is not None else stats
    ref_mean = ref["mean"]
    std = ref["std"].clamp_min(floor_std)

    if method == "raw_magnitude":
        if acts is not None:
            scores = acts.abs().mean(dim=0)
        else:
            # From stats: E[|x|] unavailable; use |mean| + std as a proxy of
            # typical magnitude.
            scores = stats["mean"].abs() + stats["std"]
    elif method == "abs_zscore":
        if acts is not None:
            z = (acts - ref_mean) / std
            scores = z.abs().mean(dim=0)
        else:
            scores = (stats["mean"] - ref_mean).abs() / std
    elif method == "robust_zscore":
        if "median" not in ref:
            raise ValueError("robust_zscore needs median/MAD (reservoir stats).")
        median = ref["median"]
        scale = (1.4826 * ref["mad"]).clamp_min(floor_std)
        if acts is not None:
            z = (acts - median) / scale
            scores = z.abs().mean(dim=0)
        else:
            scores = (stats["mean"] - median).abs() / scale
    elif method == "gain_weighted_zscore":
        if gains is None:
            raise ValueError("gain_weighted_zscore requires `gains`.")
        if gains.shape[-1] != hidden:
            raise ValueError(
                f"gains dim {gains.shape[-1]} != hidden {hidden}")
        if acts is not None:
            z = (acts - ref_mean) / std
            base = z.abs().mean(dim=0)
        else:
            base = (stats["mean"] - ref_mean).abs() / std
        scores = base * gains.abs()
    elif method == "outlier_frequency":
        if acts is None:
            raise ValueError("outlier_frequency requires raw activations.")
        z = (acts - ref_mean) / std
        scores = (z.abs() > outlier_z).float().mean(dim=0)

    scores = scores.clone().float()

    excluded = torch.empty(0, dtype=torch.long)
    if exclude_sinks:
        mask = _sink_mask(stats, acts, var_floor=var_floor)
        excluded = mask.nonzero(as_tuple=False).flatten()
        scores[mask] = float("-inf")

    order = torch.argsort(scores, descending=True)
    # Drop excluded (sink) channels entirely from the ranked output; they carry
    # -inf scores and should never appear as ranked candidates.
    if excluded.numel() > 0:
        excluded_set = set(excluded.tolist())
        order = torch.tensor([i for i in order.tolist()
                              if i not in excluded_set], dtype=torch.long)
    if top_k is not None:
        order = order[:top_k]
    return {
        "indices": order,
        "scores": scores[order],
        "method": method,
        "excluded": excluded,
    }
