"""Kolmogorov-Smirnov test for per-dimension embedding drift."""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from guardian.models import DriftResult


class KSTestStrategy:
    """Per-dimension KS test on embedding vectors.

    For each dimension of the embedding space, runs a two-sample KS test
    comparing the baseline and current distributions. Reports the fraction
    of dimensions showing significant drift.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha

    def test(self, baseline: np.ndarray, current: np.ndarray) -> DriftResult:
        n_dims = baseline.shape[1]
        drifted_dims = 0
        max_stat = 0.0
        min_p = 1.0
        dim_results: list[dict] = []

        for dim in range(n_dims):
            stat, p_value = ks_2samp(baseline[:, dim], current[:, dim])
            if p_value < self._alpha:
                drifted_dims += 1
            max_stat = max(max_stat, stat)
            min_p = min(min_p, p_value)
            dim_results.append({"dim": dim, "statistic": round(stat, 4), "p_value": round(p_value, 4)})

        drifted_fraction = drifted_dims / n_dims
        # Drift if more than 10% of dimensions show significant shift
        is_drifted = drifted_fraction > 0.10

        return DriftResult(
            test_name="ks_test",
            statistic=round(drifted_fraction, 4),
            p_value=round(min_p, 6),
            threshold=0.10,
            is_drifted=is_drifted,
            details={
                "drifted_dimensions": drifted_dims,
                "total_dimensions": n_dims,
                "drifted_fraction": round(drifted_fraction, 4),
                "max_ks_statistic": round(max_stat, 4),
                "alpha": self._alpha,
                "top_drifted": sorted(
                    dim_results, key=lambda x: x["p_value"]
                )[:10],
            },
        )
