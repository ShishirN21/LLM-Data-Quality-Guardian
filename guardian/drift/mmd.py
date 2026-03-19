"""Maximum Mean Discrepancy (MMD) for distribution-level drift detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

from guardian.models import DriftResult


class MMDStrategy:
    """MMD with RBF kernel and median bandwidth heuristic.

    Computes MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where k is the RBF kernel with bandwidth selected via the median heuristic.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self._threshold = threshold

    def test(self, baseline: np.ndarray, current: np.ndarray) -> DriftResult:
        # Subsample if too large (for performance)
        max_samples = 500
        if len(baseline) > max_samples:
            idx = np.random.choice(len(baseline), max_samples, replace=False)
            baseline = baseline[idx]
        if len(current) > max_samples:
            idx = np.random.choice(len(current), max_samples, replace=False)
            current = current[idx]

        # Median heuristic for RBF bandwidth
        all_data = np.vstack([baseline, current])
        dists = euclidean_distances(all_data, all_data)
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * median_dist**2) if median_dist > 0 else 1.0

        # Compute kernel matrices
        k_xx = rbf_kernel(baseline, baseline, gamma=gamma)
        k_yy = rbf_kernel(current, current, gamma=gamma)
        k_xy = rbf_kernel(baseline, current, gamma=gamma)

        # MMD^2 estimate
        mmd_squared = float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())
        mmd_squared = max(mmd_squared, 0.0)  # numerical stability

        is_drifted = mmd_squared > self._threshold

        return DriftResult(
            test_name="mmd",
            statistic=round(mmd_squared, 6),
            p_value=None,
            threshold=self._threshold,
            is_drifted=is_drifted,
            details={
                "gamma": round(gamma, 6),
                "median_distance": round(float(median_dist), 4),
                "baseline_samples": len(baseline),
                "current_samples": len(current),
            },
        )
