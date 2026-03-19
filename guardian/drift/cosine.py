"""Centroid-based cosine similarity drift detection."""

from __future__ import annotations

import numpy as np

from guardian.models import DriftResult


class CosineCentroidStrategy:
    """Compare embedding centroids between baseline and current data.

    Computes the cosine similarity between the mean embedding vectors
    of the baseline and current sets. Simple but highly interpretable.
    """

    def __init__(self, threshold: float = 0.85) -> None:
        self._threshold = threshold

    def test(self, baseline: np.ndarray, current: np.ndarray) -> DriftResult:
        centroid_baseline = np.mean(baseline, axis=0)
        centroid_current = np.mean(current, axis=0)

        dot = np.dot(centroid_baseline, centroid_current)
        norm_b = np.linalg.norm(centroid_baseline)
        norm_c = np.linalg.norm(centroid_current)
        similarity = float(dot / (norm_b * norm_c)) if (norm_b * norm_c) > 0 else 0.0

        is_drifted = similarity < self._threshold

        return DriftResult(
            test_name="cosine_centroid",
            statistic=round(similarity, 6),
            p_value=None,
            threshold=self._threshold,
            is_drifted=is_drifted,
            details={
                "baseline_centroid_norm": round(float(norm_b), 4),
                "current_centroid_norm": round(float(norm_c), 4),
                "baseline_samples": len(baseline),
                "current_samples": len(current),
            },
        )
