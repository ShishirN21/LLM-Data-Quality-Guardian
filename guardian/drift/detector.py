"""Drift detection orchestrator using Strategy pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from guardian.models import DriftResult

from guardian.drift.cosine import CosineCentroidStrategy
from guardian.drift.ks_test import KSTestStrategy
from guardian.drift.mmd import MMDStrategy


class DriftStrategy(ABC):
    """Abstract base for drift detection algorithms."""

    @abstractmethod
    def test(self, baseline: np.ndarray, current: np.ndarray) -> DriftResult:
        """Run drift test between baseline and current embedding sets."""


class DriftDetector:
    """Orchestrates multiple drift detection strategies."""

    def __init__(self, config: dict) -> None:
        self._strategies: list[DriftStrategy] = [
            KSTestStrategy(alpha=config.get("ks_test_alpha", 0.05)),
            MMDStrategy(threshold=config.get("mmd_threshold", 0.1)),
            CosineCentroidStrategy(
                threshold=config.get("cosine_sim_threshold", 0.85)
            ),
        ]

    def detect(
        self, baseline: np.ndarray, current: np.ndarray
    ) -> list[DriftResult]:
        """Run all drift detection strategies and return results."""
        results: list[DriftResult] = []
        for strategy in self._strategies:
            results.append(strategy.test(baseline, current))
        return results
