"""Tests for the drift detection module."""

from __future__ import annotations

import numpy as np
import pytest

from guardian.drift.cosine import CosineCentroidStrategy
from guardian.drift.detector import DriftDetector
from guardian.drift.ks_test import KSTestStrategy
from guardian.drift.mmd import MMDStrategy


class TestKSTestStrategy:
    def test_no_drift(self) -> None:
        rng = np.random.RandomState(42)
        baseline = rng.randn(100, 384).astype(np.float32)
        current = rng.randn(100, 384).astype(np.float32)

        strategy = KSTestStrategy(alpha=0.05)
        result = strategy.test(baseline, current)

        assert result.test_name == "ks_test"
        assert not result.is_drifted  # same distribution

    def test_drift_detected(self) -> None:
        rng = np.random.RandomState(42)
        baseline = rng.randn(100, 384).astype(np.float32)
        current = (rng.randn(100, 384) + 3.0).astype(np.float32)  # shifted mean

        strategy = KSTestStrategy(alpha=0.05)
        result = strategy.test(baseline, current)

        assert result.is_drifted
        assert result.details["drifted_fraction"] > 0.1


class TestMMDStrategy:
    def test_no_drift(self) -> None:
        rng = np.random.RandomState(42)
        baseline = rng.randn(50, 384).astype(np.float32)
        current = rng.randn(50, 384).astype(np.float32)

        strategy = MMDStrategy(threshold=0.1)
        result = strategy.test(baseline, current)

        assert result.test_name == "mmd"
        assert result.statistic >= 0  # MMD^2 is non-negative

    def test_drift_detected(self) -> None:
        rng = np.random.RandomState(42)
        baseline = rng.randn(50, 384).astype(np.float32)
        current = (rng.randn(50, 384) + 5.0).astype(np.float32)

        strategy = MMDStrategy(threshold=0.01)
        result = strategy.test(baseline, current)

        assert result.is_drifted


class TestCosineCentroidStrategy:
    def test_no_drift(self) -> None:
        rng = np.random.RandomState(42)
        # Create data with a clear shared centroid direction
        mean = rng.randn(384).astype(np.float32)
        baseline = (mean + 0.1 * rng.randn(50, 384)).astype(np.float32)
        current = (mean + 0.1 * rng.randn(50, 384)).astype(np.float32)

        strategy = CosineCentroidStrategy(threshold=0.5)
        result = strategy.test(baseline, current)

        assert result.test_name == "cosine_centroid"
        assert result.statistic > 0.9  # should be very similar

    def test_drift_detected(self) -> None:
        rng = np.random.RandomState(42)
        baseline = rng.randn(50, 384).astype(np.float32)
        current = -baseline  # opposite direction

        strategy = CosineCentroidStrategy(threshold=0.85)
        result = strategy.test(baseline, current)

        assert result.is_drifted
        assert result.statistic < 0


class TestDriftDetector:
    def test_runs_all_strategies(self, config: dict) -> None:
        rng = np.random.RandomState(42)
        baseline = rng.randn(50, 384).astype(np.float32)
        current = rng.randn(50, 384).astype(np.float32)

        detector = DriftDetector(config["drift"])
        results = detector.detect(baseline, current)

        assert len(results) == 3
        test_names = {r.test_name for r in results}
        assert test_names == {"ks_test", "mmd", "cosine_centroid"}
