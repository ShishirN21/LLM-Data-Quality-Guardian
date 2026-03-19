"""Tests for the hallucination pattern detection module."""

from __future__ import annotations

import numpy as np
import pytest

from guardian.hallucination.contradiction import ContradictionDetector
from guardian.hallucination.temporal import TemporalDetector
from guardian.hallucination.sparse import SparseContextDetector
from guardian.models import Document


class TestContradictionDetector:
    def test_detects_negation_contradiction(self, config: dict) -> None:
        docs = [
            Document(doc_id="a", content="Neural networks are inspired by the human brain."),
            Document(doc_id="b", content="Neural networks are not inspired by biological neurons."),
        ]
        # Create embeddings that are very similar (simulating high cosine sim)
        rng = np.random.RandomState(42)
        base = rng.randn(1, 384).astype(np.float32)
        embeddings = np.vstack([base, base + 0.01 * rng.randn(1, 384)]).astype(np.float32)

        detector = ContradictionDetector(
            {"contradiction_sim_threshold": 0.5}, None  # type: ignore
        )
        risks = detector.detect(docs, embeddings)

        assert len(risks) >= 1
        assert risks[0].risk_type == "contradiction"

    def test_no_contradiction(self, config: dict) -> None:
        docs = [
            Document(doc_id="a", content="Python is great for data science."),
            Document(doc_id="b", content="Java is widely used in enterprise software."),
        ]
        rng = np.random.RandomState(42)
        embeddings = rng.randn(2, 384).astype(np.float32)

        detector = ContradictionDetector(
            {"contradiction_sim_threshold": 0.95}, None  # type: ignore
        )
        risks = detector.detect(docs, embeddings)
        assert len(risks) == 0


class TestTemporalDetector:
    def test_detects_stale_content(self, config: dict) -> None:
        docs = [
            Document(
                doc_id="old",
                content="As of 2018, the latest version of Python is 3.6.",
            ),
        ]
        embeddings = np.random.randn(1, 384).astype(np.float32)

        detector = TemporalDetector({"temporal_staleness_days": 365})
        risks = detector.detect(docs, embeddings)

        assert len(risks) >= 1
        assert risks[0].risk_type == "temporal"

    def test_no_staleness(self, config: dict) -> None:
        docs = [
            Document(
                doc_id="fresh",
                content="In 2025, AI continues to advance rapidly.",
            ),
        ]
        embeddings = np.random.randn(1, 384).astype(np.float32)

        detector = TemporalDetector({"temporal_staleness_days": 365})
        risks = detector.detect(docs, embeddings)
        assert len(risks) == 0


class TestSparseContextDetector:
    def test_detects_isolated_document(self, config: dict) -> None:
        # One document far from the others
        rng = np.random.RandomState(42)
        cluster = rng.randn(8, 384).astype(np.float32)
        outlier = (rng.randn(1, 384) * 10 + 20).astype(np.float32)
        embeddings = np.vstack([cluster, outlier])

        docs = [Document(doc_id=f"d{i}", content=f"Doc {i}") for i in range(9)]

        detector = SparseContextDetector(
            {"sparse_context_min_neighbors": 3, "sparse_context_radius": 0.5},
            None,  # type: ignore
        )
        risks = detector.detect(docs, embeddings)

        # The outlier should be flagged
        assert any(r.risk_type == "sparse_context" for r in risks)
