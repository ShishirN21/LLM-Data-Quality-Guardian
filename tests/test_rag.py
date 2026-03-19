"""Tests for the RAG quality module."""

from __future__ import annotations

import numpy as np
import pytest

from guardian.models import Document
from guardian.rag.chunking import ChunkQualityAnalyzer
from guardian.rag.context import ContextCoverageAnalyzer


class TestChunkQualityAnalyzer:
    def test_chunk_size_analysis(self, sample_documents: list[Document], config: dict) -> None:
        analyzer = ChunkQualityAnalyzer(config["rag"])
        result = analyzer.analyze_chunk_sizes(sample_documents)

        assert result.metric_name == "chunk_size_quality"
        assert 0.0 <= result.score <= 1.0
        assert result.details["total_chunks"] == len(sample_documents)

    def test_chunk_coherence(
        self, sample_documents: list[Document], sample_embeddings: np.ndarray, config: dict
    ) -> None:
        analyzer = ChunkQualityAnalyzer(config["rag"])
        result = analyzer.analyze_chunk_coherence(sample_documents, sample_embeddings)

        assert result.metric_name == "chunk_coherence"
        assert result.score >= 0.0

    def test_empty_documents(self, config: dict) -> None:
        analyzer = ChunkQualityAnalyzer(config["rag"])
        result = analyzer.analyze_chunk_sizes([])
        assert result.score == 0.0

    def test_run_all(
        self, sample_documents: list[Document], sample_embeddings: np.ndarray, config: dict
    ) -> None:
        analyzer = ChunkQualityAnalyzer(config["rag"])
        results = analyzer.run_all(sample_documents, sample_embeddings)
        assert len(results) == 2


class TestContextCoverageAnalyzer:
    def test_coverage_analysis(self, config: dict) -> None:
        rng = np.random.RandomState(42)
        query_emb = rng.randn(384).astype(np.float32)
        doc_embs = rng.randn(5, 384).astype(np.float32)

        analyzer = ContextCoverageAnalyzer(config["rag"])
        result = analyzer.analyze_coverage(query_emb, doc_embs)

        assert result.metric_name == "context_coverage"
        assert 0.0 <= result.score <= 1.0

    def test_redundancy_detection(self, config: dict) -> None:
        rng = np.random.RandomState(42)
        base = rng.randn(1, 384).astype(np.float32)
        # Very similar documents (high redundancy)
        doc_embs = np.vstack([base + 0.01 * rng.randn(1, 384) for _ in range(5)]).astype(
            np.float32
        )

        analyzer = ContextCoverageAnalyzer(config["rag"])
        result = analyzer.detect_redundancy(doc_embs)

        assert result.metric_name == "context_redundancy"
        assert result.score < 0.3  # high redundancy = low diversity score

    def test_empty_docs(self, config: dict) -> None:
        rng = np.random.RandomState(42)
        query_emb = rng.randn(384).astype(np.float32)

        analyzer = ContextCoverageAnalyzer(config["rag"])
        result = analyzer.analyze_coverage(query_emb, np.array([]).reshape(0, 384))
        assert result.score == 0.0
