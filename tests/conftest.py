"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from guardian.models import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create a set of sample documents for testing."""
    return [
        Document(
            doc_id=f"test_{i:03d}",
            content=content,
            source_path="test/data.csv",
            metadata={"category": category},
        )
        for i, (content, category) in enumerate(
            [
                ("Python is a popular programming language for data science.", "programming"),
                ("Machine learning models learn patterns from data.", "ai"),
                ("PostgreSQL is a relational database system.", "database"),
                ("Neural networks are inspired by the human brain.", "ai"),
                ("Docker containers provide lightweight virtualization.", "devops"),
                ("Data quality is essential for reliable ML models.", "ai"),
                ("The transformer architecture revolutionized NLP.", "ai"),
                ("Redis is an in-memory data structure store.", "database"),
                ("Python does not support true multi-threading for CPU tasks.", "programming"),
                ("Kubernetes orchestrates containerized applications.", "devops"),
            ]
        )
    ]


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """384-dimensional random embeddings matching MiniLM output dimensions."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 384).astype(np.float32)


@pytest.fixture
def drifted_embeddings() -> np.ndarray:
    """Embeddings with a different distribution (shifted mean)."""
    rng = np.random.RandomState(99)
    return (rng.randn(10, 384) + 2.0).astype(np.float32)


@pytest.fixture
def config() -> dict:
    """Test configuration."""
    return {
        "embeddings": {
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "cache_dir": None,
        },
        "vector_store": {
            "persist_directory": ".test_data/chromadb",
            "collection_name": "test_docs",
        },
        "metrics_store": {"database_path": ":memory:"},
        "drift": {
            "baseline_window_days": 30,
            "detection_window_days": 7,
            "ks_test_alpha": 0.05,
            "cosine_sim_threshold": 0.85,
            "mmd_threshold": 0.1,
            "min_samples": 5,
        },
        "hallucination": {
            "contradiction_sim_threshold": 0.8,
            "contradiction_neg_threshold": 0.3,
            "entity_ambiguity_threshold": 2,
            "temporal_staleness_days": 365,
            "sparse_context_min_neighbors": 3,
            "sparse_context_radius": 0.5,
        },
        "quality": {
            "freshness_warning_hours": 24,
            "freshness_critical_hours": 72,
        },
        "rag": {
            "relevance_threshold": 0.7,
            "chunk_size_target": 512,
            "chunk_overlap": 50,
            "context_coverage_min": 0.6,
            "top_k": 5,
        },
        "alerts": {
            "console_enabled": False,
            "webhook_enabled": False,
            "severity_threshold": "WARNING",
        },
    }


@pytest.fixture
def tmp_metrics_store(tmp_path):
    """MetricsStore backed by a temp SQLite database."""
    from guardian.storage.metrics_store import MetricsStore

    return MetricsStore(db_path=str(tmp_path / "test_metrics.db"))
