"""Tests for the storage module."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from guardian.models import (
    Alert,
    DriftResult,
    HallucinationRisk,
    PipelineRunResult,
    QualityCheckResult,
    RAGQualityResult,
    Severity,
)
from guardian.storage.metrics_store import MetricsStore


class TestMetricsStore:
    def test_save_and_retrieve_run(self, tmp_metrics_store: MetricsStore) -> None:
        now = datetime.now(timezone.utc)
        run = PipelineRunResult(
            run_id="test_001",
            started_at=now,
            completed_at=now,
            documents_processed=10,
            drift_results=[
                DriftResult(
                    test_name="ks_test",
                    statistic=0.15,
                    p_value=0.03,
                    threshold=0.1,
                    is_drifted=True,
                )
            ],
            hallucination_risks=[
                HallucinationRisk(
                    risk_type="contradiction",
                    severity=Severity.WARNING,
                    doc_ids=["d1", "d2"],
                    description="Test contradiction",
                    confidence=0.8,
                )
            ],
            quality_results=[
                QualityCheckResult(
                    check_name="row_count",
                    dataset="documents",
                    passed=True,
                    metric_value=10.0,
                )
            ],
            rag_results=[
                RAGQualityResult(
                    metric_name="retrieval_relevance",
                    score=0.85,
                )
            ],
            alerts=[
                Alert(
                    severity=Severity.WARNING,
                    source_module="drift",
                    title="Test alert",
                    message="This is a test alert",
                )
            ],
        )

        tmp_metrics_store.save_run(run)
        latest = tmp_metrics_store.get_latest_run()

        assert latest is not None
        assert latest.run_id == "test_001"
        assert latest.documents_processed == 10
        assert len(latest.drift_results) == 1
        assert latest.drift_results[0].is_drifted
        assert len(latest.hallucination_risks) == 1
        assert len(latest.quality_results) == 1
        assert len(latest.rag_results) == 1
        assert len(latest.alerts) == 1

    def test_get_drift_history(self, tmp_metrics_store: MetricsStore) -> None:
        now = datetime.now(timezone.utc)
        run = PipelineRunResult(
            run_id="drift_test",
            started_at=now,
            completed_at=now,
            drift_results=[
                DriftResult(test_name="ks_test", statistic=0.05, threshold=0.1, is_drifted=False),
                DriftResult(test_name="mmd", statistic=0.08, threshold=0.1, is_drifted=False),
            ],
        )
        tmp_metrics_store.save_run(run)

        df = tmp_metrics_store.get_drift_history(days=1)
        assert len(df) == 2

        df_ks = tmp_metrics_store.get_drift_history(test_name="ks_test", days=1)
        assert len(df_ks) == 1

    def test_empty_store(self, tmp_metrics_store: MetricsStore) -> None:
        assert tmp_metrics_store.get_latest_run() is None
        df = tmp_metrics_store.get_drift_history(days=1)
        assert df.empty
