"""SQLite-backed storage for pipeline metrics and results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from guardian.models import (
    Alert,
    DriftResult,
    HallucinationRisk,
    PipelineRunResult,
    QualityCheckResult,
    RAGQualityResult,
    Severity,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    documents_processed INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS drift_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    test_name TEXT NOT NULL,
    statistic REAL NOT NULL,
    p_value REAL,
    threshold REAL NOT NULL,
    is_drifted INTEGER NOT NULL,
    details_json TEXT,
    measured_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hallucination_risks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    risk_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    doc_ids_json TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence REAL NOT NULL,
    detected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quality_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    check_name TEXT NOT NULL,
    dataset TEXT NOT NULL,
    passed INTEGER NOT NULL,
    metric_value REAL,
    details_json TEXT,
    checked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rag_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    metric_name TEXT NOT NULL,
    score REAL NOT NULL,
    details_json TEXT,
    measured_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id),
    severity TEXT NOT NULL,
    source_module TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    context_json TEXT,
    created_at TEXT NOT NULL
);
"""


class MetricsStore:
    """SQLite-backed storage for pipeline metrics and results."""

    def __init__(self, db_path: str = ".data/metrics.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ── Save ──────────────────────────────────────────────────────────

    def save_run(self, result: PipelineRunResult) -> None:
        """Persist a complete pipeline run and all its sub-results."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pipeline_runs VALUES (?, ?, ?, ?, ?)",
                (
                    result.run_id,
                    result.started_at.isoformat(),
                    result.completed_at.isoformat() if result.completed_at else None,
                    result.documents_processed,
                    "completed" if result.completed_at else "running",
                ),
            )
            for dr in result.drift_results:
                conn.execute(
                    "INSERT INTO drift_results (run_id, test_name, statistic, p_value, "
                    "threshold, is_drifted, details_json, measured_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        result.run_id,
                        dr.test_name,
                        dr.statistic,
                        dr.p_value,
                        dr.threshold,
                        int(dr.is_drifted),
                        json.dumps(dr.details),
                        dr.measured_at.isoformat(),
                    ),
                )
            for hr in result.hallucination_risks:
                conn.execute(
                    "INSERT INTO hallucination_risks (run_id, risk_type, severity, "
                    "doc_ids_json, description, confidence, detected_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        result.run_id,
                        hr.risk_type,
                        hr.severity.value,
                        json.dumps(hr.doc_ids),
                        hr.description,
                        hr.confidence,
                        hr.detected_at.isoformat(),
                    ),
                )
            for qc in result.quality_results:
                conn.execute(
                    "INSERT INTO quality_checks (run_id, check_name, dataset, passed, "
                    "metric_value, details_json, checked_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        result.run_id,
                        qc.check_name,
                        qc.dataset,
                        int(qc.passed),
                        qc.metric_value,
                        json.dumps(qc.details),
                        qc.checked_at.isoformat(),
                    ),
                )
            for rq in result.rag_results:
                conn.execute(
                    "INSERT INTO rag_metrics (run_id, metric_name, score, "
                    "details_json, measured_at) VALUES (?,?,?,?,?)",
                    (
                        result.run_id,
                        rq.metric_name,
                        rq.score,
                        json.dumps(rq.details),
                        rq.measured_at.isoformat(),
                    ),
                )
            for al in result.alerts:
                conn.execute(
                    "INSERT INTO alerts (run_id, severity, source_module, title, "
                    "message, context_json, created_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        result.run_id,
                        al.severity.value,
                        al.source_module,
                        al.title,
                        al.message,
                        json.dumps(al.context),
                        al.created_at.isoformat(),
                    ),
                )

    # ── Query ─────────────────────────────────────────────────────────

    def get_drift_history(self, test_name: str | None = None, days: int = 30) -> pd.DataFrame:
        cutoff = self._cutoff(days)
        query = "SELECT * FROM drift_results WHERE measured_at >= ?"
        params: list = [cutoff]
        if test_name:
            query += " AND test_name = ?"
            params.append(test_name)
        query += " ORDER BY measured_at"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_hallucination_risks(self, days: int = 30) -> pd.DataFrame:
        cutoff = self._cutoff(days)
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM hallucination_risks WHERE detected_at >= ? ORDER BY detected_at",
                conn,
                params=[cutoff],
            )

    def get_quality_history(self, days: int = 30) -> pd.DataFrame:
        cutoff = self._cutoff(days)
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM quality_checks WHERE checked_at >= ? ORDER BY checked_at",
                conn,
                params=[cutoff],
            )

    def get_rag_history(self, days: int = 30) -> pd.DataFrame:
        cutoff = self._cutoff(days)
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM rag_metrics WHERE measured_at >= ? ORDER BY measured_at",
                conn,
                params=[cutoff],
            )

    def get_alerts(self, severity: Severity | None = None, days: int = 7) -> pd.DataFrame:
        cutoff = self._cutoff(days)
        query = "SELECT * FROM alerts WHERE created_at >= ?"
        params: list = [cutoff]
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_latest_run(self) -> PipelineRunResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            run_id = row[0]
            drift_rows = conn.execute(
                "SELECT * FROM drift_results WHERE run_id = ?", (run_id,)
            ).fetchall()
            hall_rows = conn.execute(
                "SELECT * FROM hallucination_risks WHERE run_id = ?", (run_id,)
            ).fetchall()
            qual_rows = conn.execute(
                "SELECT * FROM quality_checks WHERE run_id = ?", (run_id,)
            ).fetchall()
            rag_rows = conn.execute(
                "SELECT * FROM rag_metrics WHERE run_id = ?", (run_id,)
            ).fetchall()
            alert_rows = conn.execute(
                "SELECT * FROM alerts WHERE run_id = ?", (run_id,)
            ).fetchall()

        return PipelineRunResult(
            run_id=run_id,
            started_at=datetime.fromisoformat(row[1]),
            completed_at=datetime.fromisoformat(row[2]) if row[2] else None,
            documents_processed=row[3],
            drift_results=[
                DriftResult(
                    test_name=r[2],
                    statistic=r[3],
                    p_value=r[4],
                    threshold=r[5],
                    is_drifted=bool(r[6]),
                    details=json.loads(r[7]) if r[7] else {},
                    measured_at=datetime.fromisoformat(r[8]),
                )
                for r in drift_rows
            ],
            hallucination_risks=[
                HallucinationRisk(
                    risk_type=r[2],
                    severity=Severity(r[3]),
                    doc_ids=json.loads(r[4]),
                    description=r[5],
                    confidence=r[6],
                    detected_at=datetime.fromisoformat(r[7]),
                )
                for r in hall_rows
            ],
            quality_results=[
                QualityCheckResult(
                    check_name=r[2],
                    dataset=r[3],
                    passed=bool(r[4]),
                    metric_value=r[5],
                    details=json.loads(r[6]) if r[6] else {},
                    checked_at=datetime.fromisoformat(r[7]),
                )
                for r in qual_rows
            ],
            rag_results=[
                RAGQualityResult(
                    metric_name=r[2],
                    score=r[3],
                    details=json.loads(r[4]) if r[4] else {},
                    measured_at=datetime.fromisoformat(r[5]),
                )
                for r in rag_rows
            ],
            alerts=[
                Alert(
                    severity=Severity(r[2]),
                    source_module=r[3],
                    title=r[4],
                    message=r[5],
                    context=json.loads(r[6]) if r[6] else {},
                    created_at=datetime.fromisoformat(r[7]),
                )
                for r in alert_rows
            ],
        )

    @staticmethod
    def _cutoff(days: int) -> str:
        from datetime import timedelta

        return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
