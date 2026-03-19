"""Main pipeline orchestrator — ties all Guardian modules together."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from guardian.alerts.base import AlertManager
from guardian.alerts.console import ConsoleAlertHandler
from guardian.alerts.webhook import WebhookAlertHandler
from guardian.drift.detector import DriftDetector
from guardian.embeddings.manager import EmbeddingManager
from guardian.hallucination.detector import HallucinationDetector
from guardian.ingestion import create_default_registry
from guardian.models import (
    Alert,
    Document,
    HallucinationRisk,
    PipelineRunResult,
    Severity,
)
from guardian.quality.checks import CustomQualityChecks
from guardian.quality.soda_runner import SodaQualityRunner
from guardian.rag.chunking import ChunkQualityAnalyzer
from guardian.rag.context import ContextCoverageAnalyzer
from guardian.rag.retrieval import RetrievalQualityMonitor
from guardian.storage.metrics_store import MetricsStore
from guardian.storage.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class GuardianPipeline:
    """Main orchestrator for the LLM Data Quality Guardian pipeline."""

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        self._config = self._load_config(config_path)
        self._base_dir = Path(config_path).resolve().parent.parent

        # Core components
        emb_cfg = self._config["embeddings"]
        self._embedding_manager = EmbeddingManager(
            model_name=emb_cfg["model_name"],
            cache_dir=emb_cfg.get("cache_dir"),
        )

        vs_cfg = self._config["vector_store"]
        self._vector_store = VectorStoreManager(
            persist_directory=vs_cfg["persist_directory"],
            collection_name=vs_cfg["collection_name"],
        )

        ms_cfg = self._config["metrics_store"]
        self._metrics_store = MetricsStore(db_path=ms_cfg["database_path"])

        # Detection modules
        self._drift_detector = DriftDetector(self._config["drift"])
        self._hallucination_detector = HallucinationDetector(
            self._config["hallucination"], self._embedding_manager
        )
        self._quality_runner = SodaQualityRunner(self._config["quality"])
        self._custom_checks = CustomQualityChecks()
        self._chunk_analyzer = ChunkQualityAnalyzer(self._config["rag"])
        self._rag_monitor = RetrievalQualityMonitor(
            self._config["rag"], self._embedding_manager, self._vector_store
        )
        self._context_analyzer = ContextCoverageAnalyzer(self._config["rag"])

        # Ingestion
        self._loader_registry = create_default_registry()

        # Alerts
        alert_cfg = self._config["alerts"]
        self._alert_manager = AlertManager(alert_cfg)
        if alert_cfg.get("console_enabled", True):
            self._alert_manager.register_handler(ConsoleAlertHandler())
        if alert_cfg.get("webhook_enabled") and alert_cfg.get("webhook_url"):
            self._alert_manager.register_handler(
                WebhookAlertHandler(alert_cfg["webhook_url"])
            )

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path) as f:
            return yaml.safe_load(f)

    def run(self) -> PipelineRunResult:
        """Execute a complete pipeline run."""
        run_id = uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        result = PipelineRunResult(run_id=run_id, started_at=now)

        logger.info("Pipeline run %s started", run_id)

        # ── Phase 1: Ingest ────────────────────────────────────────
        documents = self._ingest_documents()
        result.documents_processed = len(documents)
        logger.info("Ingested %d documents", len(documents))

        if not documents:
            logger.warning("No documents found — skipping analysis")
            result.completed_at = datetime.now(timezone.utc)
            self._metrics_store.save_run(result)
            return result

        # ── Phase 2: Embed ─────────────────────────────────────────
        logger.info("Generating embeddings...")
        embeddings = self._embedding_manager.embed_documents(documents)
        self._vector_store.upsert_documents(documents, embeddings)
        logger.info("Embeddings generated and stored (%d vectors)", len(embeddings))

        # ── Phase 3: Drift Detection ──────────────────────────────
        logger.info("Running drift detection...")
        drift_cfg = self._config["drift"]
        min_samples = drift_cfg.get("min_samples", 50)

        # Use all existing embeddings as baseline; compare against current batch
        baseline = self._vector_store.get_all_embeddings()

        if baseline is not None and len(baseline) >= min_samples:
            result.drift_results = self._drift_detector.detect(baseline, embeddings)
            for dr in result.drift_results:
                if dr.is_drifted:
                    result.alerts.append(
                        self._alert_manager.create_and_send(
                            Severity.WARNING,
                            "drift",
                            f"Drift detected: {dr.test_name}",
                            f"{dr.test_name} detected distribution shift "
                            f"(statistic={dr.statistic}, threshold={dr.threshold})",
                        )
                    )
        else:
            logger.info(
                "Insufficient baseline data for drift detection (need %d samples)", min_samples
            )

        # ── Phase 4: Hallucination Pattern Scan ────────────────────
        logger.info("Scanning for hallucination-triggering patterns...")
        result.hallucination_risks = self._hallucination_detector.scan(
            documents, embeddings
        )
        for risk in result.hallucination_risks:
            if risk.severity in (Severity.WARNING, Severity.CRITICAL):
                result.alerts.append(
                    self._alert_manager.create_and_send(
                        risk.severity,
                        "hallucination",
                        f"Hallucination risk: {risk.risk_type}",
                        risk.description,
                    )
                )

        # ── Phase 5: Traditional Quality Checks ───────────────────
        logger.info("Running data quality checks...")
        df = pd.DataFrame([{"doc_id": d.doc_id, "content": d.content} for d in documents])
        result.quality_results.extend(
            self._quality_runner.run_checks(df, "documents")
        )
        result.quality_results.extend(self._custom_checks.run_all(documents))

        for qc in result.quality_results:
            if not qc.passed:
                result.alerts.append(
                    self._alert_manager.create_and_send(
                        Severity.WARNING,
                        "quality",
                        f"Quality check failed: {qc.check_name}",
                        f"Check '{qc.check_name}' on dataset '{qc.dataset}' "
                        f"failed (value={qc.metric_value})",
                    )
                )

        # ── Phase 6: RAG Quality ───────────────────────────────────
        logger.info("Evaluating RAG pipeline quality...")
        result.rag_results.extend(
            self._chunk_analyzer.run_all(documents, embeddings)
        )

        # Run sample queries if available
        queries = self._load_sample_queries()
        if queries and self._vector_store.count() > 0:
            result.rag_results.extend(self._rag_monitor.batch_evaluate(queries))

        # ── Finalize ───────────────────────────────────────────────
        result.completed_at = datetime.now(timezone.utc)
        self._metrics_store.save_run(result)

        logger.info(
            "Pipeline run %s completed. Docs=%d, Drift=%d, Risks=%d, Quality=%d, RAG=%d, Alerts=%d",
            run_id,
            result.documents_processed,
            len(result.drift_results),
            len(result.hallucination_risks),
            len(result.quality_results),
            len(result.rag_results),
            len(result.alerts),
        )

        return result

    def _ingest_documents(self) -> list[Document]:
        """Load documents from all configured data directories."""
        documents: list[Document] = []
        pipeline_cfg = self._config.get("pipeline", {})
        for dir_name in pipeline_cfg.get("data_directories", []):
            dir_path = Path(dir_name)
            if not dir_path.exists():
                logger.warning("Data directory not found: %s", dir_path)
                continue
            docs = self._loader_registry.load_directory(dir_path)
            documents.extend(docs)
        return documents

    def _load_sample_queries(self) -> list[str]:
        """Load sample queries from data/sample/queries.txt if it exists."""
        queries_file = Path("data/sample/queries.txt")
        if not queries_file.exists():
            return []
        text = queries_file.read_text(encoding="utf-8")
        return [q.strip() for q in text.strip().splitlines() if q.strip()]
