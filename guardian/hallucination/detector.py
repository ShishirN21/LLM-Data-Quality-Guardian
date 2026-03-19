"""Hallucination pattern detection orchestrator."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from guardian.embeddings.manager import EmbeddingManager
from guardian.models import Document, HallucinationRisk

from guardian.hallucination.ambiguity import AmbiguityDetector
from guardian.hallucination.contradiction import ContradictionDetector
from guardian.hallucination.sparse import SparseContextDetector
from guardian.hallucination.temporal import TemporalDetector


class PatternDetector(ABC):
    """Abstract base for hallucination pattern detectors."""

    @abstractmethod
    def detect(
        self, documents: list[Document], embeddings: np.ndarray
    ) -> list[HallucinationRisk]:
        """Scan documents for hallucination-triggering patterns."""


class HallucinationDetector:
    """Orchestrates multiple hallucination pattern detectors."""

    def __init__(self, config: dict, embedding_manager: EmbeddingManager) -> None:
        self._detectors: list[PatternDetector] = [
            ContradictionDetector(config, embedding_manager),
            AmbiguityDetector(config, embedding_manager),
            TemporalDetector(config),
            SparseContextDetector(config, embedding_manager),
        ]

    def scan(
        self, documents: list[Document], embeddings: np.ndarray
    ) -> list[HallucinationRisk]:
        """Run all pattern detectors and return identified risks."""
        risks: list[HallucinationRisk] = []
        for detector in self._detectors:
            risks.extend(detector.detect(documents, embeddings))
        return risks
