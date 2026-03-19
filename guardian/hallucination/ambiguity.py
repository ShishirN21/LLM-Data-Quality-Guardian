"""Entity ambiguity detection — same name/term referring to different things."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from guardian.embeddings.manager import EmbeddingManager
from guardian.models import Document, HallucinationRisk, Severity

# Simple named entity extraction: capitalized multi-word phrases
ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
# Also catch acronyms
ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,6})\b")


class AmbiguityDetector:
    """Detects entity ambiguity in the document corpus.

    When the same named entity appears in multiple documents but those
    documents have high embedding variance, the entity likely refers to
    different things — creating ambiguity that can trigger hallucinations.
    """

    def __init__(self, config: dict, embedding_manager: EmbeddingManager) -> None:
        self._threshold = config.get("entity_ambiguity_threshold", 3)
        self._embedding_manager = embedding_manager

    def detect(
        self, documents: list[Document], embeddings: np.ndarray
    ) -> list[HallucinationRisk]:
        # Map entities to document indices
        entity_docs: dict[str, list[int]] = defaultdict(list)
        for idx, doc in enumerate(documents):
            entities = self._extract_entities(doc.content)
            for entity in entities:
                entity_docs[entity].append(idx)

        risks: list[HallucinationRisk] = []
        for entity, doc_indices in entity_docs.items():
            if len(doc_indices) < self._threshold:
                continue

            # Compute embedding variance for documents mentioning this entity
            entity_embeddings = embeddings[doc_indices]
            centroid = entity_embeddings.mean(axis=0)
            distances = np.linalg.norm(entity_embeddings - centroid, axis=1)
            variance = float(np.var(distances))
            mean_dist = float(np.mean(distances))

            # High variance means the entity is used in very different contexts
            if mean_dist > 0.5:
                severity = Severity.CRITICAL if mean_dist > 0.8 else Severity.WARNING
                confidence = min(mean_dist / 1.5, 1.0)
                risks.append(
                    HallucinationRisk(
                        risk_type="ambiguity",
                        severity=severity,
                        doc_ids=[documents[i].doc_id for i in doc_indices],
                        description=(
                            f"Entity '{entity}' appears in {len(doc_indices)} documents "
                            f"with high contextual variance (mean_distance={mean_dist:.3f}, "
                            f"variance={variance:.3f}). May refer to different things."
                        ),
                        confidence=round(confidence, 3),
                    )
                )

        return risks

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Extract named entities using regex heuristics."""
        entities: set[str] = set()
        for m in ENTITY_PATTERN.finditer(text):
            entities.add(m.group(1))
        for m in ACRONYM_PATTERN.finditer(text):
            entities.add(m.group(1))
        return entities
