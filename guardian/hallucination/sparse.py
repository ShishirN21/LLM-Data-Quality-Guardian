"""Sparse context detection — finds isolated documents in embedding space."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from guardian.embeddings.manager import EmbeddingManager
from guardian.models import Document, HallucinationRisk, Severity


class SparseContextDetector:
    """Detects low-density knowledge regions in the embedding space.

    Documents with few neighbors in embedding space represent isolated
    knowledge — areas where the LLM has insufficient context and is
    more likely to hallucinate.
    """

    def __init__(self, config: dict, embedding_manager: EmbeddingManager) -> None:
        self._min_neighbors = config.get("sparse_context_min_neighbors", 5)
        self._radius = config.get("sparse_context_radius", 0.3)
        self._embedding_manager = embedding_manager

    def detect(
        self, documents: list[Document], embeddings: np.ndarray
    ) -> list[HallucinationRisk]:
        if len(documents) < self._min_neighbors + 1:
            return []

        nn = NearestNeighbors(
            radius=self._radius, metric="cosine", algorithm="brute"
        )
        nn.fit(embeddings)
        distances, indices = nn.radius_neighbors(embeddings)

        risks: list[HallucinationRisk] = []
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            # Exclude self from neighbor count
            neighbor_count = len(idxs) - 1
            if neighbor_count < self._min_neighbors:
                if neighbor_count == 0:
                    severity = Severity.CRITICAL
                    confidence = 0.95
                elif neighbor_count < self._min_neighbors // 2:
                    severity = Severity.WARNING
                    confidence = 0.7
                else:
                    severity = Severity.INFO
                    confidence = 0.5

                mean_dist = float(np.mean(dists[dists > 0])) if len(dists[dists > 0]) > 0 else self._radius

                risks.append(
                    HallucinationRisk(
                        risk_type="sparse_context",
                        severity=severity,
                        doc_ids=[documents[i].doc_id],
                        description=(
                            f"Document is in a sparse region of the knowledge space. "
                            f"Only {neighbor_count} neighbors within radius {self._radius} "
                            f"(minimum: {self._min_neighbors}). Mean neighbor distance: {mean_dist:.3f}. "
                            f"LLM may lack sufficient context for accurate responses."
                        ),
                        confidence=round(confidence, 3),
                    )
                )

        return risks
