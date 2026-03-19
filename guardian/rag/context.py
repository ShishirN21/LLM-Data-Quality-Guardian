"""Context coverage and redundancy analysis for RAG pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from guardian.models import Document, RAGQualityResult


class ContextCoverageAnalyzer:
    """Monitors how well retrieved context covers query topics."""

    def __init__(self, config: dict) -> None:
        self._coverage_min = config.get("context_coverage_min", 0.6)

    def analyze_coverage(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
    ) -> RAGQualityResult:
        """Measure how well retrieved documents cover the query's semantic space."""
        if len(doc_embeddings) == 0:
            return RAGQualityResult(
                metric_name="context_coverage", score=0.0,
                details={"error": "no retrieved documents"},
            )

        # Query-to-document similarities
        q_emb = query_embedding.reshape(1, -1)
        sims = cosine_similarity(q_emb, doc_embeddings)[0]

        # Coverage = best similarity (how well the top doc covers the query)
        # Combined with diversity of coverage across retrieved docs
        best_sim = float(np.max(sims))
        mean_sim = float(np.mean(sims))

        # Weighted score: best coverage matters most, spread matters too
        score = 0.7 * best_sim + 0.3 * mean_sim

        return RAGQualityResult(
            metric_name="context_coverage",
            score=round(score, 4),
            details={
                "best_similarity": round(best_sim, 4),
                "mean_similarity": round(mean_sim, 4),
                "min_similarity": round(float(np.min(sims)), 4),
                "retrieved_count": len(doc_embeddings),
                "above_threshold": int(np.sum(sims >= self._coverage_min)),
            },
        )

    def detect_redundancy(
        self, doc_embeddings: np.ndarray
    ) -> RAGQualityResult:
        """Detect when retrieved documents are too similar (wasted context window)."""
        if len(doc_embeddings) < 2:
            return RAGQualityResult(
                metric_name="context_redundancy", score=0.0,
                details={"error": "need at least 2 documents"},
            )

        sim_matrix = cosine_similarity(doc_embeddings)
        # Get upper triangle (exclude diagonal)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        mean_sim = float(np.mean(upper))
        max_sim = float(np.max(upper))
        high_redundancy_pairs = int(np.sum(upper > 0.9))

        # Lower score = more redundancy (bad)
        diversity_score = 1.0 - mean_sim

        return RAGQualityResult(
            metric_name="context_redundancy",
            score=round(diversity_score, 4),
            details={
                "mean_inter_doc_similarity": round(mean_sim, 4),
                "max_inter_doc_similarity": round(max_sim, 4),
                "high_redundancy_pairs": high_redundancy_pairs,
                "total_pairs": len(upper),
                "diversity_score": round(diversity_score, 4),
            },
        )
