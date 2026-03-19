"""RAG retrieval quality monitoring."""

from __future__ import annotations

import numpy as np

from guardian.embeddings.manager import EmbeddingManager
from guardian.models import RAGQualityResult
from guardian.storage.vectorstore import VectorStoreManager


class RetrievalQualityMonitor:
    """Monitors the quality of RAG retrieval results."""

    def __init__(
        self,
        config: dict,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStoreManager,
    ) -> None:
        self._relevance_threshold = config.get("relevance_threshold", 0.7)
        self._top_k = config.get("top_k", 5)
        self._embedding_manager = embedding_manager
        self._vector_store = vector_store

    def score_retrieval(self, query: str, top_k: int | None = None) -> RAGQualityResult:
        """Score how relevant retrieved documents are to a query."""
        k = top_k or self._top_k
        query_embedding = self._embedding_manager.embed_query(query)
        results = self._vector_store.query(query_embedding, top_k=k)

        if not results:
            return RAGQualityResult(
                metric_name="retrieval_relevance",
                score=0.0,
                details={"query": query, "retrieved_count": 0},
            )

        # ChromaDB returns cosine distances; convert to similarity
        similarities = [1.0 - dist for _, dist in results]
        mean_sim = float(np.mean(similarities))
        above_threshold = sum(1 for s in similarities if s >= self._relevance_threshold)

        return RAGQualityResult(
            metric_name="retrieval_relevance",
            score=round(mean_sim, 4),
            details={
                "query": query,
                "retrieved_count": len(results),
                "mean_similarity": round(mean_sim, 4),
                "above_threshold": above_threshold,
                "threshold": self._relevance_threshold,
                "per_document": [
                    {"doc_id": doc.doc_id, "similarity": round(1.0 - dist, 4)}
                    for doc, dist in results
                ],
            },
        )

    def batch_evaluate(self, queries: list[str]) -> list[RAGQualityResult]:
        """Evaluate retrieval quality across multiple queries."""
        results = [self.score_retrieval(q) for q in queries]

        if results:
            scores = [r.score for r in results]
            summary = RAGQualityResult(
                metric_name="retrieval_relevance_summary",
                score=round(float(np.mean(scores)), 4),
                details={
                    "total_queries": len(queries),
                    "mean_score": round(float(np.mean(scores)), 4),
                    "min_score": round(float(np.min(scores)), 4),
                    "max_score": round(float(np.max(scores)), 4),
                    "below_threshold": sum(
                        1 for s in scores if s < self._relevance_threshold
                    ),
                },
            )
            results.append(summary)

        return results
