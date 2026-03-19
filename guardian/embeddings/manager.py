"""Embedding generation with sentence-transformers and disk caching."""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np

from guardian.embeddings.cache import EmbeddingCache
from guardian.models import Document

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Generates and caches document embeddings using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._cache = (
            EmbeddingCache(Path(cache_dir)) if cache_dir else None
        )

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(
        self, documents: list[Document], batch_size: int = 64
    ) -> np.ndarray:
        """Generate embeddings for documents, using cache where available."""
        results: list[np.ndarray] = []
        to_embed: list[tuple[int, str]] = []  # (index, content)

        for i, doc in enumerate(documents):
            cached = None
            if self._cache:
                cached = self._cache.get(doc.doc_id, self._model_name)
            if cached is not None:
                results.append((i, cached))
            else:
                to_embed.append((i, doc.content))

        if to_embed:
            indices, texts = zip(*to_embed)
            new_embeddings = self.model.encode(
                list(texts), batch_size=batch_size, show_progress_bar=False
            )
            for idx, emb in zip(indices, new_embeddings):
                emb = np.array(emb, dtype=np.float32)
                results.append((idx, emb))
                if self._cache:
                    self._cache.put(
                        documents[idx].doc_id, self._model_name, emb
                    )

        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query string."""
        return np.array(
            self.model.encode(query, show_progress_bar=False), dtype=np.float32
        )

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
        if denom == 0:
            return 0.0
        return float(np.dot(a_flat, b_flat) / denom)
