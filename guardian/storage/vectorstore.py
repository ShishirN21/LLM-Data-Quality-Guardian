"""ChromaDB wrapper for vector storage and retrieval."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np

from guardian.models import Document


class VectorStoreManager:
    """Manages ChromaDB collections for document embeddings."""

    def __init__(
        self,
        persist_directory: str = ".data/chromadb",
        collection_name: str = "documents",
    ) -> None:
        path = Path(persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_documents(
        self, documents: list[Document], embeddings: np.ndarray
    ) -> None:
        """Add or update documents and their embeddings."""
        self._collection.upsert(
            ids=[d.doc_id for d in documents],
            embeddings=embeddings.tolist(),
            documents=[d.content for d in documents],
            metadatas=[
                {
                    "source_path": d.source_path,
                    "ingested_at": d.ingested_at.isoformat(),
                    **{k: str(v) for k, v in d.metadata.items()},
                }
                for d in documents
            ],
        )

    def query(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Query for similar documents, returning (document, distance) pairs."""
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        pairs: list[tuple[Document, float]] = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc = Document(
                doc_id=doc_id,
                content=results["documents"][0][i] if results["documents"] else "",
                source_path=meta.pop("source_path", ""),
                metadata=meta,
            )
            distance = results["distances"][0][i] if results["distances"] else 0.0
            pairs.append((doc, distance))
        return pairs

    def get_all_embeddings(self) -> np.ndarray | None:
        """Retrieve all embeddings for drift analysis."""
        count = self._collection.count()
        if count == 0:
            return None
        result = self._collection.get(include=["embeddings"])
        if result["embeddings"] is None or len(result["embeddings"]) == 0:
            return None
        return np.array(result["embeddings"], dtype=np.float32)

    def get_embeddings_by_time_window(
        self, start: datetime, end: datetime
    ) -> np.ndarray | None:
        """Get embeddings ingested within a time window."""
        result = self._collection.get(
            where={
                "$and": [
                    {"ingested_at": {"$gte": start.isoformat()}},
                    {"ingested_at": {"$lte": end.isoformat()}},
                ]
            },
            include=["embeddings"],
        )
        if result["embeddings"] is None or len(result["embeddings"]) == 0:
            return None
        return np.array(result["embeddings"], dtype=np.float32)

    def count(self) -> int:
        return self._collection.count()
