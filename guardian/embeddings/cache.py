"""Disk-based embedding cache using numpy .npy files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class EmbeddingCache:
    """Disk-based cache for computed embeddings, keyed by (doc_id, model_name)."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, doc_id: str, model_name: str) -> Path:
        raw = f"{doc_id}::{model_name}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:24]
        return self._cache_dir / f"{h}.npy"

    def get(self, doc_id: str, model_name: str) -> np.ndarray | None:
        path = self._key_path(doc_id, model_name)
        if path.exists():
            return np.load(path)
        return None

    def put(self, doc_id: str, model_name: str, embedding: np.ndarray) -> None:
        path = self._key_path(doc_id, model_name)
        np.save(path, embedding)

    def invalidate(self, doc_id: str, model_name: str) -> None:
        path = self._key_path(doc_id, model_name)
        path.unlink(missing_ok=True)

    def clear(self) -> None:
        for f in self._cache_dir.glob("*.npy"):
            f.unlink()
