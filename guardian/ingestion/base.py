"""Abstract base loader and loader registry."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from guardian.models import Document


class BaseLoader(ABC):
    """Abstract base for document loaders."""

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions this loader handles (e.g., ['.csv'])."""

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """Load documents from a single file."""

    def load_directory(self, dir_path: Path) -> list[Document]:
        """Recursively load all supported files from a directory."""
        documents: list[Document] = []
        exts = set(self.supported_extensions())
        for path in sorted(dir_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                documents.extend(self.load(path))
        return documents

    @staticmethod
    def generate_doc_id(source: str, index: int = 0) -> str:
        """Generate a deterministic document ID from source path and index."""
        raw = f"{source}::{index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class LoaderRegistry:
    """Registry mapping file extensions to loader implementations."""

    def __init__(self) -> None:
        self._loaders: dict[str, BaseLoader] = {}

    def register(self, loader: BaseLoader) -> None:
        for ext in loader.supported_extensions():
            self._loaders[ext.lower()] = loader

    def get_loader(self, extension: str) -> BaseLoader | None:
        return self._loaders.get(extension.lower())

    def load_file(self, file_path: Path) -> list[Document]:
        loader = self.get_loader(file_path.suffix)
        if loader is None:
            return []
        return loader.load(file_path)

    def load_directory(self, dir_path: Path) -> list[Document]:
        """Load all supported files from a directory."""
        documents: list[Document] = []
        for path in sorted(dir_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in self._loaders:
                documents.extend(self.load_file(path))
        return documents

    @property
    def supported_extensions(self) -> list[str]:
        return list(self._loaders.keys())


def create_default_registry() -> LoaderRegistry:
    """Create a registry with all built-in loaders."""
    from guardian.ingestion.csv_loader import CSVLoader
    from guardian.ingestion.json_loader import JSONLoader
    from guardian.ingestion.text_loader import TextLoader

    registry = LoaderRegistry()
    registry.register(CSVLoader())
    registry.register(JSONLoader())
    registry.register(TextLoader())
    return registry
