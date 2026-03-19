"""Plain text and Markdown document loader."""

from __future__ import annotations

from pathlib import Path

from guardian.ingestion.base import BaseLoader
from guardian.models import Document


class TextLoader(BaseLoader):
    """Load documents from plain text and Markdown files."""

    def __init__(self, chunk_separator: str = "\n\n") -> None:
        self._separator = chunk_separator

    def supported_extensions(self) -> list[str]:
        return [".txt", ".md"]

    def load(self, file_path: Path) -> list[Document]:
        text = file_path.read_text(encoding="utf-8")
        chunks = [c.strip() for c in text.split(self._separator) if c.strip()]
        if not chunks:
            return []

        documents: list[Document] = []
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    doc_id=self.generate_doc_id(str(file_path), idx),
                    content=chunk,
                    source_path=str(file_path),
                    metadata={"chunk_index": idx, "total_chunks": len(chunks)},
                )
            )
        return documents
