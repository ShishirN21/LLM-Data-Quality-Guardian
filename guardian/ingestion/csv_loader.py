"""CSV document loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from guardian.ingestion.base import BaseLoader
from guardian.models import Document


class CSVLoader(BaseLoader):
    """Load documents from CSV files."""

    def __init__(self, content_column: str | None = None) -> None:
        self._content_column = content_column

    def supported_extensions(self) -> list[str]:
        return [".csv"]

    def load(self, file_path: Path) -> list[Document]:
        df = pd.read_csv(file_path, dtype=str).fillna("")
        content_col = self._content_column or self._detect_content_column(df)
        documents: list[Document] = []
        for idx, row in df.iterrows():
            content = row[content_col] if content_col else " | ".join(row.astype(str))
            metadata = {
                col: row[col] for col in df.columns if col != content_col
            }
            documents.append(
                Document(
                    doc_id=self.generate_doc_id(str(file_path), int(idx)),
                    content=content,
                    source_path=str(file_path),
                    metadata=metadata,
                )
            )
        return documents

    @staticmethod
    def _detect_content_column(df: pd.DataFrame) -> str | None:
        """Heuristic: pick the column with the longest average string length."""
        candidates = ["content", "text", "body", "description", "document"]
        for name in candidates:
            if name in df.columns:
                return name
        str_cols = df.select_dtypes(include=["object"]).columns
        if len(str_cols) == 0:
            return None
        avg_lengths = {col: df[col].str.len().mean() for col in str_cols}
        return max(avg_lengths, key=avg_lengths.get)  # type: ignore[arg-type]
