"""JSON and JSON-Lines document loader."""

from __future__ import annotations

import json
from pathlib import Path

from guardian.ingestion.base import BaseLoader
from guardian.models import Document


class JSONLoader(BaseLoader):
    """Load documents from JSON or JSON-Lines files."""

    def __init__(self, content_field: str | None = None) -> None:
        self._content_field = content_field

    def supported_extensions(self) -> list[str]:
        return [".json", ".jsonl"]

    def load(self, file_path: Path) -> list[Document]:
        text = file_path.read_text(encoding="utf-8")
        records = self._parse(text, file_path.suffix)
        documents: list[Document] = []
        for idx, record in enumerate(records):
            if isinstance(record, str):
                content = record
                metadata: dict = {}
            elif isinstance(record, dict):
                content = self._extract_content(record)
                metadata = {
                    k: v for k, v in record.items()
                    if k != self._content_field and isinstance(v, (str, int, float, bool))
                }
            else:
                content = json.dumps(record)
                metadata = {}

            documents.append(
                Document(
                    doc_id=self.generate_doc_id(str(file_path), idx),
                    content=content,
                    source_path=str(file_path),
                    metadata=metadata,
                )
            )
        return documents

    def _parse(self, text: str, suffix: str) -> list:
        if suffix == ".jsonl":
            return [json.loads(line) for line in text.strip().splitlines() if line.strip()]
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]

    def _extract_content(self, record: dict) -> str:
        if self._content_field and self._content_field in record:
            return str(record[self._content_field])
        for key in ("content", "text", "body", "description", "document"):
            if key in record:
                return str(record[key])
        return json.dumps(record)
