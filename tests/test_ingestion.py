"""Tests for the ingestion module."""

from __future__ import annotations

from pathlib import Path

import pytest

from guardian.ingestion.csv_loader import CSVLoader
from guardian.ingestion.json_loader import JSONLoader
from guardian.ingestion.text_loader import TextLoader
from guardian.ingestion.base import LoaderRegistry, create_default_registry


class TestCSVLoader:
    def test_load_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,content,category\n1,Hello world,greeting\n2,Goodbye world,farewell\n")

        loader = CSVLoader()
        docs = loader.load(csv_file)

        assert len(docs) == 2
        assert "Hello world" in docs[0].content
        assert docs[0].source_path == str(csv_file)

    def test_supported_extensions(self) -> None:
        loader = CSVLoader()
        assert loader.supported_extensions() == [".csv"]

    def test_content_column_detection(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("id,text,meta\n1,This is a long text document with lots of words,short\n")

        loader = CSVLoader()
        docs = loader.load(csv_file)
        assert "long text document" in docs[0].content


class TestJSONLoader:
    def test_load_json_array(self, tmp_path: Path) -> None:
        json_file = tmp_path / "test.json"
        json_file.write_text('[{"content": "doc one"}, {"content": "doc two"}]')

        loader = JSONLoader()
        docs = loader.load(json_file)

        assert len(docs) == 2
        assert docs[0].content == "doc one"

    def test_load_jsonl(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"text": "line one"}\n{"text": "line two"}\n')

        loader = JSONLoader()
        docs = loader.load(jsonl_file)

        assert len(docs) == 2
        assert docs[0].content == "line one"

    def test_supported_extensions(self) -> None:
        loader = JSONLoader()
        assert ".json" in loader.supported_extensions()
        assert ".jsonl" in loader.supported_extensions()


class TestTextLoader:
    def test_load_text(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")

        loader = TextLoader()
        docs = loader.load(txt_file)

        assert len(docs) == 3
        assert docs[0].content == "First paragraph."
        assert docs[2].content == "Third paragraph."

    def test_empty_file(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("")

        loader = TextLoader()
        docs = loader.load(txt_file)
        assert len(docs) == 0


class TestLoaderRegistry:
    def test_default_registry(self) -> None:
        registry = create_default_registry()
        assert ".csv" in registry.supported_extensions
        assert ".json" in registry.supported_extensions
        assert ".txt" in registry.supported_extensions

    def test_load_file(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("content\nHello\nWorld\n")

        registry = create_default_registry()
        docs = registry.load_file(csv_file)
        assert len(docs) == 2

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')")

        registry = create_default_registry()
        docs = registry.load_file(py_file)
        assert len(docs) == 0
