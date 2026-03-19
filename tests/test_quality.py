"""Tests for the quality checks module."""

from __future__ import annotations

import pytest

from guardian.models import Document
from guardian.quality.checks import CustomQualityChecks


class TestCustomQualityChecks:
    def setup_method(self) -> None:
        self.checker = CustomQualityChecks()

    def test_document_lengths_pass(self) -> None:
        docs = [
            Document(doc_id=f"d{i}", content=f"This is document {i} with enough content to pass.")
            for i in range(5)
        ]
        result = self.checker.check_document_lengths(docs)
        assert result.passed
        assert result.metric_value == 0

    def test_document_lengths_too_short(self) -> None:
        docs = [Document(doc_id="d1", content="Hi")]
        result = self.checker.check_document_lengths(docs, min_length=10)
        assert not result.passed
        assert result.details["too_short"] == 1

    def test_empty_content_detected(self) -> None:
        docs = [
            Document(doc_id="d1", content="Valid content"),
            Document(doc_id="d2", content=""),
            Document(doc_id="d3", content="   "),
        ]
        result = self.checker.check_empty_content(docs)
        assert not result.passed
        assert result.metric_value == 2

    def test_no_empty_content(self) -> None:
        docs = [Document(doc_id="d1", content="Valid content")]
        result = self.checker.check_empty_content(docs)
        assert result.passed

    def test_duplicate_content_detected(self) -> None:
        docs = [
            Document(doc_id="d1", content="Same content here"),
            Document(doc_id="d2", content="Same content here"),
            Document(doc_id="d3", content="Different content"),
        ]
        result = self.checker.check_duplicate_content(docs)
        assert not result.passed
        assert result.metric_value == 1

    def test_no_duplicates(self) -> None:
        docs = [
            Document(doc_id="d1", content="Content A"),
            Document(doc_id="d2", content="Content B"),
        ]
        result = self.checker.check_duplicate_content(docs)
        assert result.passed

    def test_run_all(self) -> None:
        docs = [
            Document(doc_id=f"d{i}", content=f"Document number {i} with valid content.")
            for i in range(5)
        ]
        results = self.checker.run_all(docs)
        assert len(results) == 3
