"""Custom data quality checks beyond Soda Core."""

from __future__ import annotations

import statistics

from guardian.models import Document, QualityCheckResult


class CustomQualityChecks:
    """Additional quality checks for document corpora."""

    def check_document_lengths(
        self, documents: list[Document], min_length: int = 10, max_length: int = 50000
    ) -> QualityCheckResult:
        """Check that document content lengths are within acceptable range."""
        lengths = [len(d.content) for d in documents]
        too_short = sum(1 for l in lengths if l < min_length)
        too_long = sum(1 for l in lengths if l > max_length)
        violations = too_short + too_long
        passed = violations == 0

        return QualityCheckResult(
            check_name="document_length_range",
            dataset="documents",
            passed=passed,
            metric_value=float(violations),
            details={
                "too_short": too_short,
                "too_long": too_long,
                "min_length": min(lengths) if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "mean_length": round(statistics.mean(lengths), 1) if lengths else 0,
                "median_length": round(statistics.median(lengths), 1) if lengths else 0,
            },
        )

    def check_empty_content(self, documents: list[Document]) -> QualityCheckResult:
        """Check that no documents have empty or whitespace-only content."""
        empty = [d.doc_id for d in documents if not d.content.strip()]
        return QualityCheckResult(
            check_name="empty_content",
            dataset="documents",
            passed=len(empty) == 0,
            metric_value=float(len(empty)),
            details={"empty_doc_ids": empty[:20]},
        )

    def check_duplicate_content(self, documents: list[Document]) -> QualityCheckResult:
        """Check for exact duplicate document content."""
        seen: dict[str, str] = {}
        duplicates: list[tuple[str, str]] = []
        for doc in documents:
            normalized = doc.content.strip().lower()
            if normalized in seen:
                duplicates.append((doc.doc_id, seen[normalized]))
            else:
                seen[normalized] = doc.doc_id

        return QualityCheckResult(
            check_name="duplicate_content",
            dataset="documents",
            passed=len(duplicates) == 0,
            metric_value=float(len(duplicates)),
            details={"duplicate_pairs": duplicates[:20]},
        )

    def run_all(self, documents: list[Document]) -> list[QualityCheckResult]:
        """Run all custom quality checks."""
        return [
            self.check_document_lengths(documents),
            self.check_empty_content(documents),
            self.check_duplicate_content(documents),
        ]
