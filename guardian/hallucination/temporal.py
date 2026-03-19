"""Temporal staleness detection — finds outdated information in documents."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np

from guardian.models import Document, HallucinationRisk, Severity

DATE_PATTERNS = [
    # 2023-01-15, 2023/01/15
    re.compile(r"\b(20\d{2})[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])\b"),
    # January 15, 2023
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2}),?\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
    # 15 Jan 2023
    re.compile(
        r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
    # Just year: "in 2019", "since 2020"
    re.compile(r"\b(?:in|since|from|as of|until|before|after)\s+(20\d{2})\b", re.IGNORECASE),
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


class TemporalDetector:
    """Detects outdated information using date extraction.

    Flags documents where the most recent date mentioned is older
    than the configured staleness threshold.
    """

    def __init__(self, config: dict) -> None:
        self._staleness_days = config.get("temporal_staleness_days", 365)

    def detect(
        self, documents: list[Document], embeddings: np.ndarray
    ) -> list[HallucinationRisk]:
        now = datetime.now(timezone.utc)
        risks: list[HallucinationRisk] = []

        for doc in documents:
            dates = self._extract_dates(doc.content)
            if not dates:
                continue

            most_recent = max(dates)
            age_days = (now - most_recent).days

            if age_days > self._staleness_days:
                if age_days > self._staleness_days * 3:
                    severity = Severity.CRITICAL
                elif age_days > self._staleness_days * 2:
                    severity = Severity.WARNING
                else:
                    severity = Severity.INFO

                confidence = min(age_days / (self._staleness_days * 5), 1.0)
                risks.append(
                    HallucinationRisk(
                        risk_type="temporal",
                        severity=severity,
                        doc_ids=[doc.doc_id],
                        description=(
                            f"Document contains potentially outdated information. "
                            f"Most recent date found: {most_recent.strftime('%Y-%m-%d')} "
                            f"({age_days} days old, threshold: {self._staleness_days} days)."
                        ),
                        confidence=round(confidence, 3),
                    )
                )

        return risks

    @staticmethod
    def _extract_dates(text: str) -> list[datetime]:
        """Extract dates from text using regex patterns."""
        dates: list[datetime] = []
        # ISO-like dates: 2023-01-15
        for m in DATE_PATTERNS[0].finditer(text):
            try:
                dates.append(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc))
            except ValueError:
                pass
        # "January 15, 2023"
        for m in DATE_PATTERNS[1].finditer(text):
            month = MONTH_MAP.get(m.group(1).lower())
            if month:
                try:
                    dates.append(datetime(int(m.group(3)), month, int(m.group(2)), tzinfo=timezone.utc))
                except ValueError:
                    pass
        # "15 Jan 2023"
        for m in DATE_PATTERNS[2].finditer(text):
            month = MONTH_MAP.get(m.group(2).lower())
            if month:
                try:
                    dates.append(datetime(int(m.group(3)), month, int(m.group(1)), tzinfo=timezone.utc))
                except ValueError:
                    pass
        # Just year
        for m in DATE_PATTERNS[3].finditer(text):
            try:
                dates.append(datetime(int(m.group(1)), 7, 1, tzinfo=timezone.utc))
            except ValueError:
                pass
        return dates
