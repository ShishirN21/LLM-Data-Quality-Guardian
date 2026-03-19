"""Contradiction detection — finds semantically similar documents with conflicting content."""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from guardian.embeddings.manager import EmbeddingManager
from guardian.models import Document, HallucinationRisk, Severity

NEGATION_PATTERNS = [
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bno longer\b", re.IGNORECASE),
    re.compile(r"\bdoes(?:n't| not)\b", re.IGNORECASE),
    re.compile(r"\bis(?:n't| not)\b", re.IGNORECASE),
    re.compile(r"\bwas(?:n't| not)\b", re.IGNORECASE),
    re.compile(r"\bwere(?:n't| not)\b", re.IGNORECASE),
    re.compile(r"\bcan(?:'t|not)\b", re.IGNORECASE),
    re.compile(r"\bwon't\b", re.IGNORECASE),
    re.compile(r"\bshouldn't\b", re.IGNORECASE),
    re.compile(r"\bfalse\b", re.IGNORECASE),
    re.compile(r"\bincorrect\b", re.IGNORECASE),
    re.compile(r"\bdiscontinued\b", re.IGNORECASE),
    re.compile(r"\bdeprecated\b", re.IGNORECASE),
]

NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand)?\b")


class ContradictionDetector:
    """Detects contradictory facts in the document corpus.

    Algorithm:
    1. Find semantically similar document pairs (cosine sim > threshold).
    2. Among those, check for negation signal divergence — one doc has
       negation patterns where the other doesn't for the same topic.
    3. Check for conflicting numerical values in similar contexts.
    """

    def __init__(self, config: dict, embedding_manager: EmbeddingManager) -> None:
        self._sim_threshold = config.get("contradiction_sim_threshold", 0.8)
        self._embedding_manager = embedding_manager

    def detect(
        self, documents: list[Document], embeddings: np.ndarray
    ) -> list[HallucinationRisk]:
        if len(documents) < 2:
            return []

        sim_matrix = cosine_similarity(embeddings)
        risks: list[HallucinationRisk] = []
        seen_pairs: set[tuple[str, str]] = set()

        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                if sim_matrix[i][j] < self._sim_threshold:
                    continue

                pair_key = (documents[i].doc_id, documents[j].doc_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                neg_i = self._count_negations(documents[i].content)
                neg_j = self._count_negations(documents[j].content)

                # One has negations, the other doesn't → potential contradiction
                has_negation_divergence = (neg_i > 0) != (neg_j > 0)

                # Check for conflicting numbers in similar contexts
                nums_i = self._extract_numbers(documents[i].content)
                nums_j = self._extract_numbers(documents[j].content)
                has_number_conflict = bool(nums_i and nums_j and nums_i != nums_j)

                if has_negation_divergence or has_number_conflict:
                    confidence = float(sim_matrix[i][j]) * 0.8
                    if has_negation_divergence and has_number_conflict:
                        confidence = min(confidence + 0.15, 1.0)
                        severity = Severity.CRITICAL
                    elif has_negation_divergence:
                        severity = Severity.WARNING
                    else:
                        severity = Severity.WARNING

                    reasons = []
                    if has_negation_divergence:
                        reasons.append("negation divergence")
                    if has_number_conflict:
                        reasons.append(f"conflicting numbers: {nums_i} vs {nums_j}")

                    risks.append(
                        HallucinationRisk(
                            risk_type="contradiction",
                            severity=severity,
                            doc_ids=[documents[i].doc_id, documents[j].doc_id],
                            description=(
                                f"Potential contradiction between semantically similar documents "
                                f"(similarity={sim_matrix[i][j]:.3f}). Signals: {', '.join(reasons)}."
                            ),
                            confidence=round(confidence, 3),
                        )
                    )

        return risks

    @staticmethod
    def _count_negations(text: str) -> int:
        return sum(1 for p in NEGATION_PATTERNS if p.search(text))

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        return {m.group(0).strip() for m in NUMBER_PATTERN.finditer(text)}
