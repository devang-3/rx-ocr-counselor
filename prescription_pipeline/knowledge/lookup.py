from __future__ import annotations

import logging

from prescription_pipeline.ner.parser import normalize_frequency, should_match_knowledge_base
from prescription_pipeline.schemas import ExtractedMedication, MatchedMedication

logger = logging.getLogger(__name__)


class KnowledgeLookup:
    def __init__(self, drug_matcher) -> None:
        self.drug_matcher = drug_matcher

    def _build_query(self, medication: ExtractedMedication) -> str:
        parts = [medication.drug_name]
        if medication.dosage:
            parts.append(medication.dosage)
        return " ".join(parts).strip()

    def match_medication(self, medication: ExtractedMedication) -> MatchedMedication:
        frequency_normalized = normalize_frequency(medication.frequency) if medication.frequency else ""

        if self.drug_matcher is None:
            return MatchedMedication(
                ocr=medication,
                match_status="disabled",
                frequency_normalized=frequency_normalized,
            )

        if not should_match_knowledge_base(medication):
            logger.debug("Skipping KB match for low-quality extraction: %r", medication.drug_name)
            return MatchedMedication(
                ocr=medication,
                match_status="skipped",
                frequency_normalized=frequency_normalized,
            )

        query = self._build_query(medication)
        result = self.drug_matcher.match(query, top_k=3)
        best = result.get("best_match") or {}

        return MatchedMedication(
            ocr=medication,
            match_status=result.get("status", "no_match"),
            canonical_name=best.get("generic_name", ""),
            canonical_strength=best.get("strength", ""),
            canonical_form=best.get("form", ""),
            match_score=float(best.get("score", 0.0)),
            drug_id=best.get("drug_id", ""),
            database_facts=result.get("database_facts", {}),
            frequency_normalized=frequency_normalized,
        )

    def match_all(self, medications: list[ExtractedMedication]) -> list[MatchedMedication]:
        return [self.match_medication(med) for med in medications]
