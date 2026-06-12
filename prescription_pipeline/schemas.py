from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BBoxPrediction:
    level: str
    index: int
    bbox: list[int]
    text: str
    block_index: int | None = None
    line_index: int | None = None
    word_index: int | None = None


@dataclass
class ExtractedMedication:
    drug_name: str
    dosage: str = ""
    frequency: str = ""
    duration: str = ""
    source_line: str = ""
    bbox: list[int] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class MatchedMedication:
    ocr: ExtractedMedication
    match_status: str
    canonical_name: str = ""
    canonical_strength: str = ""
    canonical_form: str = ""
    match_score: float = 0.0
    drug_id: str = ""
    database_facts: dict[str, str] = field(default_factory=dict)
    frequency_normalized: str = ""


@dataclass
class CounselorResult:
    patient_query: str
    patient_response: str
    reasoning: str = ""
    medication_drug_name: str = ""
    counseling_backend: str = "dspy-chain-of-thought"
    counselor_model: str = ""


@dataclass
class PrescriptionResult:
    image_name: str
    image_path: str
    width: int
    height: int
    full_text: str
    medications: list[MatchedMedication]
    ocr_predictions: list[BBoxPrediction]
    overlay_path: str = ""
    ner_backend: str = "regex"
    counseling: CounselorResult | None = None
    action_triggers: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
