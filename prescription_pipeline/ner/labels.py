"""BIO label schema for LayoutLMv3 prescription NER (Phase 3)."""

from __future__ import annotations

NER_LABELS: list[str] = [
    "O",
    "B-DRUG",
    "I-DRUG",
    "B-DOSAGE",
    "I-DOSAGE",
    "B-FREQUENCY",
    "I-FREQUENCY",
    "B-DURATION",
    "I-DURATION",
    "B-FORM",
    "I-FORM",
]

LABEL2ID: dict[str, int] = {label: idx for idx, label in enumerate(NER_LABELS)}
ID2LABEL: dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}
