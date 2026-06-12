from __future__ import annotations

from prescription_pipeline.schemas import MatchedMedication

FREQUENCY_TRIGGER_MAP: dict[str, dict[str, str]] = {
    "bd": {
        "trigger_type": "calendar_reminder",
        "schedule": "morning_evening",
        "reason": "Frequency BD detected (twice daily)",
    },
    "twice daily": {
        "trigger_type": "calendar_reminder",
        "schedule": "morning_evening",
        "reason": "Twice-daily frequency detected",
    },
    "tds": {
        "trigger_type": "calendar_reminder",
        "schedule": "morning_afternoon_evening",
        "reason": "Frequency TDS detected (three times daily)",
    },
    "three times daily": {
        "trigger_type": "calendar_reminder",
        "schedule": "morning_afternoon_evening",
        "reason": "Three-times-daily frequency detected",
    },
    "once daily": {
        "trigger_type": "calendar_reminder",
        "schedule": "daily",
        "reason": "Once-daily frequency detected",
    },
    "od": {
        "trigger_type": "calendar_reminder",
        "schedule": "daily",
        "reason": "Frequency OD detected (once daily)",
    },
}


def build_action_triggers(medications: list[MatchedMedication]) -> list[dict[str, str]]:
    triggers: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for med in medications:
        key = (med.ocr.drug_name.strip().lower(), med.frequency_normalized.strip().lower())
        if key in seen:
            continue

        freq = med.frequency_normalized.strip().lower()
        if not freq:
            freq = med.ocr.frequency.strip().lower()
        template = FREQUENCY_TRIGGER_MAP.get(freq)
        if not template:
            continue

        seen.add(key)
        triggers.append(
            {
                **template,
                "drug_name": med.canonical_name or med.ocr.drug_name,
                "frequency": med.frequency_normalized or med.ocr.frequency,
            }
        )

    return triggers
