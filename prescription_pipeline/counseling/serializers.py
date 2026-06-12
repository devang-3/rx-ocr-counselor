from __future__ import annotations

import json
from typing import Any

from prescription_pipeline.schemas import MatchedMedication, PrescriptionResult


def medication_to_ocr_variables(medication: MatchedMedication) -> str:
    payload = {
        "drug_name": medication.ocr.drug_name,
        "dosage": medication.ocr.dosage,
        "frequency": medication.ocr.frequency,
        "frequency_normalized": medication.frequency_normalized,
        "duration": medication.ocr.duration,
        "source_line": medication.ocr.source_line,
        "match_status": medication.match_status,
        "canonical_name": medication.canonical_name,
        "canonical_strength": medication.canonical_strength,
        "canonical_form": medication.canonical_form,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def medication_to_database_facts(medication: MatchedMedication) -> str:
    if medication.database_facts:
        return json.dumps(medication.database_facts, ensure_ascii=False, indent=2)
    return json.dumps(
        {
            "note": "No knowledge-base match available for this extraction.",
            "ocr_drug_name": medication.ocr.drug_name,
        },
        ensure_ascii=False,
        indent=2,
    )


def prescription_summary(result: PrescriptionResult) -> str:
    meds: list[dict[str, Any]] = []
    for med in result.medications:
        meds.append(
            {
                "ocr": {
                    "drug_name": med.ocr.drug_name,
                    "dosage": med.ocr.dosage,
                    "frequency": med.ocr.frequency,
                    "duration": med.ocr.duration,
                },
                "canonical_name": med.canonical_name,
                "match_status": med.match_status,
            }
        )
    return json.dumps({"medications": meds}, ensure_ascii=False, indent=2)
