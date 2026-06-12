from __future__ import annotations

from prescription_pipeline.schemas import (
    BBoxPrediction,
    ExtractedMedication,
    MatchedMedication,
    PrescriptionResult,
)


def prescription_from_dict(data: dict) -> PrescriptionResult:
    medications: list[MatchedMedication] = []
    for item in data.get("medications", []):
        ocr_data = item.get("ocr", {})
        ocr = ExtractedMedication(
            drug_name=ocr_data.get("drug_name", ""),
            dosage=ocr_data.get("dosage", ""),
            frequency=ocr_data.get("frequency", ""),
            duration=ocr_data.get("duration", ""),
            source_line=ocr_data.get("source_line", ""),
            bbox=ocr_data.get("bbox", []),
            confidence=float(ocr_data.get("confidence", 0.0)),
        )
        medications.append(
            MatchedMedication(
                ocr=ocr,
                match_status=item.get("match_status", ""),
                canonical_name=item.get("canonical_name", ""),
                canonical_strength=item.get("canonical_strength", ""),
                canonical_form=item.get("canonical_form", ""),
                match_score=float(item.get("match_score", 0.0)),
                drug_id=item.get("drug_id", ""),
                database_facts=item.get("database_facts", {}),
                frequency_normalized=item.get("frequency_normalized", ""),
            )
        )

    predictions: list[BBoxPrediction] = []
    for item in data.get("ocr_predictions", []):
        predictions.append(
            BBoxPrediction(
                level=item.get("level", ""),
                index=int(item.get("index", 0)),
                bbox=item.get("bbox", []),
                text=item.get("text", ""),
                block_index=item.get("block_index"),
                line_index=item.get("line_index"),
                word_index=item.get("word_index"),
            )
        )

    return PrescriptionResult(
        image_name=data.get("image_name", ""),
        image_path=data.get("image_path", ""),
        width=int(data.get("width", 0)),
        height=int(data.get("height", 0)),
        full_text=data.get("full_text", ""),
        medications=medications,
        ocr_predictions=predictions,
        overlay_path=data.get("overlay_path", ""),
        ner_backend=data.get("ner_backend", "regex"),
        action_triggers=data.get("action_triggers", []),
    )
