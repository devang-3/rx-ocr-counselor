from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

from prescription_pipeline.ner.labels import ID2LABEL
from prescription_pipeline.ner.parser import PrescriptionParser
from prescription_pipeline.schemas import BBoxPrediction, ExtractedMedication

logger = logging.getLogger(__name__)


def pixel_boxes_to_layoutlm(boxes: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Convert pixel xyxy boxes to LayoutLM 0-1000 normalized coordinates."""
    normalized: list[list[int]] = []
    for x0, y0, x1, y1 in boxes:
        normalized.append(
            [
                max(0, min(1000, int(1000 * x0 / width))),
                max(0, min(1000, int(1000 * y0 / height))),
                max(0, min(1000, int(1000 * x1 / width))),
                max(0, min(1000, int(1000 * y1 / height))),
            ]
        )
    return normalized


def _union_bbox(boxes: list[list[int]]) -> list[int]:
    if not boxes:
        return []
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _group_bio_spans(
    words: list[str],
    labels: list[str],
    boxes: list[list[int]],
) -> dict[str, list[dict]]:
    """Group consecutive BIO tokens into spans per entity type."""
    spans: dict[str, list[dict]] = {
        "DRUG": [],
        "DOSAGE": [],
        "FREQUENCY": [],
        "DURATION": [],
        "FORM": [],
    }
    current_type: str | None = None
    current_words: list[str] = []
    current_boxes: list[list[int]] = []

    def flush() -> None:
        nonlocal current_type, current_words, current_boxes
        if current_type and current_words:
            spans[current_type].append(
                {
                    "text": " ".join(current_words).strip(),
                    "bbox": _union_bbox(current_boxes),
                }
            )
        current_type = None
        current_words = []
        current_boxes = []

    for word, label, box in zip(words, labels, boxes):
        if label == "O" or not label:
            flush()
            continue

        prefix, _, entity = label.partition("-")
        if prefix not in {"B", "I"} or entity not in spans:
            flush()
            continue

        if prefix == "B" or entity != current_type:
            flush()
            current_type = entity

        current_words.append(word)
        current_boxes.append(box)

    flush()
    return spans


def _pair_medications(spans: dict[str, list[dict]]) -> list[ExtractedMedication]:
    """Pair DRUG spans with nearest DOSAGE/FREQUENCY/DURATION on similar y-axis."""
    drugs = spans.get("DRUG", [])
    if not drugs:
        return []

    def y_center(box: list[int]) -> float:
        return (box[1] + box[3]) / 2 if box else 0.0

    def nearest(span_list: list[dict], drug_box: list[int]) -> str:
        if not span_list or not drug_box:
            return ""
        drug_y = y_center(drug_box)
        best = min(span_list, key=lambda s: abs(y_center(s["bbox"]) - drug_y))
        if abs(y_center(best["bbox"]) - drug_y) > 80:
            return ""
        return best["text"]

    medications: list[ExtractedMedication] = []
    for drug in drugs:
        box = drug["bbox"]
        dosage = nearest(spans.get("DOSAGE", []), box)
        frequency = nearest(spans.get("FREQUENCY", []), box)
        duration = nearest(spans.get("DURATION", []), box)

        confidence = 0.7
        if dosage:
            confidence += 0.1
        if frequency:
            confidence += 0.1
        if duration:
            confidence += 0.05

        medications.append(
            ExtractedMedication(
                drug_name=drug["text"],
                dosage=dosage,
                frequency=frequency,
                duration=duration,
                source_line=drug["text"],
                bbox=box,
                confidence=min(confidence, 1.0),
            )
        )

    return medications


class LayoutLMv3NerEngine:
    """
    Spatial NER using fine-tuned LayoutLMv3ForTokenClassification.

    Requires:
      - pipeline_models/layoutlmv3-base/          (processor)
      - pipeline_models/layoutlmv3-rx-ner/        (fine-tuned weights, Phase 3)

    Falls back to regex PrescriptionParser when NER weights are missing.
    """

    def __init__(
        self,
        processor: LayoutLMv3Processor,
        model: LayoutLMv3ForTokenClassification | None,
        device: torch.device,
        id2label: dict[int, str] | None = None,
    ) -> None:
        self.processor = processor
        self.model = model
        self.device = device
        self.id2label = id2label or ID2LABEL
        self.fallback = PrescriptionParser()
        self.is_ready = model is not None

    @property
    def backend_name(self) -> str:
        return "layoutlmv3" if self.is_ready else "regex"

    def parse_predictions(
        self,
        image_path: Path,
        predictions: list[BBoxPrediction],
        width: int,
        height: int,
    ) -> list[ExtractedMedication]:
        if not self.is_ready:
            logger.info("LayoutLMv3 NER weights not found — using regex fallback (see problem.txt P-PIPE-001)")
            return self.fallback.parse_predictions(predictions)

        words: list[str] = []
        boxes: list[list[int]] = []
        for pred in predictions:
            text = pred.text.strip()
            if not text:
                continue
            words.append(text)
            boxes.append(pred.bbox)

        if not words:
            return []

        image = Image.open(image_path).convert("RGB")
        norm_boxes = pixel_boxes_to_layoutlm(boxes, width, height)

        encoding = self.processor(
            image,
            words,
            boxes=norm_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encoding)

        logits = outputs.logits[0]
        pred_ids = logits.argmax(dim=-1).tolist()

        word_labels = ["O"] * len(words)
        if hasattr(encoding, "word_ids"):
            for token_idx, word_id in enumerate(encoding.word_ids(0)):
                if word_id is None or word_id >= len(words):
                    continue
                label = self.id2label.get(int(pred_ids[token_idx]), "O")
                if label.startswith("B-") or word_labels[word_id] == "O":
                    word_labels[word_id] = label
        else:
            for i in range(min(len(words), len(pred_ids))):
                word_labels[i] = self.id2label.get(int(pred_ids[i]), "O")

        spans = _group_bio_spans(words, word_labels, boxes)
        meds = _pair_medications(spans)
        if meds:
            return meds

        logger.warning("LayoutLMv3 returned no medications — falling back to regex")
        return self.fallback.parse_predictions(predictions)
