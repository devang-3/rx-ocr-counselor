from __future__ import annotations

import logging
from pathlib import Path

import cv2
import torch
from doctr.io import DocumentFile
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.models.loader import LoadedModels
from prescription_pipeline.ocr.bboxes import BBoxLevel, extract_bboxes
from prescription_pipeline.schemas import BBoxPrediction

logger = logging.getLogger(__name__)


def run_trocr_batch(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    crops: list[Image.Image],
    batch_size: int,
    device: torch.device,
) -> list[str]:
    if not crops:
        return []

    texts: list[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            pixel_values = processor(batch, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            texts.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
    return texts


class OcrEngine:
    def __init__(self, config: PipelineConfig, models: LoadedModels) -> None:
        self.config = config
        self.models = models

    def recognize(self, image_path: Path) -> tuple[list[BBoxPrediction], str, int, int]:
        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size
        bbox_level: BBoxLevel = self.config.bbox_level  # type: ignore[assignment]

        if bbox_level == "full":
            boxes = extract_bboxes({}, width, height, "full", self.config.crop_padding, self.config.min_box_height)
        else:
            doc = DocumentFile.from_images(str(image_path))
            ocr_result = self.models.det_model(doc)
            if not hasattr(ocr_result, "export"):
                raise TypeError("Expected doctr Document from ocr_predictor.")
            boxes = extract_bboxes(
                ocr_result.export(),
                width,
                height,
                bbox_level,
                self.config.crop_padding,
                self.config.min_box_height,
            )

        crops = [pil_image.crop(tuple(box["bbox"])) for box in boxes]
        texts = run_trocr_batch(
            self.models.trocr_processor,
            self.models.trocr_model,
            crops,
            self.config.batch_size,
            self.models.device,
        )

        predictions: list[BBoxPrediction] = []
        for box, text in zip(boxes, texts):
            predictions.append(
                BBoxPrediction(
                    level=box["level"],
                    index=box["index"],
                    bbox=box["bbox"],
                    text=text.strip(),
                    block_index=box.get("block_index"),
                    line_index=box.get("line_index"),
                    word_index=box.get("word_index"),
                )
            )

        full_text = " ".join(pred.text for pred in predictions if pred.text).strip()
        return predictions, full_text, width, height

    def save_overlay(
        self,
        image_path: Path,
        predictions: list[BBoxPrediction],
        output_path: Path,
        highlight_bboxes: list[list[int]] | None = None,
    ) -> Path:
        highlight_bboxes = highlight_bboxes or []
        highlight_set = {tuple(b) for b in highlight_bboxes}

        img_cv = cv2.imread(str(image_path))
        if img_cv is None:
            raise ValueError(f"Could not read image for overlay: {image_path}")

        for pred in predictions:
            x0, y0, x1, y1 = pred.bbox
            color = (255, 200, 0) if tuple(pred.bbox) in highlight_set else (0, 255, 0)
            cv2.rectangle(img_cv, (x0, y0), (x1, y1), color, 2)
            if pred.text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1
                text_y = y0 - 6 if y0 - 6 > 15 else y0 + 15
                (text_w, text_h), _ = cv2.getTextSize(pred.text, font, font_scale, thickness)
                cv2.rectangle(img_cv, (x0, text_y - text_h - 2), (x0 + text_w, text_y + 2), (0, 0, 0), -1)
                cv2.putText(img_cv, pred.text, (x0, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_cv)
        return output_path
