from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.counseling.engine import PatientCounselorEngine
from prescription_pipeline.deployment.triggers import build_action_triggers
from prescription_pipeline.knowledge.lookup import KnowledgeLookup
from prescription_pipeline.models.loader import LoadedModels, load_models
from prescription_pipeline.ner.extractor import NerExtractor
from prescription_pipeline.ocr.engine import OcrEngine
from prescription_pipeline.schemas import PrescriptionResult

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(path.glob(f"*{ext}"))
        images.extend(path.glob(f"*{ext.upper()}"))
    return sorted(set(images))


class PrescriptionPipeline:
    """
    Production pipeline:
    DocTR bboxes -> TrOCR -> LayoutLMv3 NER (or regex fallback) -> BioBERT drug lookup
    -> optional DSPy patient counseling + action triggers.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.models: LoadedModels | None = None
        self.ocr: OcrEngine | None = None
        self.ner: NerExtractor | None = None
        self.lookup: KnowledgeLookup | None = None
        self.counselor: PatientCounselorEngine | None = None

    def load(self) -> None:
        if self.models is not None:
            return
        self.models = load_models(self.config)
        self.ocr = OcrEngine(self.config, self.models)
        self.ner = NerExtractor(self.config, self.models.layoutlm_engine)
        self.lookup = KnowledgeLookup(self.models.drug_matcher)
        if self.config.enable_counseling and self.config.patient_query.strip():
            self.counselor = PatientCounselorEngine(self.config)

    def process_image(self, image_path: Path) -> PrescriptionResult:
        self.load()
        assert self.ocr is not None and self.lookup is not None and self.ner is not None

        image_path = image_path.resolve()
        logger.info("Processing %s", image_path.name)

        predictions, full_text, width, height = self.ocr.recognize(image_path)
        extracted, ner_backend = self.ner.extract(image_path, predictions, width, height)
        logger.info("NER backend: %s | extracted %d medication(s)", ner_backend, len(extracted))
        medications = self.lookup.match_all(extracted)

        overlay_path = ""
        if self.config.save_overlays:
            med_bboxes = [med.ocr.bbox for med in medications if med.ocr.bbox]
            overlay_file = self.config.output_dir / "overlays" / f"overlay_{image_path.name}"
            saved = self.ocr.save_overlay(image_path, predictions, overlay_file, med_bboxes)
            overlay_path = str(saved.resolve())

        counseling = None
        if self.counselor is not None and self.config.patient_query.strip():
            rx_partial = PrescriptionResult(
                image_name=image_path.name,
                image_path=str(image_path),
                width=width,
                height=height,
                full_text=full_text,
                medications=medications,
                ocr_predictions=predictions,
                overlay_path=overlay_path,
                ner_backend=self.ner.active_backend,
            )
            counseling = self.counselor.counsel_prescription(
                rx_partial,
                self.config.patient_query.strip(),
                medication_index=self.config.counsel_medication_index,
            )
            logger.info("DSPy counseling complete for query: %r", self.config.patient_query[:80])
        action_triggers = build_action_triggers(medications)

        return PrescriptionResult(
            image_name=image_path.name,
            image_path=str(image_path),
            width=width,
            height=height,
            full_text=full_text,
            medications=medications,
            ocr_predictions=predictions,
            overlay_path=overlay_path,
            ner_backend=self.ner.active_backend,
            counseling=counseling,
            action_triggers=action_triggers,
        )

    def process_path(self, input_path: Path) -> dict:
        self.load()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_images(input_path)
        if not image_paths:
            raise FileNotFoundError(f"No prescription images found at {input_path}")

        results: list[dict] = []
        for image_path in image_paths:
            try:
                result = self.process_image(image_path)
                results.append(result.to_dict())
            except Exception:
                logger.exception("Failed to process %s", image_path)
                raise

        payload = {
            "pipeline_version": "1.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "trocr_model": self.config.trocr_model,
                "layoutlm_processor_model": self.config.layoutlm_processor_model,
                "layoutlm_ner_model": self.config.layoutlm_ner_model,
                "ner_backend": self.config.ner_backend,
                "bbox_level": self.config.bbox_level,
                "drug_matching": self.config.enable_drug_matching,
                "counselor_model": self.config.counselor_model,
                "counselor_temperature": self.config.counselor_temperature,
                "use_compiled_counselor": self.config.use_compiled_counselor,
                "counseling_enabled": self.config.enable_counseling,
            },
            "image_count": len(results),
            "prescriptions": results,
        }

        output_json = self.config.output_dir / "prescription_results.json"
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved results -> %s", output_json)
        payload["output_path"] = str(output_json.resolve())
        return payload
