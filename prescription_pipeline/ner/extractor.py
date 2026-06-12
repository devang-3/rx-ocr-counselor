from __future__ import annotations

import logging
from pathlib import Path

from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.ner.layoutlm_engine import LayoutLMv3NerEngine
from prescription_pipeline.ner.parser import PrescriptionParser
from prescription_pipeline.schemas import BBoxPrediction, ExtractedMedication

logger = logging.getLogger(__name__)


class NerExtractor:
    """Unified NER entry point: LayoutLMv3 when weights exist, else regex."""

    def __init__(
        self,
        config: PipelineConfig,
        layoutlm_engine: LayoutLMv3NerEngine | None = None,
    ) -> None:
        self.config = config
        self.layoutlm = layoutlm_engine
        self.regex = PrescriptionParser()

    @property
    def active_backend(self) -> str:
        if self.config.ner_backend == "regex":
            return "regex"
        if self.config.ner_backend == "layoutlmv3":
            if self.layoutlm and self.layoutlm.is_ready:
                return "layoutlmv3"
            logger.warning("layoutlmv3 requested but weights missing — using regex")
            return "regex"
        # auto
        if self.layoutlm and self.layoutlm.is_ready:
            return "layoutlmv3"
        return "regex"

    def extract(
        self,
        image_path: Path,
        predictions: list[BBoxPrediction],
        width: int,
        height: int,
    ) -> tuple[list[ExtractedMedication], str]:
        backend = self.active_backend
        if backend == "layoutlmv3" and self.layoutlm is not None:
            meds = self.layoutlm.parse_predictions(image_path, predictions, width, height)
            return meds, "layoutlmv3"

        meds = self.regex.parse_predictions(predictions)
        return meds, "regex"
