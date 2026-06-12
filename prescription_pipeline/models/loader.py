from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from doctr.models import ocr_predictor
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor, TrOCRProcessor, VisionEncoderDecoderModel

from pipeline_models.paths import is_layoutlm_ner_ready, resolve_model_ref
from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.ner.labels import ID2LABEL
from prescription_pipeline.ner.layoutlm_engine import LayoutLMv3NerEngine

logger = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    device: torch.device
    det_model: object | None
    trocr_processor: TrOCRProcessor
    trocr_model: VisionEncoderDecoderModel
    layoutlm_engine: LayoutLMv3NerEngine | None = None
    drug_matcher: object | None = None


def _load_id2label(ner_model_path: Path) -> dict[int, str]:
    config_path = ner_model_path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        id2label = config.get("id2label")
        if id2label:
            return {int(k): v for k, v in id2label.items()}
    return ID2LABEL


def load_layoutlm_engine(config: PipelineConfig, device: torch.device) -> LayoutLMv3NerEngine:
    processor_path = resolve_model_ref(config.layoutlm_processor_model)
    logger.info("Loading LayoutLMv3 processor from %s", processor_path)
    processor = LayoutLMv3Processor.from_pretrained(processor_path, apply_ocr=False)

    ner_model: LayoutLMv3ForTokenClassification | None = None
    id2label = ID2LABEL

    if is_layoutlm_ner_ready(config.layoutlm_ner_model):
        ner_path = resolve_model_ref(config.layoutlm_ner_model)
        logger.info("Loading LayoutLMv3 NER model from %s", ner_path)
        ner_model = LayoutLMv3ForTokenClassification.from_pretrained(ner_path).to(device)
        id2label = _load_id2label(Path(str(ner_path)))
        ner_model.eval()
    else:
        logger.info(
            "LayoutLMv3 NER weights not in pipeline_models/%s — regex fallback until Phase 3 fine-tune",
            config.layoutlm_ner_model,
        )

    return LayoutLMv3NerEngine(processor=processor, model=ner_model, device=device, id2label=id2label)


def load_models(config: PipelineConfig) -> LoadedModels:
    device = torch.device(
        config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    det_model = None
    if config.bbox_level != "full":
        logger.info("Loading DocTR detector (assume_straight_pages=%s)", config.assume_straight_pages)
        det_model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="parseq",
            pretrained=True,
            assume_straight_pages=config.assume_straight_pages,
            export_as_straight_boxes=not config.assume_straight_pages,
        )

    trocr_path = resolve_model_ref(config.trocr_model)
    logger.info("Loading TrOCR from %s", trocr_path)
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_path)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_path).to(device)
    trocr_model.eval()

    layoutlm_engine = None
    if config.ner_backend in {"auto", "layoutlmv3"}:
        layoutlm_engine = load_layoutlm_engine(config, device)

    drug_matcher = None
    if config.enable_drug_matching:
        from Database.drug_matcher import DrugMatcher

        logger.info("Loading drug matcher from %s", config.database_dir)
        drug_matcher = DrugMatcher(
            base_dir=config.database_dir,
            model_name=config.biobert_model,
            min_score=config.min_match_score,
            device=device,
            verbose=False,
        )

    return LoadedModels(
        device=device,
        det_model=det_model,
        trocr_processor=trocr_processor,
        trocr_model=trocr_model,
        layoutlm_engine=layoutlm_engine,
        drug_matcher=drug_matcher,
    )
