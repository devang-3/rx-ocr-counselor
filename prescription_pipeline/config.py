from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline_models.paths import (
    DEFAULT_BIOBERT,
    DEFAULT_LAYOUTLM,
    DEFAULT_LAYOUTLM_NER,
    DEFAULT_COUNSELOR_LM,
    DEFAULT_TROCR,
)


@dataclass
class PipelineConfig:
    """Runtime configuration for the prescription pipeline."""

    trocr_model: str = DEFAULT_TROCR
    biobert_model: str = DEFAULT_BIOBERT
    layoutlm_processor_model: str = DEFAULT_LAYOUTLM
    layoutlm_ner_model: str = DEFAULT_LAYOUTLM_NER
    ner_backend: str = "auto"  # auto | layoutlmv3 | regex
    bbox_level: str = "line"
    batch_size: int = 8
    crop_padding: float = 0.08
    min_box_height: int = 8
    assume_straight_pages: bool = False
    min_match_score: float = 0.75
    match_top_k: int = 5
    enable_drug_matching: bool = True
    enable_counseling: bool = False
    counselor_model: str = DEFAULT_COUNSELOR_LM
    counselor_temperature: float = 0.1
    counselor_max_tokens: int = 512
    use_compiled_counselor: bool = True
    patient_query: str = ""
    counsel_medication_index: int | None = None
    save_overlays: bool = True
    device: str | None = None
    database_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "Database")
    output_dir: Path = field(default_factory=lambda: Path("output"))
