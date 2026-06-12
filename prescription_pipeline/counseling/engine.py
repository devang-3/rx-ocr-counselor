from __future__ import annotations

import logging
from pathlib import Path

import dspy

from pipeline_models.paths import DEFAULT_DSPY_COUNSELOR, is_downloaded, local_model_dir
from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.counseling.lm import configure_counselor_lm
from prescription_pipeline.counseling.serializers import (
    medication_to_database_facts,
    medication_to_ocr_variables,
    prescription_summary,
)
from prescription_pipeline.counseling.signatures import PatientCounselor
from prescription_pipeline.schemas import CounselorResult, MatchedMedication, PrescriptionResult

logger = logging.getLogger(__name__)

class PatientCounselorEngine:
    """DSPy ChainOfThought counselor backed by a local instruct LM."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._module: dspy.Module | None = None
        self._lm: dspy.BaseLM | None = None

    @property
    def compiled_program_path(self) -> Path:
        return local_model_dir(DEFAULT_DSPY_COUNSELOR) / "program"

    @property
    def is_ready(self) -> bool:
        return is_downloaded(self.config.counselor_model)

    def load(self) -> None:
        if self._module is not None:
            return
        if not self.is_ready:
            raise FileNotFoundError(
                f"Counselor LM not found. Run: python pipeline_models/download_models.py "
                f"{self.config.counselor_model}"
            )

        self._lm = configure_counselor_lm(
            self.config.counselor_model,
            temperature=self.config.counselor_temperature,
            max_tokens=self.config.counselor_max_tokens,
            device=self.config.device,
        )

        if self.config.use_compiled_counselor and self.compiled_program_path.exists():
            logger.info("Loading compiled DSPy counselor from %s", self.compiled_program_path)
            self._module = dspy.load(str(self.compiled_program_path))
        else:
            logger.info("Using base ChainOfThought counselor (not compiled)")
            self._module = dspy.ChainOfThought(PatientCounselor)

    def counsel_medication(
        self,
        medication: MatchedMedication,
        patient_query: str,
    ) -> CounselorResult:
        self.load()
        assert self._module is not None

        prediction = self._module(
            database_facts=medication_to_database_facts(medication),
            ocr_variables=medication_to_ocr_variables(medication),
            patient_query=patient_query,
        )
        reasoning = getattr(prediction, "reasoning", "") or ""
        response = getattr(prediction, "patient_response", "") or ""
        return CounselorResult(
            patient_query=patient_query,
            patient_response=response.strip(),
            reasoning=reasoning.strip(),
            medication_drug_name=medication.ocr.drug_name or medication.canonical_name,
            counseling_backend="dspy-chain-of-thought",
            counselor_model=self.config.counselor_model,
        )

    def counsel_prescription(
        self,
        result: PrescriptionResult,
        patient_query: str,
        medication_index: int | None = None,
    ) -> CounselorResult:
        if medication_index is not None:
            if medication_index < 0 or medication_index >= len(result.medications):
                raise IndexError(
                    f"medication_index {medication_index} out of range "
                    f"(0-{max(len(result.medications) - 1, 0)})"
                )
            return self.counsel_medication(result.medications[medication_index], patient_query)

        if len(result.medications) == 1:
            return self.counsel_medication(result.medications[0], patient_query)

        self.load()
        assert self._module is not None
        prediction = self._module(
            database_facts=prescription_summary(result),
            ocr_variables=prescription_summary(result),
            patient_query=patient_query,
        )
        reasoning = getattr(prediction, "reasoning", "") or ""
        response = getattr(prediction, "patient_response", "") or ""
        return CounselorResult(
            patient_query=patient_query,
            patient_response=response.strip(),
            reasoning=reasoning.strip(),
            medication_drug_name="(all medications)",
            counseling_backend="dspy-chain-of-thought",
            counselor_model=self.config.counselor_model,
        )
