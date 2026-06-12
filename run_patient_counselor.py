#!/usr/bin/env python3
"""
Ask a patient question against existing prescription_results.json (no re-OCR).

Usage:
    python run_patient_counselor.py output/prescription_results.json \\
        --patient-query "Can I take this with food?" --rx-index 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.counseling.engine import PatientCounselorEngine
from prescription_pipeline.schemas import CounselorResult, MatchedMedication, PrescriptionResult


def _dict_to_prescription(data: dict) -> PrescriptionResult:
    medications = []
    for med in data.get("medications", []):
        ocr = med.get("ocr", {})
        from prescription_pipeline.schemas import ExtractedMedication

        medications.append(
            MatchedMedication(
                ocr=ExtractedMedication(**ocr),
                match_status=med.get("match_status", ""),
                canonical_name=med.get("canonical_name", ""),
                canonical_strength=med.get("canonical_strength", ""),
                canonical_form=med.get("canonical_form", ""),
                match_score=float(med.get("match_score", 0.0)),
                drug_id=med.get("drug_id", ""),
                database_facts=med.get("database_facts", {}),
                frequency_normalized=med.get("frequency_normalized", ""),
            )
        )
    return PrescriptionResult(
        image_name=data.get("image_name", ""),
        image_path=data.get("image_path", ""),
        width=int(data.get("width", 0)),
        height=int(data.get("height", 0)),
        full_text=data.get("full_text", ""),
        medications=medications,
        ocr_predictions=[],
        overlay_path=data.get("overlay_path", ""),
        ner_backend=data.get("ner_backend", "regex"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DSPy patient counselor on saved JSON")
    parser.add_argument("results_json", type=Path, help="prescription_results.json path")
    parser.add_argument("--patient-query", required=True)
    parser.add_argument("--rx-index", type=int, default=0, help="Prescription index in JSON")
    parser.add_argument("--medication-index", type=int, default=None)
    parser.add_argument("--counselor-model", default=None)
    parser.add_argument("--counselor-temperature", type=float, default=0.1)
    parser.add_argument("--counselor-max-tokens", type=int, default=512)
    parser.add_argument("--no-compiled-counselor", action="store_true")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    prescriptions = payload.get("prescriptions", [])
    if not prescriptions:
        raise SystemExit("No prescriptions in JSON")
    if args.rx_index < 0 or args.rx_index >= len(prescriptions):
        raise SystemExit(f"rx-index out of range (0-{len(prescriptions) - 1})")

    config = PipelineConfig(
        enable_counseling=True,
        patient_query=args.patient_query,
        counsel_medication_index=args.medication_index,
        counselor_temperature=args.counselor_temperature,
        counselor_max_tokens=args.counselor_max_tokens,
        use_compiled_counselor=not args.no_compiled_counselor,
    )
    if args.counselor_model:
        config.counselor_model = args.counselor_model

    rx = _dict_to_prescription(prescriptions[args.rx_index])
    engine = PatientCounselorEngine(config)
    result: CounselorResult = engine.counsel_prescription(
        rx,
        args.patient_query,
        medication_index=args.medication_index,
    )

    output = result.__dict__
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
