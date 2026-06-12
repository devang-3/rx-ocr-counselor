from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.pipeline import PrescriptionPipeline


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Production prescription pipeline: DocTR -> TrOCR -> NER -> BioBERT KB match "
            "-> optional DSPy counseling"
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("page_image"),
        help="Prescription image file or folder",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for JSON results and overlay images",
    )
    parser.add_argument("--trocr-model", default=None, help="TrOCR registry key or HF model id")
    parser.add_argument(
        "--ner-backend",
        choices=("auto", "layoutlmv3", "regex"),
        default="auto",
        help="NER engine: auto uses LayoutLMv3 when fine-tuned weights exist, else regex",
    )
    parser.add_argument("--layoutlm-model", default=None, help="LayoutLMv3 processor registry key")
    parser.add_argument("--bbox-level", choices=("line", "word", "full"), default="line")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument("--min-box-height", type=int, default=8)
    parser.add_argument("--assume-straight-pages", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-drug-match", action="store_true", help="Skip BioBERT knowledge lookup")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--min-match-score", type=float, default=0.75)
    parser.add_argument(
        "--patient-query",
        default="",
        help="Patient question for DSPy counseling (requires counselor LM in pipeline_models/)",
    )
    parser.add_argument(
        "--medication-index",
        type=int,
        default=None,
        help="Medication index for counseling when multiple meds extracted",
    )
    parser.add_argument(
        "--counselor-model",
        default=None,
        help="Counselor LM registry key (default: qwen2.5-1.5b-instruct)",
    )
    parser.add_argument(
        "--counselor-temperature",
        type=float,
        default=0.1,
        help="Counselor LM temperature (plan default: 0.1)",
    )
    parser.add_argument(
        "--no-compiled-counselor",
        action="store_true",
        help="Use base ChainOfThought instead of compiled program in pipeline_models/",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    config = PipelineConfig(
        bbox_level=args.bbox_level,
        batch_size=args.batch_size,
        crop_padding=args.crop_padding,
        min_box_height=args.min_box_height,
        assume_straight_pages=args.assume_straight_pages,
        enable_drug_matching=not args.no_drug_match,
        save_overlays=not args.no_overlays,
        min_match_score=args.min_match_score,
        device=args.device,
        output_dir=args.output_dir,
    )
    if args.trocr_model:
        config.trocr_model = args.trocr_model
    config.ner_backend = args.ner_backend
    if args.layoutlm_model:
        config.layoutlm_processor_model = args.layoutlm_model
    config.patient_query = args.patient_query
    config.counsel_medication_index = args.medication_index
    config.counselor_temperature = args.counselor_temperature
    config.use_compiled_counselor = not args.no_compiled_counselor
    if args.counselor_model:
        config.counselor_model = args.counselor_model
    if args.patient_query.strip():
        config.enable_counseling = True

    pipeline = PrescriptionPipeline(config)
    payload = pipeline.process_path(args.input)

    print(f"Processed {payload['image_count']} image(s)")
    print(f"Results: {payload['output_path']}")
    for rx in payload["prescriptions"]:
        med_count = len(rx.get("medications", []))
        print(f"  {rx['image_name']}: {med_count} medication(s)")
        counseling = rx.get("counseling")
        if counseling:
            print(f"    counseling: {counseling.get('patient_response', '')[:160]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
