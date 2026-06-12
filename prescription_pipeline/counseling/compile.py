"""
Compile the patient counselor with BootstrapFewShot and save to pipeline_models/.

Usage:
    python -m prescription_pipeline.counseling.compile
    python -m prescription_pipeline.counseling.compile --counselor-model qwen2.5-1.5b-instruct
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import dspy
from dspy.teleprompt import BootstrapFewShot

from pipeline_models.paths import DEFAULT_DSPY_COUNSELOR, local_model_dir
from prescription_pipeline.config import PipelineConfig
from prescription_pipeline.counseling.examples import build_training_examples
from prescription_pipeline.counseling.lm import configure_counselor_lm
from prescription_pipeline.counseling.signatures import PatientCounselor

logger = logging.getLogger(__name__)


def grounded_metric(example: dspy.Example, prediction, trace=None) -> bool:
    response = (getattr(prediction, "patient_response", "") or "").strip()
    if len(response) < 40:
        return False
    lowered = response.lower()
    if "provided" in lowered or "prescription" in lowered or "doctor" in lowered:
        return True
    return len(response) >= 80


def compile_counselor(config: PipelineConfig) -> dspy.Module:
    configure_counselor_lm(
        config.counselor_model,
        temperature=config.counselor_temperature,
        max_tokens=config.counselor_max_tokens,
        device=config.device,
    )
    student = dspy.ChainOfThought(PatientCounselor)
    trainset = build_training_examples()
    optimizer = BootstrapFewShot(
        metric=grounded_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=1,
    )
    logger.info("Compiling counselor on %d training examples", len(trainset))
    return optimizer.compile(student, trainset=trainset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile DSPy patient counselor (Phase 5.3)")
    parser.add_argument("--counselor-model", default=PipelineConfig.counselor_model)
    parser.add_argument("--temperature", type=float, default=PipelineConfig.counselor_temperature)
    parser.add_argument("--max-tokens", type=int, default=PipelineConfig.counselor_max_tokens)
    parser.add_argument("--device", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = PipelineConfig(
        counselor_model=args.counselor_model,
        counselor_temperature=args.temperature,
        counselor_max_tokens=args.max_tokens,
        device=args.device,
        use_compiled_counselor=False,
    )
    compiled = compile_counselor(config)

    out_dir = local_model_dir(DEFAULT_DSPY_COUNSELOR)
    out_dir.mkdir(parents=True, exist_ok=True)
    program_path = out_dir / "program"
    compiled.save(str(program_path))
    logger.info("Saved compiled counselor -> %s", program_path)
    print(f"Compiled program saved to {program_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
