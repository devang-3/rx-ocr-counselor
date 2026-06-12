"""
Download or relocate Hugging Face models into pipeline_models/.

Usage:
    python download_models.py biobert-base-cased-v1.1
    python download_models.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from pipeline_models.paths import MODELS, hf_model_id, is_downloaded, local_model_dir


def save_model(model_key: str, force: bool = False) -> Path:
    if model_key not in MODELS:
        raise KeyError(f"Unknown model key: {model_key}")

    entry = MODELS[model_key]
    hf_id = str(entry.get("hf_id", ""))
    if not hf_id:
        dest = local_model_dir(model_key)
        raise ValueError(
            f"{model_key} has no Hugging Face id — place fine-tuned weights in {dest} "
            "(LayoutLMv3ForTokenClassification after Phase 3 training)."
        )

    dest = local_model_dir(model_key)
    if is_downloaded(model_key) and not force:
        print(f"Already present: {dest}")
        return dest

    hf_id = hf_model_id(model_key)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {hf_id} -> {dest}")
    snapshot_download(repo_id=hf_id, local_dir=str(dest))
    print(f"Saved {model_key} to {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pipeline models to fixed local dirs")
    parser.add_argument(
        "models",
        nargs="*",
        help=f"Model keys: {', '.join(MODELS)}",
    )
    parser.add_argument("--all", action="store_true", help="Download every registered model")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = parser.parse_args()

    keys = list(MODELS) if args.all else args.models
    if not keys:
        parser.error("Pass model keys or use --all")

    for key in keys:
        save_model(key, force=args.force)


if __name__ == "__main__":
    main()
