"""
Build BioBERT embeddings for the cleaned drug knowledge base.

Input:  mid_clean.csv
Output: embeddings/drug_embeddings.npy
        embeddings/drug_ids.json
        embeddings/index_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_models.paths import DEFAULT_BIOBERT, resolve_model_ref

DEFAULT_MODEL = DEFAULT_BIOBERT
DEFAULT_INPUT = Path(__file__).resolve().parent / "mid_clean.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "embeddings"


def embedding_text(row: dict[str, str]) -> str:
    parts = [row["generic_name"], row["strength"], row["form"]]
    return " ".join(part.strip() for part in parts if part.strip())


def load_records(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        cls_vectors = outputs.last_hidden_state[:, 0, :]
        cls_vectors = torch.nn.functional.normalize(cls_vectors, p=2, dim=1)
        vectors.append(cls_vectors.cpu().numpy())

    return np.vstack(vectors).astype(np.float32)


def build_index(
    csv_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    records = load_records(csv_path)
    if not records:
        raise ValueError(f"No records found in {csv_path}")

    texts = [embedding_text(record) for record in records]
    drug_ids = [record["drug_id"] for record in records]

    model_path = resolve_model_ref(model_name)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    print(f"Encoding {len(texts)} drugs on {device} ...")
    embeddings = encode_texts(texts, tokenizer, model, device, batch_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "drug_embeddings.npy", embeddings)
    (output_dir / "drug_ids.json").write_text(json.dumps(drug_ids, indent=2), encoding="utf-8")
    (output_dir / "embedding_texts.json").write_text(
        json.dumps(dict(zip(drug_ids, texts)), indent=2),
        encoding="utf-8",
    )

    meta = {
        "model_key": model_name,
        "model_path": str(resolve_model_ref(model_name)),
        "embedding_dim": int(embeddings.shape[1]),
        "record_count": len(records),
        "source_csv": str(csv_path.resolve()),
    }
    (output_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved embeddings: {embeddings.shape} -> {output_dir}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BioBERT drug embedding index")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_index(args.input, args.output_dir, args.model, args.batch_size, device)


if __name__ == "__main__":
    main()
