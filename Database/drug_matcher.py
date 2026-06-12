"""
Runtime drug matching against BioBERT embeddings.

Usage:
    python drug_matcher.py "Amox 500"
    python drug_matcher.py "PCM" --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline_models.paths import DEFAULT_BIOBERT, resolve_model_ref

DEFAULT_MODEL = DEFAULT_BIOBERT
DEFAULT_MIN_SCORE = 0.75

from prescription_pipeline.ner.aliases import resolve_generic_name


@dataclass
class MatchResult:
    drug_id: str
    score: float
    generic_name: str
    strength: str
    form: str
    brand_example: str
    food_instruction: str
    side_effects: str
    how_to_use: str
    safety_advice: str
    quick_tips: str


class DrugMatcher:
    def __init__(
        self,
        base_dir: Path | None = None,
        model_name: str = DEFAULT_MODEL,
        min_score: float = DEFAULT_MIN_SCORE,
        device: torch.device | None = None,
        verbose: bool = True,
    ) -> None:
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self.embeddings_dir = self.base_dir / "embeddings"
        self.model_name = model_name
        self.min_score = min_score
        self.verbose = verbose
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        meta_path = self.embeddings_dir / "index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing embedding index in {self.embeddings_dir}. Run build_embeddings.py first."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_name = meta.get("model_key", meta.get("model_name", self.model_name))

        self.embeddings = np.load(self.embeddings_dir / "drug_embeddings.npy")
        self.drug_ids: list[str] = json.loads((self.embeddings_dir / "drug_ids.json").read_text(encoding="utf-8"))
        self.records = self._load_records(self.base_dir / "mid_clean.csv")
        self.records_by_id = {record["drug_id"]: record for record in self.records}
        self.generic_names = {record["generic_name"] for record in self.records}

        model_ref = meta.get("model_key", meta.get("model_name", self.model_name))
        model_path = resolve_model_ref(model_ref)
        if self.verbose:
            print(f"Loading matcher model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @staticmethod
    def _load_records(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def normalize_query(self, query: str) -> str:
        query = re.sub(r"[^\w\s./+-]", " ", query.lower())
        query = re.sub(r"\s+", " ", query).strip()
        tokens = [resolve_generic_name(token) for token in query.split()]
        return " ".join(tokens)

    def extract_generic_hint(self, normalized_query: str) -> str | None:
        for token in normalized_query.split():
            if token in self.generic_names:
                return token
        return None

    @staticmethod
    def extract_strength_hint(query: str) -> str | None:
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|%|mg/ml|mcg/ml|mg/5ml)?",
            query.lower(),
        )
        if not match:
            return None
        amount, unit = match.group(1), match.group(2) or "mg"
        amount = amount.rstrip("0").rstrip(".") if "." in amount else amount
        return f"{amount}{unit}"

    @staticmethod
    def strength_overlap(query_strength: str | None, record_strength: str) -> float:
        if not query_strength or not record_strength:
            return 0.0
        query = query_strength.lower().replace(" ", "")
        record = record_strength.lower().replace(" ", "")
        if query == record:
            return 1.0
        query_amount = re.match(r"(\d+(?:\.\d+)?)", query)
        record_amount = re.match(r"(\d+(?:\.\d+)?)", record)
        if query_amount and record_amount and query_amount.group(1) == record_amount.group(1):
            return 0.7
        if query_amount and query_amount.group(1) in record:
            return 0.5
        return 0.0

    def rerank_score(
        self,
        vector_score: float,
        normalized_query: str,
        generic_hint: str | None,
        strength_hint: str | None,
        record: dict[str, str],
    ) -> float:
        score = vector_score
        generic_name = record["generic_name"]

        if generic_hint and generic_name == generic_hint:
            score += 0.12
        elif generic_hint and generic_hint in generic_name:
            score += 0.06
        elif generic_name in normalized_query:
            score += 0.08

        score += 0.08 * self.strength_overlap(strength_hint, record["strength"])
        return score

    @torch.no_grad()
    def _encode_query(self, query: str) -> np.ndarray:
        encoded = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded)
        vector = outputs.last_hidden_state[:, 0, :]
        vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        return vector.cpu().numpy().astype(np.float32)[0]

    def search(self, query: str, top_k: int = 5) -> list[MatchResult]:
        normalized = self.normalize_query(query)
        generic_hint = self.extract_generic_hint(normalized)
        strength_hint = self.extract_strength_hint(normalized)
        query_vector = self._encode_query(normalized)
        scores = self.embeddings @ query_vector

        candidate_count = min(len(scores), max(top_k * 20, 50))
        top_indices = np.argsort(scores)[::-1][:candidate_count]
        candidate_indices = set(int(i) for i in top_indices)

        if generic_hint:
            for index, drug_id in enumerate(self.drug_ids):
                if self.records_by_id[drug_id]["generic_name"] == generic_hint:
                    candidate_indices.add(index)

        ranked: list[tuple[float, float, dict[str, str]]] = []

        for index in candidate_indices:
            vector_score = float(scores[index])
            drug_id = self.drug_ids[index]
            record = self.records_by_id[drug_id]
            final_score = self.rerank_score(
                vector_score, normalized, generic_hint, strength_hint, record
            )
            ranked.append((final_score, vector_score, record))

        ranked.sort(key=lambda item: item[0], reverse=True)

        if generic_hint:
            generic_ranked = [item for item in ranked if item[2]["generic_name"] == generic_hint]
            if generic_ranked:
                ranked = generic_ranked

        results: list[MatchResult] = []

        for final_score, vector_score, record in ranked[:top_k]:
            results.append(
                MatchResult(
                    drug_id=record["drug_id"],
                    score=round(final_score, 4),
                    generic_name=record["generic_name"],
                    strength=record["strength"],
                    form=record["form"],
                    brand_example=record["brand_example"],
                    food_instruction=record["food_instruction"],
                    side_effects=record["side_effects"],
                    how_to_use=record["how_to_use"],
                    safety_advice=record["safety_advice"],
                    quick_tips=record["quick_tips"],
                )
            )

        return results

    @staticmethod
    def database_facts(match: MatchResult) -> dict[str, str]:
        return {
            "drug": f"{match.generic_name} {match.strength}".strip(),
            "form": match.form,
            "how_to_use": match.how_to_use,
            "side_effects": match.side_effects[:500],
            "safety_advice": match.safety_advice[:800],
            "food_instruction": match.food_instruction,
            "quick_tips": match.quick_tips[:400],
        }

    def match(self, query: str, top_k: int = 5) -> dict:
        normalized = self.normalize_query(query)
        results = self.search(query, top_k=top_k)
        if not results:
            return {"status": "no_match", "query": query, "matches": []}

        best = results[0]
        same_generic = [result for result in results if result.generic_name == best.generic_name]
        ambiguous = (
            len(same_generic) > 1
            and abs(same_generic[0].score - same_generic[1].score) < 0.03
            and same_generic[0].strength != same_generic[1].strength
        )

        if best.score < self.min_score:
            status = "low_confidence"
        elif ambiguous:
            status = "ambiguous"
        else:
            status = "matched"

        return {
            "status": status,
            "query": query,
            "normalized_query": normalized,
            "generic_hint": self.extract_generic_hint(normalized),
            "strength_hint": self.extract_strength_hint(normalized),
            "best_match": best.__dict__,
            "database_facts": self.database_facts(best),
            "matches": [result.__dict__ for result in results],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Match OCR drug text to knowledge base")
    parser.add_argument("query", help='Drug text from OCR, e.g. "Amox 500"')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args()

    matcher = DrugMatcher(min_score=args.min_score)
    result = matcher.match(args.query, top_k=args.top_k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
