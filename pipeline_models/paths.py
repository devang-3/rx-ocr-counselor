"""Fixed local paths for pipeline model weights."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS: dict[str, dict[str, str | Path | bool]] = {
    "biobert-base-cased-v1.1": {
        "hf_id": "dmis-lab/biobert-base-cased-v1.1",
        "local_dir": ROOT / "biobert-base-cased-v1.1",
    },
    "trocr-base-handwritten": {
        "hf_id": "microsoft/trocr-base-handwritten",
        "local_dir": ROOT / "trocr-base-handwritten",
    },
    "layoutlmv3-base": {
        "hf_id": "microsoft/layoutlmv3-base",
        "local_dir": ROOT / "layoutlmv3-base",
    },
    "layoutlmv3-rx-ner": {
        "hf_id": "",
        "local_dir": ROOT / "layoutlmv3-rx-ner",
        "optional": True,
        "description": "Fine-tuned LayoutLMv3ForTokenClassification (Phase 3 training)",
    },
    "qwen2.5-1.5b-instruct": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "local_dir": ROOT / "qwen2.5-1.5b-instruct",
        "description": "Default DSPy counselor (~1.5B, plan ~2B target)",
    },
    "qwen2.5-0.5b-instruct": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_dir": ROOT / "qwen2.5-0.5b-instruct",
        "description": "Smallest counselor option (~0.5B, fastest on CPU)",
    },
    "phi-3-mini-4k-instruct": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "local_dir": ROOT / "phi-3-mini-4k-instruct",
        "description": "Larger counselor option (~3.8B, slower on CPU)",
    },
    "dspy-patient-counselor": {
        "hf_id": "",
        "local_dir": ROOT / "dspy-patient-counselor",
        "optional": True,
        "description": "BootstrapFewShot-compiled DSPy ChainOfThought patient counselor",
    },
}

DEFAULT_BIOBERT = "biobert-base-cased-v1.1"
DEFAULT_TROCR = "trocr-base-handwritten"
DEFAULT_LAYOUTLM = "layoutlmv3-base"
DEFAULT_LAYOUTLM_NER = "layoutlmv3-rx-ner"
DEFAULT_COUNSELOR_LM = "qwen2.5-1.5b-instruct"
DEFAULT_DSPY_COUNSELOR = "dspy-patient-counselor"


def local_model_dir(model_key: str) -> Path:
    if model_key not in MODELS:
        raise KeyError(f"Unknown model key: {model_key}. Known: {', '.join(MODELS)}")
    return Path(MODELS[model_key]["local_dir"])


def hf_model_id(model_key: str) -> str:
    if model_key not in MODELS:
        raise KeyError(f"Unknown model key: {model_key}. Known: {', '.join(MODELS)}")
    return str(MODELS[model_key]["hf_id"])


def is_downloaded(model_key: str) -> bool:
    model_dir = local_model_dir(model_key)
    return model_dir.is_dir() and (model_dir / "config.json").exists()


def resolve_model_path(model_key: str) -> str | Path:
    """Return local path when downloaded, otherwise the Hugging Face model id."""
    if is_downloaded(model_key):
        return local_model_dir(model_key)
    return hf_model_id(model_key)


def resolve_model_ref(model_ref: str) -> str | Path:
    """Resolve a registry key or known HF id; pass through other refs."""
    if model_ref in MODELS:
        return resolve_model_path(model_ref)
    for key, entry in MODELS.items():
        if entry.get("hf_id") and entry["hf_id"] == model_ref:
            return resolve_model_path(key)
    return model_ref


def is_layoutlm_ner_ready(model_key: str = DEFAULT_LAYOUTLM_NER) -> bool:
    """True when fine-tuned LayoutLMv3 token-classification weights exist locally."""
    if model_key not in MODELS:
        return False
    model_dir = local_model_dir(model_key)
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    architectures = config.get("architectures", [])
    return any("TokenClassification" in arch for arch in architectures)
