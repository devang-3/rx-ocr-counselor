from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline_models.paths import resolve_model_ref

logger = logging.getLogger(__name__)


def _chat_completion_response(text: str, model_name: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, role="assistant")
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return SimpleNamespace(choices=[choice], model=model_name, usage=usage)


class LocalChatLM(dspy.BaseLM):
    """DSPy LM backed by a local HuggingFace instruct checkpoint in pipeline_models/."""

    def __init__(
        self,
        model_ref: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        device: str | None = None,
    ) -> None:
        super().__init__(
            model=str(model_ref),
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=True,
        )
        self.model_ref = model_ref
        self.model_path = Path(str(resolve_model_ref(model_ref)))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        path = str(self.model_path)
        logger.info("Loading counselor LM for DSPy from %s on %s", path, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # float16 on CPU too — halves RAM vs float32 (important for ~3B+ models).
        dtype = torch.float16 if self.device in {"cuda", "cpu"} else torch.float32
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                path,
                attn_implementation="sdpa",
                **load_kwargs,
            )
        except (ValueError, ImportError):
            self._model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        self._model.to(self.device)
        self._model.eval()

    def _build_messages(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> list[dict[str, str]]:
        if messages:
            return [{"role": m["role"], "content": str(m["content"])} for m in messages]
        if prompt:
            return [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": ""}]

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        chat_messages = self._build_messages(prompt, messages)
        temperature = float(kwargs.get("temperature", self.kwargs.get("temperature", 0.1)))
        max_new_tokens = int(kwargs.get("max_tokens", self.kwargs.get("max_tokens", 512)))

        if hasattr(self._tokenizer, "apply_chat_template"):
            input_ids = self._tokenizer.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            joined = "\n".join(f"{m['role']}: {m['content']}" for m in chat_messages)
            input_ids = self._tokenizer(joined, return_tensors="pt").input_ids

        input_ids = input_ids.to(self.device)
        pad_token_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": pad_token_id,
            "use_cache": False,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0, input_ids.shape[-1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _chat_completion_response(text, model_name=self.model)


def configure_counselor_lm(
    model_ref: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 512,
    device: str | None = None,
) -> LocalChatLM:
    lm = LocalChatLM(
        model_ref,
        temperature=temperature,
        max_tokens=max_tokens,
        device=device,
    )
    dspy.configure(lm=lm)
    return lm


# Backward-compatible alias from earlier Phi-3-only naming.
configure_phi3_lm = configure_counselor_lm
Phi3LocalLM = LocalChatLM
