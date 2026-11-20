"""Shared model-loading / generation glue for the training and evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.executor import ConductorGenerateFn
from src.workers import CONDUCTOR_KEY, BatchedGenerator, HFBatchGenerate

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_conductor(
    model_name: str,
    *,
    adapter: str | Path | None = None,
    trainable: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 64,
) -> tuple[Any, Any]:
    """Load base model (bf16 on GPU) and attach / load a LoRA adapter.

    `adapter=None` with `trainable=True` attaches a fresh adapter; with `trainable=False`
    it returns the plain base model (used for the un-finetuned baseline).
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        kwargs.update(torch_dtype=torch.bfloat16, device_map="auto")
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if trainable:
        model.gradient_checkpointing_enable()
        # All base weights are frozen: without this, checkpointed blocks receive no grad.
        model.enable_input_require_grads()

    if adapter and Path(adapter).exists():
        print(f"[model] Loading LoRA adapter from {adapter} (trainable={trainable})")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        print("[model] Attaching fresh LoRA adapter")
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=LORA_TARGETS,
            ),
        )
    elif adapter:
        raise FileNotFoundError(f"Adapter path does not exist: {adapter}")

    if not trainable:
        model.eval()
    return model, tokenizer


def build_generator(model: Any, tokenizer: Any, *, max_batch: int = 16) -> tuple[BatchedGenerator, HFBatchGenerate]:
    hf = HFBatchGenerate()
    hf.add(CONDUCTOR_KEY, model, tokenizer)
    return BatchedGenerator(hf, max_batch=max_batch), hf


def make_conductor_fn(
    generator: BatchedGenerator,
    *,
    temperature: float,
    max_new_tokens: int,
) -> ConductorGenerateFn:
    """Conductor policy pi_theta (adapter ON) as an async text generator."""

    async def _gen(messages: list[dict[str, str]]) -> str:
        text, _, _ = await generator.generate(
            messages, (CONDUCTOR_KEY, True, temperature, max_new_tokens)
        )
        return text

    return _gen
