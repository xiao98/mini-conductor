"""Stage 0: Lightweight LoRA SFT warm-start on verified Conductor trajectories.

Computes cross-entropy loss strictly over Conductor completion tokens (prompt tokens masked
with -100) and saves the LoRA adapter ready for Stage 1 GRPO training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from tqdm import tqdm


def _load_sft_examples(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT warm-start for mini-conductor.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", type=str, default="data/sft_warmup.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_warmup")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    examples = _load_sft_examples(Path(args.data))
    print(f"[sft] Loaded {len(examples)} turns from {args.data}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    device = next(model.parameters()).device
    model.train()

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(examples)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        steps_in_epoch = 0

        pbar = tqdm(enumerate(examples, start=1), total=len(examples), desc=f"SFT Epoch {epoch}")
        for idx, ex in pbar:
            prompt_text = tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            comp_ids = tokenizer.encode(ex["completion"], add_special_tokens=False) + [
                int(tokenizer.eos_token_id)
            ]

            full_ids = (prompt_ids + comp_ids)[: args.max_seq_len]
            p_len = min(len(prompt_ids), len(full_ids))
            if p_len >= len(full_ids):
                continue

            labels = [-100] * p_len + full_ids[p_len:]
            input_t = torch.tensor([full_ids], dtype=torch.long, device=device)
            label_t = torch.tensor([labels], dtype=torch.long, device=device)

            out = model(input_ids=input_t, labels=label_t, use_cache=False)
            loss = out.loss / args.grad_accum
            loss.backward()

            running_loss += float(out.loss.detach().item())
            steps_in_epoch += 1

            if idx % args.grad_accum == 0 or idx == len(examples):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss=f"{running_loss / max(steps_in_epoch, 1):.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[sft] Saved warm-start LoRA adapter to {out_dir}")


if __name__ == "__main__":
    main()
