"""Stage 1: End-to-End Single-GPU GRPO Training for mini-conductor.

Features:
- Batched GPU sampling of G top-level plans per prompt (`num_return_sequences=G`)
- True concurrent async execution of all B * G orchestration DAGs via `asyncio.gather`
- Supports RQ2 (`--credit_mode top_level | all_depth`), RQ3 (`--permute_ids`), RQ4 (`--cost_lambda`)
- Colab-safe periodic checkpointing + seamless `--resume` from Google Drive
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PoolConfig, build_episode_worker_view, load_pool_config
from src.executor import RecursiveOrchestrator, RolloutTrace
from src.grpo import ConductorGRPOTrainer, GRPOConfig, compute_reward
from src.metrics import plot_topology_evolution, summarize_traces
from src.prompts import build_conductor_messages
from src.verifiers import verify_answer
from src.workers import (
    AnthropicWorkerBackend,
    LocalHFWorkerBackend,
    SQLiteResponseCache,
    WorkerRouter,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_conductor_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "sdpa"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    base_model.gradient_checkpointing_enable()

    resume_dir = Path(args.save_dir) / "latest"
    if args.resume and resume_dir.exists():
        print(f"[resume] Loading LoRA adapter from {resume_dir}")
        model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    elif args.init_adapter and Path(args.init_adapter).exists():
        print(f"[init] Loading warm-start LoRA adapter from {args.init_adapter}")
        model = PeftModel.from_pretrained(base_model, args.init_adapter, is_trainable=True)
    else:
        print("[init] Attaching fresh LoRA adapter")
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
        model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()
    return model, tokenizer


def _build_worker_router(
    pool_cfg: PoolConfig,
    conductor_model: Any,
    conductor_tokenizer: Any,
    gpu_lock: asyncio.Lock,
    cache: SQLiteResponseCache,
    max_api_concurrency: int,
) -> WorkerRouter:
    active = pool_cfg.active_workers(include_held_out=False, allow_recursion=True)
    backends_needed = {w.backend for w in active}

    anthropic_backend = None
    if "anthropic" in backends_needed:
        anthropic_backend = AnthropicWorkerBackend(
            max_concurrency=max_api_concurrency,
            cache=cache,
        )

    conductor_base_backend = None
    if "local_conductor_base" in backends_needed:
        conductor_base_backend = LocalHFWorkerBackend(
            conductor_model,
            conductor_tokenizer,
            disable_adapter=True,
            cache=cache,
            gpu_lock=gpu_lock,
        )

    local_hf_backends: dict[str, LocalHFWorkerBackend] = {}
    if "local_hf" in backends_needed:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for w in active:
            if w.backend == "local_hf" and w.model_name not in local_hf_backends:
                print(f"[worker] Loading dedicated local HF specialist: {w.model_name}")
                w_tok = AutoTokenizer.from_pretrained(w.model_name, use_fast=True)
                if w_tok.pad_token is None:
                    w_tok.pad_token = w_tok.eos_token
                w_mod = AutoModelForCausalLM.from_pretrained(
                    w.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                w_mod.eval()
                local_hf_backends[w.model_name] = LocalHFWorkerBackend(
                    w_mod,
                    w_tok,
                    disable_adapter=False,
                    cache=cache,
                    gpu_lock=gpu_lock,
                )

    return WorkerRouter(
        anthropic_backend=anthropic_backend,
        conductor_base_backend=conductor_base_backend,
        local_hf_backends=local_hf_backends,
    )


def _sample_conductor_batch_sync(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    num_return_sequences: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """Generate `num_return_sequences` Conductor completions in one batched H100 call."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        if was_training:
            model.train()

    completions: list[str] = []
    for row in out_ids:
        text = tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
        completions.append(text)
    return completions


async def train_loop(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    pool_cfg = load_pool_config(args.pool)
    train_rows = _load_jsonl(Path(args.train_data))
    print(f"[data] Loaded {len(train_rows)} training tasks from {args.train_data}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "train_metrics.jsonl"

    model, tokenizer = _load_conductor_model(args)
    gpu_lock = asyncio.Lock()
    cache = SQLiteResponseCache(save_dir / "worker_cache.sqlite")

    router = _build_worker_router(
        pool_cfg,
        model,
        tokenizer,
        gpu_lock=gpu_lock,
        cache=cache,
        max_api_concurrency=args.max_api_concurrency,
    )

    grpo_cfg = GRPOConfig(
        group_size=args.group_size,
        learning_rate=args.lr,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        grpo_epochs=args.grpo_epochs,
        format_reward=args.format_reward,
        cost_lambda=args.cost_lambda,
        reference_cost_usd=pool_cfg.reference_cost_usd,
        credit_mode=args.credit_mode,
        depth_gamma=args.depth_gamma,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    trainer = ConductorGRPOTrainer(model, tokenizer, grpo_cfg)

    if args.resume and (save_dir / "latest").exists():
        trainer.load_optimizer_state(save_dir / "latest")
        print(f"[resume] Resumed optimizer state at global_step={trainer.global_step}")

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or save_dir.name,
            config=vars(args),
        )

    active_workers = pool_cfg.active_workers(
        include_held_out=False,
        allow_recursion=args.allow_recursion,
    )
    max_depth = pool_cfg.max_depth if args.allow_recursion else 1
    orchestrator = RecursiveOrchestrator(
        worker_router=router,
        max_steps=pool_cfg.max_steps,
        max_depth=max_depth,
        cost_aware=args.cost_lambda > 0.0,
    )

    async def _recursive_conductor_gen(messages: list[dict[str, str]]) -> str:
        async with gpu_lock:
            outs = await asyncio.to_thread(
                _sample_conductor_batch_sync,
                model,
                tokenizer,
                messages,
                num_return_sequences=1,
                temperature=grpo_cfg.temperature,
                max_new_tokens=grpo_cfg.max_new_tokens,
            )
            return outs[0]

    start_step = trainer.global_step
    for step in range(start_step + 1, args.total_steps + 1):
        batch = rng.sample(train_rows, k=min(args.batch_size, len(train_rows)))

        # 1. Sample G top-level completions per prompt on GPU in batched calls
        prepared: list[tuple[dict[str, Any], Any, list[str]]] = []
        for task in batch:
            view = build_episode_worker_view(
                active_workers,
                permute_ids=args.permute_ids,
                rng=rng,
            )
            outer_msgs = build_conductor_messages(
                task_prompt=task["prompt"],
                task_type=task["task_type"],
                worker_view=view,
                max_steps=pool_cfg.max_steps,
                depth=0,
                max_depth=max_depth,
                cost_aware=args.cost_lambda > 0.0,
            )
            async with gpu_lock:
                comps = await asyncio.to_thread(
                    _sample_conductor_batch_sync,
                    model,
                    tokenizer,
                    outer_msgs,
                    num_return_sequences=args.group_size,
                    temperature=grpo_cfg.temperature,
                    max_new_tokens=grpo_cfg.max_new_tokens,
                )
            prepared.append((task, view, comps))

        # 2. Concurrently execute all B * G orchestration DAGs (with recursive self-calls)
        async def _eval_group(
            task: dict[str, Any],
            view: Any,
            comps: list[str],
        ) -> tuple[list[RolloutTrace], list[bool], list[float]]:
            traces = await asyncio.gather(
                *(
                    orchestrator.execute_episode(
                        task_prompt=task["prompt"],
                        task_type=task["task_type"],
                        worker_view=view,
                        conductor_generate_fn=_recursive_conductor_gen,
                        prefilled_completion=comp,
                    )
                    for comp in comps
                )
            )
            correct_flags: list[bool] = []
            rewards: list[float] = []
            for tr in traces:
                if tr.format_valid:
                    ok, _ = await asyncio.to_thread(
                        verify_answer,
                        task["task_type"],
                        tr.final_answer,
                        task["ground_truth"],
                        task.get("metadata"),
                    )
                else:
                    ok = False
                r = compute_reward(
                    tr,
                    ok,
                    format_reward=grpo_cfg.format_reward,
                    cost_lambda=grpo_cfg.cost_lambda,
                    reference_cost_usd=grpo_cfg.reference_cost_usd,
                )
                correct_flags.append(ok)
                rewards.append(r)
            return list(traces), correct_flags, rewards

        group_results = await asyncio.gather(
            *(_eval_group(t, v, c) for t, v, c in prepared)
        )

        # 3. Run GRPO policy gradient update on GPU
        grpo_groups = [(traces, rewards) for traces, _, rewards in group_results]
        opt_stats = trainer.update_from_groups(grpo_groups)

        # 4. Aggregate & log topology + optimization metrics
        all_traces = [tr for traces, _, _ in group_results for tr in traces]
        all_correct = [c for _, flags, _ in group_results for c in flags]
        all_rewards = [r for _, _, rewards in group_results for r in rewards]

        topo_stats = summarize_traces(all_traces, all_correct, all_rewards)
        log_row = {"step": step, **topo_stats, **opt_stats}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

        print(
            f"[step {step:03d}/{args.total_steps}] "
            f"acc={topo_stats['accuracy']*100:.1f}% "
            f"fmt={topo_stats['format_rate']*100:.1f}% "
            f"rew={topo_stats['avg_reward']:.3f} "
            f"par={topo_stats['avg_parallel_width']:.2f} "
            f"rec={topo_stats['recursion_rate']*100:.1f}% "
            f"cost=${topo_stats['avg_cost_usd']:.4f} "
            f"loss={opt_stats['loss']:.4f} kl={opt_stats['kl']:.4f} "
            f"turns={int(opt_stats['num_turns_updated'])}"
        )

        if wandb_run is not None:
            flat_log = {k: v for k, v in log_row.items() if isinstance(v, (int, float))}
            for w_name, share in topo_stats.get("worker_usage_share", {}).items():
                flat_log[f"worker_share/{w_name}"] = share
            wandb_run.log(flat_log, step=step)

        if step % args.save_every == 0 or step == args.total_steps:
            trainer.save_checkpoint(save_dir, tag="latest")
            if step % (args.save_every * 2) == 0:
                trainer.save_checkpoint(save_dir, tag=f"step_{step:04d}")
            try:
                plot_topology_evolution(metrics_path, save_dir / "rq1_topology_evolution.png")
            except Exception as exc:
                print(f"[warn] Plot update skipped: {exc}")

    cache.close()
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU GRPO training for mini-conductor.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--init_adapter", type=str, default="")
    parser.add_argument("--pool", type=str, default="configs/pool_claude_hybrid.yaml")
    parser.add_argument("--train_data", type=str, default="data/train.jsonl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/grpo_run")
    parser.add_argument("--resume", action="store_true")

    # RL & RQ experiment switches
    parser.add_argument("--total_steps", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.04)
    parser.add_argument("--grpo_epochs", type=int, default=1)
    parser.add_argument("--format_reward", type=float, default=0.2)
    parser.add_argument("--cost_lambda", type=float, default=0.0)
    parser.add_argument("--credit_mode", choices=["top_level", "all_depth"], default="all_depth")
    parser.add_argument("--depth_gamma", type=float, default=1.0)
    parser.add_argument("--allow_recursion", action="store_true")
    parser.add_argument("--permute_ids", action="store_true")

    # Generation & LoRA
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_api_concurrency", type=int, default=24)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(train_loop(args))


if __name__ == "__main__":
    main()
