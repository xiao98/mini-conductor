"""Stage 1: End-to-End Single-GPU GRPO Training for mini-conductor.

Features:
- All B * G rollouts run concurrently; every local generation (top-level plans, recursive
  inner plans, free local-worker calls) is micro-batched on the GPU by `BatchedGenerator`
- Supports RQ2 (`--credit_mode`, `--allow_recursion`), RQ3 (`--permute_ids`), RQ4 (`--cost_lambda`)
- Colab-safe: atomic periodic checkpoints + deterministic `--resume` from Google Drive
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import build_episode_worker_view, load_pool_config
from src.executor import RecursiveOrchestrator, RolloutTrace
from src.grpo import ConductorGRPOTrainer, GRPOConfig, compute_reward
from src.metrics import plot_topology_evolution, summarize_traces
from src.runtime import build_generator, load_conductor, load_jsonl, make_conductor_fn
from src.verifiers import verify_answer
from src.workers import SQLiteResponseCache, build_router


def step_rng(seed: int, step: int) -> random.Random:
    """Per-step RNG so a resumed run samples exactly the batches the original run would have."""
    return random.Random(f"{seed}:{step}")


def truncate_metrics(metrics_path: Path, last_step: int) -> None:
    """Drop logged rows newer than the restored checkpoint (they will be re-run)."""
    if not metrics_path.exists():
        return
    kept = [
        line
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["step"] <= last_step
    ]
    metrics_path.write_text("".join(k + "\n" for k in kept), encoding="utf-8")


def save_latest_atomic(trainer: ConductorGRPOTrainer, save_dir: Path) -> None:
    """Write to `latest.tmp` then swap, so a disconnect mid-save never corrupts `latest`."""
    tmp = trainer.save_checkpoint(save_dir, tag="latest.tmp")
    latest = save_dir / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    tmp.rename(latest)


async def train_loop(args: argparse.Namespace) -> None:
    pool_cfg = load_pool_config(args.pool)
    train_rows = load_jsonl(args.train_data)
    print(f"[data] Loaded {len(train_rows)} training tasks from {args.train_data}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "train_metrics.jsonl"
    latest = save_dir / "latest"
    resuming = args.resume and latest.exists()

    model, tokenizer = load_conductor(
        args.model,
        adapter=latest if resuming else (args.init_adapter or None),
        trainable=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model.print_trainable_parameters()

    generator, hf = build_generator(model, tokenizer, max_batch=args.gen_batch_size)
    cache = SQLiteResponseCache(args.cache_db or (save_dir / "worker_cache.sqlite"))
    router = build_router(
        pool_cfg,
        generator=generator,
        hf=hf,
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
    if resuming:
        trainer.load_optimizer_state(latest)
        print(f"[resume] Resumed at global_step={trainer.global_step}")
    truncate_metrics(metrics_path, trainer.global_step)
    (save_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or save_dir.name,
            config=vars(args),
            resume="allow",
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
    conductor_fn = make_conductor_fn(
        generator, temperature=grpo_cfg.temperature, max_new_tokens=grpo_cfg.max_new_tokens
    )

    async def _rollout_group(
        task: dict[str, Any], view: Any
    ) -> tuple[list[RolloutTrace], list[bool], list[float]]:
        """G independent rollouts of one task under one (possibly permuted) worker view."""
        traces = await asyncio.gather(
            *(
                orchestrator.execute_episode(
                    task_prompt=task["prompt"],
                    task_type=task["task_type"],
                    worker_view=view,
                    conductor_generate_fn=conductor_fn,
                )
                for _ in range(args.group_size)
            )
        )
        flags: list[bool] = []
        rewards: list[float] = []
        for tr in traces:
            ok = False
            if tr.format_valid:
                ok, _ = await asyncio.to_thread(
                    verify_answer,
                    task["task_type"],
                    tr.final_answer,
                    task["ground_truth"],
                    task.get("metadata"),
                )
            flags.append(ok)
            rewards.append(
                compute_reward(
                    tr,
                    ok,
                    format_reward=grpo_cfg.format_reward,
                    cost_lambda=grpo_cfg.cost_lambda,
                    reference_cost_usd=grpo_cfg.reference_cost_usd,
                )
            )
        return list(traces), flags, rewards

    for step in range(trainer.global_step + 1, args.total_steps + 1):
        rng = step_rng(args.seed, step)
        torch.manual_seed(args.seed * 100_003 + step)
        batch = rng.sample(train_rows, k=min(args.batch_size, len(train_rows)))
        views = [
            build_episode_worker_view(active_workers, permute_ids=args.permute_ids, rng=rng)
            for _ in batch
        ]

        # 1. Sample + execute all B * G orchestration DAGs concurrently
        group_results = await asyncio.gather(
            *(_rollout_group(t, v) for t, v in zip(batch, views))
        )

        # 2. GRPO policy-gradient update
        opt_stats = trainer.update_from_groups(
            [(traces, rewards) for traces, _, rewards in group_results]
        )

        # 3. Aggregate & log topology + optimization metrics
        all_traces = [tr for traces, _, _ in group_results for tr in traces]
        all_correct = [c for _, flags, _ in group_results for c in flags]
        all_rewards = [r for _, _, rewards in group_results for r in rewards]
        topo_stats = summarize_traces(all_traces, all_correct, all_rewards)
        log_row = {"step": step, **topo_stats, **opt_stats}
        if router.anthropic_backend is not None:
            log_row["api_stop_reasons"] = dict(router.anthropic_backend.stop_reasons)
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
            f"turns={int(opt_stats['num_turns_updated'])}",
            flush=True,
        )

        if wandb_run is not None:
            flat_log = {k: v for k, v in log_row.items() if isinstance(v, (int, float))}
            for w_name, share in topo_stats.get("worker_usage_share", {}).items():
                flat_log[f"worker_share/{w_name}"] = share
            wandb_run.log(flat_log, step=step)

        if step % args.save_every == 0 or step == args.total_steps:
            save_latest_atomic(trainer, save_dir)
            try:
                plot_topology_evolution(metrics_path, save_dir / "rq1_topology_evolution.png")
            except Exception as exc:
                print(f"[warn] Plot update skipped: {exc}")

    cache.close()
    if wandb_run is not None:
        wandb_run.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-GPU GRPO training for mini-conductor.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--init_adapter", type=str, default="")
    parser.add_argument("--pool", type=str, default="configs/pool_claude_hybrid.yaml")
    parser.add_argument("--train_data", type=str, default="data/train.jsonl")
    parser.add_argument("--save_dir", type=str, default="checkpoints/grpo_run")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache_db", type=str, default="", help="Default: <save_dir>/worker_cache.sqlite")

    # RL & RQ experiment switches
    parser.add_argument("--total_steps", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=8)
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
    parser.add_argument("--gen_batch_size", type=int, default=16, help="Max prompts per GPU generate() call")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_api_concurrency", type=int, default=24)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    asyncio.run(train_loop(build_arg_parser().parse_args()))
