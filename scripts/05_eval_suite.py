"""Evaluation suite covering RQ1 (baselines), RQ2 (recursion), RQ3 (zero-shot held-out Opus & ID swap), and RQ4 (Pareto)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import WorkerSpec, build_episode_worker_view, load_pool_config
from src.executor import RecursiveOrchestrator, RolloutTrace
from src.metrics import plot_pareto_frontier, summarize_traces
from src.prompts import TASK_FORMAT_GUIDANCE
from src.verifiers import verify_answer
from src.workers import (
    AnthropicWorkerBackend,
    LocalHFWorkerBackend,
    SQLiteResponseCache,
    WorkerRouter,
)


def _load_eval_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


async def _eval_single_worker(
    spec: WorkerSpec,
    router: WorkerRouter,
    rows: list[dict[str, Any]],
    concurrency: int = 12,
) -> dict[str, Any]:
    """Evaluate a single concrete worker answering all problems directly."""
    sem = asyncio.Semaphore(concurrency)
    correct_flags: list[bool] = []
    costs: list[float] = []

    async def _run_one(item: dict[str, Any]) -> tuple[bool, float]:
        hint = TASK_FORMAT_GUIDANCE.get(item["task_type"], "")
        msgs = [
            {"role": "system", "content": f"Solve the problem accurately. {hint}"},
            {"role": "user", "content": item["prompt"]},
        ]
        async with sem:
            resp = await router.call(spec, msgs)
        ok, _ = await asyncio.to_thread(
            verify_answer,
            item["task_type"],
            resp.text,
            item["ground_truth"],
            item.get("metadata"),
        )
        return ok, resp.cost_usd

    results = await asyncio.gather(*(_run_one(r) for r in rows))
    for ok, c in results:
        correct_flags.append(ok)
        costs.append(c)

    return {
        "label": spec.name,
        "group": "single_model",
        "accuracy": round(sum(1 for x in correct_flags if x) / max(len(correct_flags), 1), 4),
        "avg_cost_usd": round(sum(costs) / max(len(costs), 1), 6),
        "num_episodes": len(rows),
    }


async def _eval_conductor_setting(
    *,
    label: str,
    rows: list[dict[str, Any]],
    workers: list[WorkerSpec],
    router: WorkerRouter,
    conductor_generate_fn: Any,
    max_steps: int,
    max_depth: int,
    permute_ids: bool,
    cost_aware: bool,
    seed: int = 123,
    concurrency: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    orchestrator = RecursiveOrchestrator(
        worker_router=router,
        max_steps=max_steps,
        max_depth=max_depth,
        cost_aware=cost_aware,
    )
    sem = asyncio.Semaphore(concurrency)

    async def _run_one(idx: int, item: dict[str, Any]) -> tuple[RolloutTrace, bool, dict[str, Any]]:
        view = build_episode_worker_view(
            workers,
            permute_ids=permute_ids,
            rng=random.Random(seed + idx),
        )
        async with sem:
            tr = await orchestrator.execute_episode(
                task_prompt=item["prompt"],
                task_type=item["task_type"],
                worker_view=view,
                conductor_generate_fn=conductor_generate_fn,
            )
        if tr.format_valid:
            ok, reason = await asyncio.to_thread(
                verify_answer,
                item["task_type"],
                tr.final_answer,
                item["ground_truth"],
                item.get("metadata"),
            )
        else:
            ok, reason = False, tr.parse_error or "invalid_format"

        detail = {
            "task_id": item["id"],
            "task_type": item["task_type"],
            "correct": ok,
            "reason": reason,
            "format_valid": tr.format_valid,
            "cost_usd": tr.total_cost_usd,
            "num_steps": len(tr.steps),
            "parallel_width": tr.max_parallel_width,
            "recursion_triggered": tr.recursion_triggered,
            "workers_used": [s.worker_name for s in tr.steps],
        }
        return tr, ok, detail

    tasks = [_run_one(i, item) for i, item in enumerate(rows)]
    traces: list[RolloutTrace] = []
    flags: list[bool] = []
    details: list[dict[str, Any]] = []

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Eval [{label}]"):
        tr, ok, det = await coro
        traces.append(tr)
        flags.append(ok)
        details.append(det)

    summary = summarize_traces(traces, flags)
    summary["label"] = label
    summary["group"] = "conductor"
    return summary, details


async def run_eval(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pool_cfg = load_pool_config(args.pool)
    rows = _load_eval_rows(Path(args.eval_data), args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter and Path(args.adapter).exists():
        model = PeftModel.from_pretrained(base, args.adapter)
    else:
        model = base
    model.eval()

    gpu_lock = asyncio.Lock()
    cache = SQLiteResponseCache(out_dir / "eval_cache.sqlite")
    anthropic_backend = AnthropicWorkerBackend(max_concurrency=args.max_concurrency, cache=cache)
    conductor_base_backend = LocalHFWorkerBackend(
        model, tokenizer, disable_adapter=True, cache=cache, gpu_lock=gpu_lock
    )
    router = WorkerRouter(
        anthropic_backend=anthropic_backend,
        conductor_base_backend=conductor_base_backend,
    )

    def _greedy_conductor_sync(messages: list[dict[str, str]]) -> str:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        p_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.eval_temperature > 0.0,
                temperature=args.eval_temperature if args.eval_temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, p_len:], skip_special_tokens=True).strip()

    async def _conductor_gen(messages: list[dict[str, str]]) -> str:
        async with gpu_lock:
            return await asyncio.to_thread(_greedy_conductor_sync, messages)

    report: dict[str, Any] = {"pool": pool_cfg.name, "num_eval_tasks": len(rows), "results": []}

    # 1. Single-worker baselines (RQ1 & RQ4 anchor points)
    if args.run_single_baselines:
        all_concrete = [
            w
            for w in pool_cfg.active_workers(include_held_out=args.include_held_out, allow_recursion=False)
        ]
        for w in all_concrete:
            res = await _eval_single_worker(w, router, rows, concurrency=args.max_concurrency)
            print(f"[single] {w.name}: acc={res['accuracy']*100:.1f}% cost=${res['avg_cost_usd']:.5f}")
            report["results"].append(res)

    # 2. Conductor canonical (with recursion)
    train_workers_rec = pool_cfg.active_workers(include_held_out=False, allow_recursion=True)
    res_rec, det_rec = await _eval_conductor_setting(
        label="Conductor (Recursive)",
        rows=rows,
        workers=train_workers_rec,
        router=router,
        conductor_generate_fn=_conductor_gen,
        max_steps=pool_cfg.max_steps,
        max_depth=pool_cfg.max_depth,
        permute_ids=False,
        cost_aware=args.cost_aware,
    )
    report["results"].append(res_rec)
    (out_dir / "details_recursive.json").write_text(json.dumps(det_rec, indent=2), encoding="utf-8")

    # 3. RQ2: No-Recursion ablation
    train_workers_norec = pool_cfg.active_workers(include_held_out=False, allow_recursion=False)
    res_norec, _ = await _eval_conductor_setting(
        label="Conductor (No-Recursion)",
        rows=rows,
        workers=train_workers_norec,
        router=router,
        conductor_generate_fn=_conductor_gen,
        max_steps=pool_cfg.max_steps,
        max_depth=1,
        permute_ids=False,
        cost_aware=args.cost_aware,
    )
    report["results"].append(res_norec)

    # 4. RQ3a: Permuted Worker IDs stress test
    res_perm, _ = await _eval_conductor_setting(
        label="Conductor (Permuted IDs)",
        rows=rows,
        workers=train_workers_rec,
        router=router,
        conductor_generate_fn=_conductor_gen,
        max_steps=pool_cfg.max_steps,
        max_depth=pool_cfg.max_depth,
        permute_ids=True,
        cost_aware=args.cost_aware,
    )
    report["results"].append(res_perm)

    # 5. RQ3b: Zero-shot Held-out Worker injection (e.g., Claude Opus unseen in training)
    if args.include_held_out and pool_cfg.held_out_workers:
        workers_plus_opus = pool_cfg.active_workers(include_held_out=True, allow_recursion=True)
        res_zs, det_zs = await _eval_conductor_setting(
            label="Conductor (+Zero-Shot Opus)",
            rows=rows,
            workers=workers_plus_opus,
            router=router,
            conductor_generate_fn=_conductor_gen,
            max_steps=pool_cfg.max_steps,
            max_depth=pool_cfg.max_depth,
            permute_ids=False,
            cost_aware=args.cost_aware,
        )
        report["results"].append(res_zs)
        (out_dir / "details_zeroshot_opus.json").write_text(json.dumps(det_zs, indent=2), encoding="utf-8")

    (out_dir / "eval_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_pareto_frontier(report["results"], out_dir / "rq4_pareto_frontier.png")
    cache.close()
    print(f"[done] Saved evaluation report & Pareto figure to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RQ1-RQ4 evaluation suite for mini-conductor.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--pool", type=str, default="configs/pool_claude_hybrid.yaml")
    parser.add_argument("--eval_data", type=str, default="data/eval_all.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/rq_eval")
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--include_held_out", action="store_true")
    parser.add_argument("--run_single_baselines", action="store_true", default=True)
    parser.add_argument("--cost_aware", action="store_true")
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_concurrency", type=int, default=16)
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
