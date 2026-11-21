"""Evaluate ONE Conductor adapter under several test-time pool conditions (+ optional baselines).

Conditions (greedy decoding, same eval tasks):
  canonical        training pool, canonical worker IDs
  permuted_ids     training pool, worker IDs randomly swapped per task          (RQ3)
  +heldout         held-out worker (e.g. Claude Opus) injected zero-shot        (RQ3)
  +heldout+perm    held-out worker injected AND all IDs randomly swapped        (RQ3)

`--baselines` adds single-worker direct answering and a static "all solve in parallel, strongest
judges" topology (RQ1 / RQ4 anchors). Cross-run comparisons (RQ2 / RQ3 / RQ4) are assembled
from several of these summaries by `scripts/06_run_matrix.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EpisodeWorkerView, WorkerSpec, build_episode_worker_view, load_pool_config
from src.executor import RecursiveOrchestrator, RolloutTrace
from src.metrics import accuracy_stats, summarize_traces
from src.prompts import TASK_FORMAT_GUIDANCE
from src.runtime import build_generator, load_conductor, load_jsonl, make_conductor_fn
from src.verifiers import verify_answer
from src.workers import SQLiteResponseCache, WorkerRouter, build_router


def stratified_limit(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Seeded subsample with (near-)equal task-type counts; `limit=None/0` keeps everything."""
    if not limit or limit >= len(rows):
        return rows
    by_type: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_type.setdefault(r["task_type"], []).append(r)
    rng = random.Random(seed)
    per_type = limit // len(by_type)
    out: list[dict[str, Any]] = []
    for t in sorted(by_type):
        rng.shuffle(by_type[t])
        out.extend(by_type[t][:per_type])
    return out


def static_parallel_judge_plan(view: EpisodeWorkerView) -> str:
    """Hand-written static topology: every concrete worker solves in parallel, the last one judges."""
    concrete = [s for s in view.ordered_specs if not s.is_self]
    ids = [s.id for s in concrete] + [concrete[-1].id]
    subtasks = ["Solve the problem independently and give your final answer."] * len(concrete)
    subtasks.append(
        "Compare the candidate solutions above, resolve any disagreement, and give the final answer."
    )
    access = [[] for _ in concrete] + [list(range(len(concrete)))]
    return f"<plan>\nmodel_id = {ids!r}\nsubtasks = {subtasks!r}\naccess_list = {access!r}\n</plan>"


async def _verify(item: dict[str, Any], text: str) -> tuple[bool, str]:
    return await asyncio.to_thread(
        verify_answer, item["task_type"], text, item["ground_truth"], item.get("metadata")
    )


async def eval_single_worker(
    spec: WorkerSpec, router: WorkerRouter, rows: list[dict[str, Any]], concurrency: int
) -> dict[str, Any]:
    """One concrete worker answers every problem directly (no orchestration)."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(item: dict[str, Any]) -> tuple[bool, float]:
        hint = TASK_FORMAT_GUIDANCE.get(item["task_type"], "")
        msgs = [
            {"role": "system", "content": f"Solve the problem accurately. {hint}"},
            {"role": "user", "content": item["prompt"]},
        ]
        async with sem:
            resp = await router.call(spec, msgs)
        ok, _ = await _verify(item, resp.text)
        return ok, resp.cost_usd

    results = await asyncio.gather(*(_one(r) for r in rows))
    return {
        "label": spec.name,
        "group": "single_model",
        "condition": "direct",
        **accuracy_stats([ok for ok, _ in results], [r["task_type"] for r in rows]),
        "avg_cost_usd": round(sum(c for _, c in results) / max(len(results), 1), 6),
    }


async def eval_orchestration(
    *,
    label: str,
    group: str,
    condition: str,
    rows: list[dict[str, Any]],
    workers: list[WorkerSpec],
    router: WorkerRouter,
    conductor_fn: Any,
    max_steps: int,
    max_depth: int,
    permute_ids: bool,
    cost_aware: bool,
    static_plan: bool = False,
    seed: int = 123,
    concurrency: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    orchestrator = RecursiveOrchestrator(
        worker_router=router, max_steps=max_steps, max_depth=max_depth, cost_aware=cost_aware
    )
    sem = asyncio.Semaphore(concurrency)

    async def _one(idx: int, item: dict[str, Any]) -> tuple[RolloutTrace, bool, dict[str, Any]]:
        view = build_episode_worker_view(
            workers, permute_ids=permute_ids, rng=random.Random(f"{seed}:{item['id']}")
        )
        async with sem:
            tr = await orchestrator.execute_episode(
                task_prompt=item["prompt"],
                task_type=item["task_type"],
                worker_view=view,
                conductor_generate_fn=conductor_fn,
                prefilled_completion=static_parallel_judge_plan(view) if static_plan else None,
            )
        if tr.format_valid:
            ok, reason = await _verify(item, tr.final_answer)
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
            "plan": tr.conductor_turns[0].completion_text,
        }
        return tr, ok, detail

    results = await asyncio.gather(*(_one(i, item) for i, item in enumerate(rows)))
    traces = [t for t, _, _ in results]
    flags = [ok for _, ok, _ in results]
    summary = summarize_traces(traces, flags)
    summary.update(accuracy_stats(flags, [r["task_type"] for r in rows]))
    summary.update(label=label, group=group, condition=condition)
    print(
        f"[eval] {label:22s} {condition:14s} acc={summary['accuracy']*100:5.1f}% "
        f"(+-{summary['acc_ci95']*100:.1f}) fmt={summary['format_rate']*100:5.1f}% "
        f"cost=${summary['avg_cost_usd']:.5f}",
        flush=True,
    )
    return summary, [d for _, _, d in results]


async def run_eval(args: argparse.Namespace) -> None:
    pool_cfg = load_pool_config(args.pool)
    rows = stratified_limit(load_jsonl(args.eval_data), args.limit, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_conductor(args.model, adapter=args.adapter or None, trainable=False)
    generator, hf = build_generator(model, tokenizer, max_batch=args.gen_batch_size)
    cache = SQLiteResponseCache(args.cache_db or (out_dir / "eval_cache.sqlite"))
    router = build_router(
        pool_cfg, generator=generator, hf=hf, cache=cache, max_api_concurrency=args.max_concurrency
    )
    conductor_fn = make_conductor_fn(generator, temperature=0.0, max_new_tokens=args.max_new_tokens)

    label = args.label or (Path(args.adapter).parent.name if args.adapter else "base-model")
    max_depth = pool_cfg.max_depth if args.allow_recursion else 1
    train_workers = pool_cfg.active_workers(include_held_out=False, allow_recursion=args.allow_recursion)
    plus_heldout = pool_cfg.active_workers(include_held_out=True, allow_recursion=args.allow_recursion)
    common = dict(
        rows=rows,
        router=router,
        conductor_fn=conductor_fn,
        max_steps=pool_cfg.max_steps,
        cost_aware=args.cost_aware,
        seed=args.seed,
        concurrency=args.max_concurrency,
    )
    report: dict[str, Any] = {
        "pool": pool_cfg.name,
        "label": label,
        "adapter": args.adapter,
        "num_eval_tasks": len(rows),
        "eval_task_ids": [r["id"] for r in rows],
        "results": [],
    }

    if args.baselines:
        for w in pool_cfg.active_workers(include_held_out=True, allow_recursion=False):
            res = await eval_single_worker(w, router, rows, args.max_concurrency)
            print(f"[single] {w.name}: acc={res['accuracy']*100:.1f}% cost=${res['avg_cost_usd']:.5f}", flush=True)
            report["results"].append(res)
        res, det = await eval_orchestration(
            label="static parallel+judge", group="static", condition="canonical",
            workers=pool_cfg.active_workers(include_held_out=False, allow_recursion=False),
            max_depth=1, permute_ids=False, static_plan=True, **common,
        )
        report["results"].append(res)
        (out_dir / "details_static.json").write_text(json.dumps(det, indent=2), encoding="utf-8")

    if not args.skip_conductor:
        conditions = [("canonical", train_workers, False), ("permuted_ids", train_workers, True)]
        if pool_cfg.held_out_workers:
            conditions += [("+heldout", plus_heldout, False), ("+heldout+perm", plus_heldout, True)]
        for cond, workers, permute in conditions:
            res, det = await eval_orchestration(
                label=label, group="conductor", condition=cond, workers=workers,
                max_depth=max_depth, permute_ids=permute, **common,
            )
            report["results"].append(res)
            safe = cond.replace("+", "plus_")
            (out_dir / f"details_{safe}.json").write_text(json.dumps(det, indent=2), encoding="utf-8")

    if router.anthropic_backend is not None:
        report["api_stop_reasons"] = dict(router.anthropic_backend.stop_reasons)
    (out_dir / "eval_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cache.close()
    print(f"[done] Saved evaluation summary to {out_dir / 'eval_summary.json'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one Conductor adapter (+ baselines).")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--pool", type=str, default="configs/pool_local_weak.yaml")
    parser.add_argument("--eval_data", type=str, default="data/eval_all.jsonl")
    parser.add_argument("--output_dir", type=str, default="results/rq_eval")
    parser.add_argument("--limit", type=int, default=150, help="Stratified across task types; 0 = all")
    parser.add_argument("--allow_recursion", action=argparse.BooleanOptionalAction, default=True,
                        help="Must match how the adapter was trained")
    parser.add_argument("--cost_aware", action="store_true", help="Must match training (cost_lambda > 0)")
    parser.add_argument("--baselines", action="store_true", help="Also run single-worker + static baselines")
    parser.add_argument("--skip_conductor", action="store_true", help="Baselines only")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--gen_batch_size", type=int, default=16)
    parser.add_argument("--max_concurrency", type=int, default=16)
    parser.add_argument("--cache_db", type=str, default="", help="Default: <output_dir>/eval_cache.sqlite")
    parser.add_argument("--seed", type=int, default=123)
    return parser


if __name__ == "__main__":
    asyncio.run(run_eval(build_arg_parser().parse_args()))
