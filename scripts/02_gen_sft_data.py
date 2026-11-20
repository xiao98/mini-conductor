"""Stage 0: Synthesize and verify diverse-topology SFT warm-start trajectories using Claude.

This prevents cold-start GRPO collapse (format errors -> zero group variance) by teaching the
base 7B Conductor valid `<think>+<plan>` syntax across single-hop, parallel-verify, chain,
and recursive topologies before starting RL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import sys
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import WorkerSpec, build_episode_worker_view, load_pool_config
from src.executor import RecursiveOrchestrator
from src.verifiers import verify_answer
from src.workers import AnthropicWorkerBackend, SQLiteResponseCache, WorkerRouter


TOPOLOGY_HINTS = [
    ("single_hop", "If the problem looks straightforward, design a concise 1-step plan routing directly to the single best-suited worker."),
    ("parallel_verify", "Design a 3-step parallel-then-verify DAG: two independent solvers with `access_list=[]`, followed by a verifier with `access_list=[0, 1]`."),
    ("sequential_chain", "Design a 2- or 3-step sequential chain (`[]`, `[0]`, `[0, 1]`) separating decomposition/drafting from rigorous checking."),
    ("recursive_check", "If `self` is available, design a plan where concrete workers produce initial attempts and the final step invokes `self` to inspect outputs and orchestrate verification."),
]


async def synthesize_dataset(args: argparse.Namespace) -> None:
    pool_cfg = load_pool_config(args.pool)
    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    target_rows = rows[: args.num_samples]

    cache = SQLiteResponseCache(args.cache_db)
    anthropic_backend = AnthropicWorkerBackend(
        max_concurrency=args.max_concurrency,
        cache=cache,
    )
    # During SFT synthesis we use a cheap Claude Haiku proxy for any local_conductor_base worker
    # so this script can even run on CPU before renting an H100!
    haiku_proxy_spec = WorkerSpec(
        id=99,
        name="haiku-proxy",
        backend="anthropic",
        model_name=args.teacher_fast_model,
        description="Fast proxy",
        input_cost_per_mtok=0.10,
        output_cost_per_mtok=0.50,
    )

    class _ProxyLocalBackend(AnthropicWorkerBackend):
        async def generate(self, spec, messages, **kwargs):
            return await anthropic_backend.generate(haiku_proxy_spec, messages, **kwargs)

    router = WorkerRouter(
        anthropic_backend=anthropic_backend,
        conductor_base_backend=_ProxyLocalBackend(max_concurrency=args.max_concurrency, cache=cache),
    )

    teacher_spec = WorkerSpec(
        id=-1,
        name="teacher-conductor",
        backend="anthropic",
        model_name=args.teacher_model,
        description="SFT teacher",
        input_cost_per_mtok=2.00,
        output_cost_per_mtok=10.00,
        max_tokens=768,
        temperature=0.6,
    )

    orchestrator = RecursiveOrchestrator(
        worker_router=router,
        max_steps=pool_cfg.max_steps,
        max_depth=pool_cfg.max_depth if args.allow_recursion else 1,
        cost_aware=args.cost_aware,
    )

    workers = pool_cfg.active_workers(
        include_held_out=False,
        allow_recursion=args.allow_recursion,
    )
    sem = asyncio.Semaphore(args.task_concurrency)
    accepted: list[dict[str, Any]] = []
    stats = {"attempted": 0, "format_ok": 0, "verified_ok": 0, "total_cost_usd": 0.0}

    async def _process_one(idx: int, item: dict[str, Any]) -> None:
        topo_name, topo_hint = TOPOLOGY_HINTS[idx % len(TOPOLOGY_HINTS)]
        if topo_name == "recursive_check" and not args.allow_recursion:
            topo_name, topo_hint = TOPOLOGY_HINTS[1]

        view = build_episode_worker_view(
            workers,
            permute_ids=args.permute_ids,
            rng=random.Random(args.seed + idx),
        )

        async def _teacher_gen(messages: list[dict[str, str]]) -> str:
            guided = [dict(m) for m in messages]
            if len(guided) >= 2 and "Intermediate Outputs" not in guided[1]["content"]:
                guided[1]["content"] += f"\n\n(Orchestration style hint: {topo_hint})"
            resp = await anthropic_backend.generate(teacher_spec, guided)
            stats["total_cost_usd"] += resp.cost_usd
            return resp.text

        async with sem:
            stats["attempted"] += 1
            try:
                trace = await orchestrator.execute_episode(
                    task_prompt=item["prompt"],
                    task_type=item["task_type"],
                    worker_view=view,
                    conductor_generate_fn=_teacher_gen,
                )
            except Exception as exc:
                print(f"[warn] task {item['id']} failed: {exc}")
                return

            stats["total_cost_usd"] += trace.total_cost_usd
            if not trace.format_valid:
                return
            stats["format_ok"] += 1

            ok, _reason = verify_answer(
                item["task_type"],
                trace.final_answer,
                item["ground_truth"],
                metadata=item.get("metadata"),
            )
            if not ok and not args.keep_unverified:
                return
            if ok:
                stats["verified_ok"] += 1

            # Save every valid Conductor turn (both outer depth 0 and recursive inner turns)
            for turn in trace.conductor_turns:
                if turn.valid_format:
                    accepted.append(
                        {
                            "task_id": item["id"],
                            "task_type": item["task_type"],
                            "topology_hint": topo_name,
                            "depth": turn.depth,
                            "verified": ok,
                            "messages": turn.messages,
                            "completion": turn.completion_text,
                        }
                    )

    tasks = [_process_one(i, item) for i, item in enumerate(target_rows)]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Synthesizing SFT"):
        await coro

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in accepted:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    cache.close()
    print(
        f"[done] Saved {len(accepted)} Conductor turns to {out_path} | "
        f"attempted={stats['attempted']} format_ok={stats['format_ok']} "
        f"verified_ok={stats['verified_ok']} cost=${stats['total_cost_usd']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize verified SFT warmup trajectories.")
    parser.add_argument("--pool", type=str, default="configs/pool_claude_hybrid.yaml")
    parser.add_argument("--dataset", type=str, default="data/train.jsonl")
    parser.add_argument("--output", type=str, default="data/sft_warmup.jsonl")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--teacher_model", type=str, default="claude-sonnet-5-5")
    parser.add_argument("--teacher_fast_model", type=str, default="claude-haiku-5-5")
    parser.add_argument("--allow_recursion", action="store_true", default=True)
    parser.add_argument("--permute_ids", action="store_true", default=True)
    parser.add_argument("--cost_aware", action="store_true", default=False)
    parser.add_argument("--keep_unverified", action="store_true", default=False)
    parser.add_argument("--max_concurrency", type=int, default=16)
    parser.add_argument("--task_concurrency", type=int, default=8)
    parser.add_argument("--cache_db", type=str, default="cache/sft_workers.sqlite")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(synthesize_dataset(args))


if __name__ == "__main__":
    main()
