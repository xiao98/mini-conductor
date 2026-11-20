"""Async layer-wise DAG executor with bounded recursive self-orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from src.config import EpisodeWorkerView
from src.parser import OrchestrationPlan, compute_dag_layers, parse_and_validate_plan
from src.prompts import build_conductor_messages, build_worker_messages
from src.workers import WorkerResponse, WorkerRouter


# Async callable taking chat messages and returning raw Conductor completion text.
ConductorGenerateFn = Callable[[list[dict[str, str]]], Awaitable[str]]


@dataclass
class ConductorTurnRecord:
    """Single Conductor LLM generation inside a (potentially recursive) rollout.

    Recording every turn allows GRPO to train either on `turns[:1]` (top_level)
    or on all `turns` (all_depth recursive credit assignment).
    """

    depth: int
    parent_step_idx: int | None
    messages: list[dict[str, str]]
    completion_text: str
    plan: OrchestrationPlan | None
    parse_error: str | None
    id_to_worker_name: dict[int, str]

    @property
    def valid_format(self) -> bool:
        return self.plan is not None and self.parse_error is None


@dataclass
class StepExecutionRecord:
    """Execution result of one node inside a DAG plan."""

    step_idx: int
    depth: int
    worker_id: int
    worker_name: str
    is_recursive_self: bool
    subtask: str
    access_indices: list[int]
    output_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    child_trace: RolloutTrace | None = None


@dataclass
class RolloutTrace:
    """Complete execution trace for one Conductor episode (or recursive sub-episode)."""

    task_prompt: str
    task_type: str
    depth: int
    format_valid: bool
    final_answer: str
    parse_error: str | None = None
    conductor_turns: list[ConductorTurnRecord] = field(default_factory=list)
    steps: list[StepExecutionRecord] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    max_depth_reached: int = 0
    num_layers: int = 0
    max_parallel_width: int = 0
    recursion_triggered: bool = False


class RecursiveOrchestrator:
    """Executes Conductor DAG plans asynchronously with bounded self-recursion."""

    def __init__(
        self,
        *,
        worker_router: WorkerRouter,
        max_steps: int = 5,
        max_depth: int = 2,
        cost_aware: bool = False,
    ) -> None:
        self.worker_router = worker_router
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.cost_aware = cost_aware

    async def execute_episode(
        self,
        *,
        task_prompt: str,
        task_type: str,
        worker_view: EpisodeWorkerView,
        conductor_generate_fn: ConductorGenerateFn,
        depth: int = 0,
        parent_step_idx: int | None = None,
        parent_subtask: str | None = None,
        prior_outputs: dict[int, tuple[str, str]] | None = None,
        prefilled_completion: str | None = None,
    ) -> RolloutTrace:
        """Run one (possibly recursive) orchestration episode and collect all Conductor turns."""
        messages = build_conductor_messages(
            task_prompt=task_prompt,
            task_type=task_type,
            worker_view=worker_view,
            max_steps=self.max_steps,
            depth=depth,
            max_depth=self.max_depth,
            parent_subtask=parent_subtask,
            prior_outputs=prior_outputs,
            cost_aware=self.cost_aware,
        )

        if prefilled_completion is not None and depth == 0:
            completion_text = prefilled_completion
        else:
            completion_text = await conductor_generate_fn(messages)

        disallowed: set[int] = set()
        if worker_view.self_id is not None and (depth + 1 >= self.max_depth):
            disallowed.add(worker_view.self_id)

        parsed = parse_and_validate_plan(
            completion_text,
            valid_worker_ids=worker_view.valid_ids,
            max_steps=self.max_steps,
            disallowed_ids=disallowed,
        )

        turn_record = ConductorTurnRecord(
            depth=depth,
            parent_step_idx=parent_step_idx,
            messages=messages,
            completion_text=completion_text,
            plan=parsed.plan,
            parse_error=parsed.error,
            id_to_worker_name={wid: w.name for wid, w in worker_view.id_to_worker.items()},
        )

        if not parsed.ok or parsed.plan is None:
            return RolloutTrace(
                task_prompt=task_prompt,
                task_type=task_type,
                depth=depth,
                format_valid=False,
                final_answer="",
                parse_error=parsed.error,
                conductor_turns=[turn_record],
                max_depth_reached=depth,
            )

        plan = parsed.plan
        layers = compute_dag_layers(plan.access_lists)
        step_records: dict[int, StepExecutionRecord] = {}
        all_turns: list[ConductorTurnRecord] = [turn_record]

        total_cost = 0.0
        total_in = 0
        total_out = 0
        max_depth_reached = depth
        recursion_triggered = False
        inner_format_ok = True

        async def _run_step(step_idx: int) -> StepExecutionRecord:
            wid = plan.model_ids[step_idx]
            spec = worker_view.id_to_worker[wid]
            subtask = plan.subtasks[step_idx]
            deps = plan.access_lists[step_idx]
            is_final = step_idx == (plan.num_steps - 1)

            visible = [
                (d, step_records[d].worker_name, step_records[d].output_text)
                for d in deps
            ]

            if spec.is_self:
                child_prior = {
                    d: (step_records[d].worker_name, step_records[d].output_text)
                    for d in deps
                }
                child = await self.execute_episode(
                    task_prompt=task_prompt,
                    task_type=task_type,
                    worker_view=worker_view,
                    conductor_generate_fn=conductor_generate_fn,
                    depth=depth + 1,
                    parent_step_idx=step_idx,
                    parent_subtask=subtask,
                    prior_outputs=child_prior,
                )
                return StepExecutionRecord(
                    step_idx=step_idx,
                    depth=depth,
                    worker_id=wid,
                    worker_name=spec.name,
                    is_recursive_self=True,
                    subtask=subtask,
                    access_indices=deps,
                    output_text=child.final_answer,
                    input_tokens=child.total_input_tokens,
                    output_tokens=child.total_output_tokens,
                    cost_usd=child.total_cost_usd,
                    child_trace=child,
                )

            worker_msgs = build_worker_messages(
                task_prompt=task_prompt,
                task_type=task_type,
                subtask=subtask,
                visible_outputs=visible,
                is_final_step=is_final,
            )
            resp: WorkerResponse = await self.worker_router.call(spec, worker_msgs)
            return StepExecutionRecord(
                step_idx=step_idx,
                depth=depth,
                worker_id=wid,
                worker_name=spec.name,
                is_recursive_self=False,
                subtask=subtask,
                access_indices=deps,
                output_text=resp.text,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=resp.cost_usd,
                latency_s=resp.latency_s,
            )

        for layer in layers:
            layer_results = await asyncio.gather(*(_run_step(idx) for idx in layer))
            for rec in layer_results:
                step_records[rec.step_idx] = rec
                total_cost += rec.cost_usd
                total_in += rec.input_tokens
                total_out += rec.output_tokens
                if rec.child_trace is not None:
                    recursion_triggered = True
                    max_depth_reached = max(max_depth_reached, rec.child_trace.max_depth_reached)
                    all_turns.extend(rec.child_trace.conductor_turns)
                    if not rec.child_trace.format_valid:
                        inner_format_ok = False

        ordered_steps = [step_records[i] for i in range(plan.num_steps)]
        final_answer = ordered_steps[-1].output_text

        return RolloutTrace(
            task_prompt=task_prompt,
            task_type=task_type,
            depth=depth,
            format_valid=inner_format_ok,
            final_answer=final_answer,
            parse_error=None if inner_format_ok else "Inner recursive plan failed format validation.",
            conductor_turns=all_turns,
            steps=ordered_steps,
            total_cost_usd=total_cost,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            max_depth_reached=max_depth_reached,
            num_layers=len(layers),
            max_parallel_width=max(len(layer) for layer in layers),
            recursion_triggered=recursion_triggered,
        )
