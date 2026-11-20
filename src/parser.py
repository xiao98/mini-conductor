"""Safe parser and strict DAG validator for Conductor 3-list orchestration plans."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class OrchestrationPlan:
    """Validated DAG orchestration plan emitted by the Conductor."""

    model_ids: list[int]
    subtasks: list[str]
    access_lists: list[list[int]]
    think_text: str
    raw_text: str

    @property
    def num_steps(self) -> int:
        return len(self.model_ids)


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing and validating a Conductor completion."""

    plan: OrchestrationPlan | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.error is None


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


def _extract_assignments(block: str) -> dict[str, object]:
    """Safely parse `model_id = ...`, `subtasks = ...`, `access_list = ...` via Python AST."""
    cleaned = block.strip()
    # Strip optional markdown code fences inside <plan>
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    tree = ast.parse(cleaned, mode="exec")
    found: dict[str, object] = {}
    allowed_names = {"model_id", "model_ids", "subtasks", "access_list", "access_lists"}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in allowed_names:
            found[target.id] = ast.literal_eval(node.value)
    return found


def parse_and_validate_plan(
    raw_text: str,
    *,
    valid_worker_ids: set[int],
    max_steps: int = 5,
    disallowed_ids: set[int] | None = None,
) -> ParseResult:
    """Parse Conductor output into a validated, strictly acyclic OrchestrationPlan."""
    think_match = _THINK_RE.search(raw_text)
    think_text = think_match.group(1).strip() if think_match else ""

    plan_match = _PLAN_RE.search(raw_text)
    if plan_match:
        plan_body = plan_match.group(1)
    else:
        # Fallback: attempt parsing code block or tail containing the three assignments
        if "model_id" not in raw_text or "subtasks" not in raw_text or "access_list" not in raw_text:
            return ParseResult(None, "Missing <plan> block or required list assignments.")
        start = raw_text.find("model_id")
        plan_body = raw_text[start:]

    try:
        assignments = _extract_assignments(plan_body)
    except Exception as exc:
        return ParseResult(None, f"AST parse failed: {exc}")

    raw_model_ids = assignments.get("model_id", assignments.get("model_ids"))
    raw_subtasks = assignments.get("subtasks")
    raw_access = assignments.get("access_list", assignments.get("access_lists"))

    if not isinstance(raw_model_ids, list) or not isinstance(raw_subtasks, list) or not isinstance(raw_access, list):
        return ParseResult(None, "model_id, subtasks, and access_list must all be Python lists.")

    n = len(raw_model_ids)
    if n == 0:
        return ParseResult(None, "Plan must contain at least 1 step.")
    if n > max_steps:
        return ParseResult(None, f"Plan exceeds max_steps ({n} > {max_steps}).")
    if len(raw_subtasks) != n or len(raw_access) != n:
        return ParseResult(
            None,
            f"List length mismatch: model_id={len(raw_model_ids)}, subtasks={len(raw_subtasks)}, access_list={len(raw_access)}.",
        )

    blocked = disallowed_ids or set()
    model_ids: list[int] = []
    for idx, mid in enumerate(raw_model_ids):
        if isinstance(mid, bool) or not isinstance(mid, int):
            return ParseResult(None, f"Step {idx} model_id must be an int, got {type(mid).__name__}.")
        if mid not in valid_worker_ids:
            return ParseResult(None, f"Step {idx} model_id={mid} not in valid IDs {sorted(valid_worker_ids)}.")
        if mid in blocked:
            return ParseResult(None, f"Step {idx} model_id={mid} is disallowed at this depth.")
        model_ids.append(mid)

    subtasks: list[str] = []
    for idx, task in enumerate(raw_subtasks):
        if not isinstance(task, str) or not task.strip():
            return ParseResult(None, f"Step {idx} subtask must be a non-empty string.")
        subtasks.append(task.strip())

    access_lists: list[list[int]] = []
    for idx, acc in enumerate(raw_access):
        if isinstance(acc, str) and acc.strip().lower() == "all":
            access_lists.append(list(range(idx)))
            continue
        if not isinstance(acc, list):
            return ParseResult(None, f"Step {idx} access_list entry must be a list or 'all'.")
        step_deps: list[int] = []
        for dep in acc:
            if isinstance(dep, str) and dep.strip().lower() == "all":
                step_deps = list(range(idx))
                break
            if isinstance(dep, bool) or not isinstance(dep, int):
                return ParseResult(None, f"Step {idx} access index must be int, got {dep!r}.")
            if dep < 0 or dep >= idx:
                return ParseResult(
                    None,
                    f"Causal DAG violation at step {idx}: dependency {dep} must satisfy 0 <= dep < {idx}.",
                )
            if dep not in step_deps:
                step_deps.append(dep)
        access_lists.append(sorted(step_deps))

    return ParseResult(
        plan=OrchestrationPlan(
            model_ids=model_ids,
            subtasks=subtasks,
            access_lists=access_lists,
            think_text=think_text,
            raw_text=raw_text,
        )
    )


def compute_dag_layers(access_lists: list[list[int]]) -> list[list[int]]:
    """Group DAG steps into parallel wavefront layers using topological depth.

    Every step in layer `l` depends only on steps in layers `< l`, so all steps in the
    same layer can execute concurrently via `asyncio.gather`.
    """
    n = len(access_lists)
    if n == 0:
        return []

    depth_of: list[int] = [0] * n
    for i, deps in enumerate(access_lists):
        if deps:
            depth_of[i] = 1 + max(depth_of[d] for d in deps)
        else:
            depth_of[i] = 0

    max_layer = max(depth_of)
    layers: list[list[int]] = [[] for _ in range(max_layer + 1)]
    for i, layer_idx in enumerate(depth_of):
        layers[layer_idx].append(i)
    return layers
