"""Prompt templates for the Conductor (outer + recursive levels) and Worker models."""

from __future__ import annotations

from src.config import EpisodeWorkerView


TASK_FORMAT_GUIDANCE: dict[str, str] = {
    "math": "Ensure the FINAL step puts the final mathematical answer inside \\boxed{...}.",
    "code": "Ensure the FINAL step outputs clean, complete, executable Python code inside a ```python ... ``` block.",
    "gpqa": "Ensure the FINAL step ends with the single chosen option letter in the format `Answer: X` (where X is A, B, C, or D).",
}


def format_worker_catalog(
    worker_view: EpisodeWorkerView,
    *,
    allow_self: bool,
) -> str:
    """Format the active worker pool catalog shown to the Conductor."""
    lines: list[str] = []
    for spec in worker_view.ordered_specs:
        if spec.is_self and not allow_self:
            continue
        lines.append(
            f"- ID {spec.id} ({spec.name} | cost: {spec.cost_tier}): {spec.description}"
        )
    return "\n".join(lines)


def build_conductor_messages(
    *,
    task_prompt: str,
    task_type: str,
    worker_view: EpisodeWorkerView,
    max_steps: int,
    depth: int,
    max_depth: int,
    parent_subtask: str | None = None,
    prior_outputs: dict[int, tuple[str, str]] | None = None,
    cost_aware: bool = False,
) -> list[dict[str, str]]:
    """Build chat messages for the Conductor at either top level (depth=0) or recursive level."""
    allow_self = (worker_view.self_id is not None) and (depth + 1 < max_depth)
    catalog = format_worker_catalog(worker_view, allow_self=allow_self)
    format_hint = TASK_FORMAT_GUIDANCE.get(
        task_type,
        "Ensure the FINAL step produces a clear, complete final answer.",
    )
    cost_line = (
        "- Be cost-aware: use cheap/local workers for easy subproblems and reserve expensive workers for hard reasoning or verification.\n"
        if cost_aware
        else ""
    )
    self_line = (
        f"- Recursive orchestration is ENABLED (current depth {depth}/{max_depth - 1}): "
        f"you may assign ID {worker_view.self_id} (self) to inspect previous workers' outputs and dynamically create a follow-up sub-plan.\n"
        if allow_self
        else "- Recursive self-calls are NOT available at this depth; select only concrete worker IDs.\n"
    )

    system_prompt = (
        "You are the Conductor, a multi-agent orchestrator trained to solve complex tasks by "
        "dynamically designing communication topologies and tailored natural-language subtasks "
        "for a pool of specialized worker models.\n\n"
        "### Available Workers\n"
        f"{catalog}\n\n"
        "### Orchestration Rules\n"
        f"- Output between 1 and {max_steps} steps.\n"
        "- Do NOT solve the entire problem yourself; instead delegate focused subtasks to workers.\n"
        "- Control information flow via `access_list`: use `[]` for independent parallel execution, "
        "specific prior step indices like `[0, 1]` to compare/verify/synthesize, or `\"all\"` for all prior steps.\n"
        "- Every index j in `access_list[i]` MUST satisfy `0 <= j < i` (strict acyclic DAG).\n"
        f"{self_line}"
        f"{cost_line}"
        f"- {format_hint}\n\n"
        "### Required Output Format\n"
        "First explain your orchestration strategy briefly inside `<think>...</think>`. "
        "Then output EXACTLY three aligned Python lists inside `<plan>...</plan>`:\n"
        "<think>\n"
        "Brief reasoning on problem difficulty, worker selection, and topology.\n"
        "</think>\n"
        "<plan>\n"
        "model_id = [1, 2]\n"
        "subtasks = [\n"
        "    \"Clear instruction for step 0 worker.\",\n"
        "    \"Clear instruction for step 1 worker using step 0 output.\"\n"
        "]\n"
        "access_list = [[], [0]]\n"
        "</plan>"
    )

    if depth == 0:
        user_content = f"### Task ({task_type})\n{task_prompt}"
    else:
        context_blocks: list[str] = []
        if prior_outputs:
            for step_idx in sorted(prior_outputs.keys()):
                w_name, text = prior_outputs[step_idx]
                context_blocks.append(f"[Parent Step {step_idx} — {w_name}]\n{text}")
        joined_context = "\n\n".join(context_blocks) if context_blocks else "(None)"
        user_content = (
            f"### Original Task ({task_type})\n{task_prompt}\n\n"
            f"### Intermediate Outputs from Parent Plan\n{joined_context}\n\n"
            f"### Delegated Re-Orchestration Goal\n{parent_subtask or 'Resolve any errors or disagreements and produce the verified final answer.'}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_worker_messages(
    *,
    task_prompt: str,
    task_type: str,
    subtask: str,
    visible_outputs: list[tuple[int, str, str]],
    is_final_step: bool,
) -> list[dict[str, str]]:
    """Build chat messages for a concrete worker executing one DAG step."""
    system_prompt = (
        "You are a specialist worker inside a multi-agent reasoning team. "
        "Follow the assigned subtask instruction carefully and accurately."
    )
    if is_final_step:
        final_rule = TASK_FORMAT_GUIDANCE.get(task_type, "")
        if final_rule:
            system_prompt += f" This is the FINAL step of the team's workflow. {final_rule}"

    sections: list[str] = [f"### Original Problem\n{task_prompt}"]
    if visible_outputs:
        ctx = "\n\n".join(
            f"--- Output from Step {idx} ({w_name}) ---\n{text}"
            for idx, w_name, text in visible_outputs
        )
        sections.append(f"### Context from Prior Steps\n{ctx}")
    sections.append(f"### Your Assigned Subtask\n{subtask}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
