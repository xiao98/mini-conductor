"""Topology analytics and publication-ready plotting for RQ1–RQ4."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from src.executor import RolloutTrace


def summarize_traces(
    traces: list[RolloutTrace],
    correctness: list[bool],
    rewards: list[float] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, cost, format compliance, and emergent topology statistics."""
    n = len(traces)
    if n == 0:
        return {}

    valid_traces = [t for t in traces if t.format_valid]
    worker_counts: Counter[str] = Counter()
    total_worker_calls = 0

    def _count_steps(tr: RolloutTrace) -> None:
        nonlocal total_worker_calls
        for st in tr.steps:
            worker_counts[st.worker_name] += 1
            total_worker_calls += 1
            if st.child_trace is not None:
                _count_steps(st.child_trace)

    for tr in valid_traces:
        _count_steps(tr)

    worker_share = {
        name: round(cnt / max(total_worker_calls, 1), 4)
        for name, cnt in worker_counts.most_common()
    }

    n_valid = max(len(valid_traces), 1)
    summary: dict[str, Any] = {
        "num_episodes": n,
        "format_rate": round(len(valid_traces) / n, 4),
        "accuracy": round(sum(1 for c in correctness if c) / n, 4),
        "avg_cost_usd": round(sum(t.total_cost_usd for t in traces) / n, 6),
        "avg_top_steps": round(sum(len(t.steps) for t in valid_traces) / n_valid, 3),
        "avg_num_layers": round(sum(t.num_layers for t in valid_traces) / n_valid, 3),
        "avg_parallel_width": round(sum(t.max_parallel_width for t in valid_traces) / n_valid, 3),
        "recursion_rate": round(sum(1 for t in valid_traces if t.recursion_triggered) / n_valid, 4),
        "avg_max_depth": round(sum(t.max_depth_reached for t in valid_traces) / n_valid, 3),
        "worker_usage_share": worker_share,
    }
    if rewards is not None and len(rewards) == n:
        summary["avg_reward"] = round(sum(rewards) / n, 4)
    return summary


def plot_topology_evolution(
    metrics_jsonl: str | Path,
    output_path: str | Path,
) -> Path:
    """RQ1 plot: Accuracy, reward, parallelism, and recursion rate over GRPO steps."""
    import matplotlib.pyplot as plt

    rows = [
        json.loads(line)
        for line in Path(metrics_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    steps = [r["step"] for r in rows]
    acc = [r.get("accuracy", 0.0) * 100.0 for r in rows]
    fmt = [r.get("format_rate", 0.0) * 100.0 for r in rows]
    par = [r.get("avg_parallel_width", 1.0) for r in rows]
    rec = [r.get("recursion_rate", 0.0) * 100.0 for r in rows]
    cost = [r.get("avg_cost_usd", 0.0) * 1000.0 for r in rows]  # in millidollars

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=160)

    axes[0].plot(steps, acc, marker="o", linewidth=2, label="Accuracy (%)")
    axes[0].plot(steps, fmt, linestyle="--", linewidth=1.6, label="Valid DAG (%)")
    axes[0].set_title("RQ1: Accuracy & Format Compliance")
    axes[0].set_xlabel("GRPO Step")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, par, marker="s", linewidth=2, color="#2563eb", label="Avg Max Parallel Width")
    ax1_r = axes[1].twinx()
    ax1_r.plot(steps, rec, marker="^", linewidth=2, color="#dc2626", label="Recursion Rate (%)")
    axes[1].set_title("Emergent Topology Dynamics")
    axes[1].set_xlabel("GRPO Step")
    axes[1].set_ylabel("Parallel Width", color="#2563eb")
    ax1_r.set_ylabel("Recursion Rate (%)", color="#dc2626")
    axes[1].grid(alpha=0.25)

    axes[2].plot(steps, cost, marker="d", linewidth=2, color="#059669")
    axes[2].set_title("Orchestration Cost Trajectory")
    axes[2].set_xlabel("GRPO Step")
    axes[2].set_ylabel("Avg Cost ($ / 1k queries)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pareto_frontier(
    points: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """RQ4 plot: Cost-Accuracy Pareto Frontier (Single Models vs. mini-conductor sweep)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=160)

    singles = [p for p in points if p.get("group") == "single_model"]
    conductors = sorted(
        [p for p in points if p.get("group") == "conductor"],
        key=lambda x: x["avg_cost_usd"],
    )

    if singles:
        sx = [p["avg_cost_usd"] * 1000.0 for p in singles]
        sy = [p["accuracy"] * 100.0 for p in singles]
        ax.scatter(sx, sy, s=90, marker="X", color="#64748b", label="Single Models", zorder=3)
        for p in singles:
            ax.annotate(
                p["label"],
                (p["avg_cost_usd"] * 1000.0, p["accuracy"] * 100.0),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )

    if conductors:
        cx = [p["avg_cost_usd"] * 1000.0 for p in conductors]
        cy = [p["accuracy"] * 100.0 for p in conductors]
        ax.plot(cx, cy, marker="o", markersize=8, linewidth=2.4, color="#2563eb", label="mini-conductor (GRPO)", zorder=4)
        for p in conductors:
            ax.annotate(
                p["label"],
                (p["avg_cost_usd"] * 1000.0, p["accuracy"] * 100.0),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=9,
                color="#1d4ed8",
            )

    ax.set_title("RQ4: Cost–Accuracy Pareto Frontier")
    ax.set_xlabel("Average API Cost per 1,000 Queries (USD)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
