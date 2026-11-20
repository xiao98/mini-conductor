"""Topology analytics and publication-ready plotting for RQ1–RQ4."""

from __future__ import annotations

from collections import Counter
import json
import math
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


def accuracy_stats(flags: list[bool], task_types: list[str]) -> dict[str, Any]:
    """Accuracy with a 95% normal-approximation CI half-width, overall and per task type."""

    def _one(xs: list[bool]) -> dict[str, Any]:
        n = len(xs)
        acc = sum(xs) / max(n, 1)
        return {
            "accuracy": round(acc, 4),
            "acc_ci95": round(1.96 * math.sqrt(acc * (1.0 - acc) / max(n, 1)), 4),
            "n": n,
        }

    out = _one(flags)
    out["by_task_type"] = {
        t: _one([f for f, tt in zip(flags, task_types) if tt == t])
        for t in sorted(set(task_types))
    }
    return out


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

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), dpi=160)

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

    worker_names = sorted({w for r in rows for w in r.get("worker_usage_share", {})})
    for w in worker_names:
        axes[3].plot(
            steps,
            [r.get("worker_usage_share", {}).get(w, 0.0) * 100.0 for r in rows],
            linewidth=1.8,
            label=w,
        )
    axes[3].set_title("Worker Call Share")
    axes[3].set_xlabel("GRPO Step")
    axes[3].set_ylabel("Share of Worker Calls (%)")
    axes[3].grid(alpha=0.25)
    if worker_names:
        axes[3].legend(fontsize=8)

    fig.suptitle("Training-rollout statistics (sampled at train temperature, train tasks)", fontsize=9)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_grouped_accuracy(
    groups: dict[str, dict[str, dict[str, Any]]],
    title: str,
    output_path: str | Path,
) -> Path:
    """Grouped bars with 95% CI. `groups[series][x_label] = {"accuracy", "acc_ci95", ...}`.

    Used for RQ2 (series = training variant, x = task type) and RQ3 (series = training
    variant, x = test-time pool condition). Missing cells are simply not drawn.
    """
    import matplotlib.pyplot as plt

    series = list(groups.keys())
    x_labels: list[str] = []
    for cells in groups.values():
        x_labels += [x for x in cells if x not in x_labels]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(x_labels) + 3.0), 4.6), dpi=160)
    width = 0.8 / max(len(series), 1)
    for si, name in enumerate(series):
        for xi, xl in enumerate(x_labels):
            cell = groups[name].get(xl)
            if cell is None:
                continue
            ax.bar(
                xi + (si - (len(series) - 1) / 2) * width,
                cell["accuracy"] * 100.0,
                width,
                yerr=cell.get("acc_ci95", 0.0) * 100.0,
                capsize=3,
                color=f"C{si}",
                label=name if xi == 0 else None,
            )
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), fontsize=8)
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

    statics = [p for p in points if p.get("group") == "static"]
    if statics:
        ax.scatter(
            [p["avg_cost_usd"] * 1000.0 for p in statics],
            [p["accuracy"] * 100.0 for p in statics],
            s=90, marker="s", color="#d97706", label="Static Topologies", zorder=3,
        )
        for p in statics:
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
