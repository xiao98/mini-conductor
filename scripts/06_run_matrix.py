"""One-command controlled experiment matrix for RQ1-RQ4 (train -> eval -> figures).

Every run shares the same SFT warm start, seed, data, pool, eval subset and hyperparameters;
each differs from `base` in exactly ONE switch:

  run                 differs from base by          answers
  base                (recursion, all_depth, lam=0) RQ1 curves; shared arm of RQ2/RQ3/RQ4
  rq2_no_recursion    no `self` worker              RQ2
  rq2_top_level       --credit_mode top_level       RQ2
  rq3_perm            --permute_ids                 RQ3
  rq4_lam0.05/.15/.30 --cost_lambda                 RQ4

Fully resumable: finished training runs / existing eval summaries are skipped, interrupted
training continues from `<root>/runs/<run>/latest`. Point `--root` at Google Drive on Colab.
Figures are drawn ONLY from files produced by real runs; missing runs are left out.

    python scripts/06_run_matrix.py --root /content/drive/MyDrive/mc_exp --init_adapter checkpoints/sft_warmup
    python scripts/06_run_matrix.py --root ... --only base,rq3_perm      # subset
    python scripts/06_run_matrix.py --root ... --figures_only            # re-draw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import plot_grouped_accuracy, plot_pareto_frontier, plot_topology_evolution

BASE = {"allow_recursion": True, "credit_mode": "all_depth", "permute_ids": False, "cost_lambda": 0.0}
RUNS: dict[str, dict[str, Any]] = {
    "base": {},
    "rq2_no_recursion": {"allow_recursion": False},
    "rq2_top_level": {"credit_mode": "top_level"},
    "rq3_perm": {"permute_ids": True},
    "rq4_lam0.05": {"cost_lambda": 0.05},
    "rq4_lam0.15": {"cost_lambda": 0.15},
    "rq4_lam0.30": {"cost_lambda": 0.30},
}


def run_spec(name: str) -> dict[str, Any]:
    return {**BASE, **RUNS[name]}


def train_cmd(name: str, args: argparse.Namespace) -> list[str]:
    spec = run_spec(name)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "04_train_grpo.py"),
        "--model", args.model, "--pool", args.pool, "--train_data", args.train_data,
        "--init_adapter", args.init_adapter,
        "--save_dir", str(Path(args.root) / "runs" / name),
        "--cache_db", str(Path(args.root) / "worker_cache.sqlite"),
        "--total_steps", str(args.total_steps), "--batch_size", str(args.batch_size),
        "--group_size", str(args.group_size), "--save_every", str(args.save_every),
        "--seed", str(args.seed), "--resume", "--max_new_tokens", str(args.max_new_tokens),
        "--gen_batch_size", str(args.gen_batch_size),
        "--credit_mode", spec["credit_mode"], "--cost_lambda", str(spec["cost_lambda"]),
    ]
    if spec["allow_recursion"]:
        cmd.append("--allow_recursion")
    if spec["permute_ids"]:
        cmd.append("--permute_ids")
    return cmd


def eval_cmd(name: str | None, args: argparse.Namespace) -> list[str]:
    """`name=None` -> baselines only (single workers + static topology)."""
    cmd = [
        sys.executable, str(ROOT / "scripts" / "05_eval_suite.py"),
        "--model", args.model, "--pool", args.pool, "--eval_data", args.eval_data,
        "--limit", str(args.eval_limit), "--seed", str(args.eval_seed),
        "--max_new_tokens", str(args.max_new_tokens), "--gen_batch_size", str(args.gen_batch_size),
        "--cache_db", str(Path(args.root) / "worker_cache.sqlite"),
        "--output_dir", str(Path(args.root) / "eval" / (name or "baselines")),
    ]
    if name is None:
        return cmd + ["--baselines", "--skip_conductor"]
    spec = run_spec(name)
    cmd += ["--adapter", str(Path(args.root) / "runs" / name / "latest"), "--label", name]
    cmd.append("--allow_recursion" if spec["allow_recursion"] else "--no-allow_recursion")
    if spec["cost_lambda"] > 0.0:
        cmd.append("--cost_aware")
    return cmd


def training_done(name: str, args: argparse.Namespace) -> bool:
    state = Path(args.root) / "runs" / name / "latest" / "trainer_state.json"
    return state.exists() and json.loads(state.read_text(encoding="utf-8"))["global_step"] >= args.total_steps


def _load_results(root: Path, name: str) -> list[dict[str, Any]]:
    path = root / "eval" / name / "eval_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else []


def _cell(results: list[dict[str, Any]], condition: str) -> dict[str, Any] | None:
    return next((r for r in results if r.get("condition") == condition and r["group"] == "conductor"), None)


def make_figures(root: Path) -> list[Path]:
    """Assemble RQ1-RQ4 figures from whatever real run outputs exist under `root`."""
    fig_dir = root / "figures"
    made: list[Path] = []
    evals = {name: _load_results(root, name) for name in RUNS}
    baselines = _load_results(root, "baselines")

    # RQ1: emergent topology over GRPO steps (training rollouts of the base run)
    metrics = root / "runs" / "base" / "train_metrics.jsonl"
    if metrics.exists() and metrics.read_text(encoding="utf-8").strip():
        made.append(plot_topology_evolution(metrics, fig_dir / "rq1_topology_evolution.png"))

    # RQ2: recursion / credit-assignment ablation, per task type
    rq2: dict[str, dict[str, dict[str, Any]]] = {}
    for label, name in [
        ("No-Recursion", "rq2_no_recursion"),
        ("Recursive + Top-Level GRPO", "rq2_top_level"),
        ("Recursive + All-Depth GRPO", "base"),
    ]:
        cell = _cell(evals[name], "canonical")
        if cell:
            rq2[label] = {"overall": cell, **cell["by_task_type"]}
    if rq2:
        made.append(plot_grouped_accuracy(rq2, "RQ2: Recursive Credit Assignment", fig_dir / "rq2_credit_assignment.png"))

    # RQ3: zero-shot worker transfer, canonical-ID training vs ID-Permutation Augmentation
    rq3: dict[str, dict[str, dict[str, Any]]] = {}
    for label, name in [("Canonical-ID training", "base"), ("ID-Permutation Augmentation", "rq3_perm")]:
        cells = {
            c: _cell(evals[name], c)
            for c in ("canonical", "permuted_ids", "+heldout", "+heldout+perm")
        }
        cells = {c: v for c, v in cells.items() if v}
        if cells:
            rq3[label] = cells
    if rq3:
        made.append(plot_grouped_accuracy(rq3, "RQ3: Zero-Shot Worker Transfer", fig_dir / "rq3_worker_transfer.png"))

    # RQ4: cost-accuracy Pareto frontier over cost_lambda vs single models / static topology
    points = list(baselines)
    for name in ["base", "rq4_lam0.05", "rq4_lam0.15", "rq4_lam0.30"]:
        cell = _cell(evals[name], "canonical")
        if cell:
            points.append({**cell, "label": f"lambda={run_spec(name)['cost_lambda']:g}"})
    if points:
        made.append(plot_pareto_frontier(points, fig_dir / "rq4_pareto_frontier.png"))

    table = {"baselines": baselines, **{k: v for k, v in evals.items() if v}}
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / "all_results.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    return made


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run the RQ1-RQ4 experiment matrix.")
    ap.add_argument("--root", required=True, help="Experiment root (Google Drive on Colab)")
    ap.add_argument("--only", default="", help="Comma-separated subset of: " + ",".join(RUNS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--pool", default="configs/pool_local_weak.yaml")
    ap.add_argument("--train_data", default="data/train.jsonl")
    ap.add_argument("--eval_data", default="data/eval_all.jsonl")
    ap.add_argument("--init_adapter", default="checkpoints/sft_warmup")
    ap.add_argument("--total_steps", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--gen_batch_size", type=int, default=16, help="Max prompts per GPU generate() call")
    ap.add_argument("--eval_limit", type=int, default=150)
    ap.add_argument("--eval_seed", type=int, default=123)
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--figures_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Print the commands without running them")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.root)
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(RUNS)
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        raise SystemExit(f"Unknown run(s): {unknown}. Choose from {list(RUNS)}")

    def _run(cmd: list[str]) -> None:
        print("\n$ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)

    if not args.figures_only:
        if not args.skip_baselines and not (root / "eval" / "baselines" / "eval_summary.json").exists():
            _run(eval_cmd(None, args))
        for name in names:
            if not training_done(name, args):
                _run(train_cmd(name, args))
            if not (root / "eval" / name / "eval_summary.json").exists():
                _run(eval_cmd(name, args))

    if not args.dry_run:
        for path in make_figures(root):
            print(f"[figure] {path}")


if __name__ == "__main__":
    main()
