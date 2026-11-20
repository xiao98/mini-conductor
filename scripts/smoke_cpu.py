"""CPU plumbing smoke test: tiny chat model + mock workers through 03 -> 04 (+resume) -> 05 -> 06.

Verifies that the real HF/PEFT code paths run end to end (LoRA SFT, batched generation,
GRPO update with disable_adapter() reference, atomic checkpoint + resume, eval, figures)
WITHOUT a GPU or API key. Everything it writes lives under `--root` (default `smoke_out/`).

*** Nothing produced here is an experimental result: the model is a 135M toy, the workers
*** are mocks, and the SFT data is templated. It only proves the pipeline executes.

    python scripts/smoke_cpu.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import build_episode_worker_view, load_pool_config
from src.prompts import build_conductor_messages

POOL = "configs/pool_smoke_mock.yaml"
RUN_NAMES = ["base", "rq2_no_recursion", "rq4_lam0.05"]
TEMPLATE_PLANS = [
    "<think>Easy task: route to one worker.</think>\n<plan>\nmodel_id = [0]\nsubtasks = [\"Solve the task.\"]\naccess_list = [[]]\n</plan>",
    "<think>Two solvers, then a judge.</think>\n<plan>\nmodel_id = [0, 1, 1]\nsubtasks = [\"Solve it.\", \"Solve it differently.\", \"Compare and answer.\"]\naccess_list = [[], [], [0, 1]]\n</plan>",
    "<think>Draft, then check.</think>\n<plan>\nmodel_id = [0, 1]\nsubtasks = [\"Draft a solution.\", \"Check the draft and answer.\"]\naccess_list = [[], [0]]\n</plan>",
    "<think>Attempt, then re-plan recursively.</think>\n<plan>\nmodel_id = [0, 2]\nsubtasks = [\"Attempt the task.\", \"Inspect the attempt and finish.\"]\naccess_list = [[], [0]]\n</plan>",
]


def write_templated_sft(demo_rows: list[dict], out_path: Path) -> None:
    pool = load_pool_config(ROOT / POOL)
    view = build_episode_worker_view(pool.active_workers())
    with open(out_path, "w", encoding="utf-8") as f:
        for row in demo_rows:
            for plan in TEMPLATE_PLANS:
                msgs = build_conductor_messages(
                    task_prompt=row["prompt"], task_type=row["task_type"], worker_view=view,
                    max_steps=pool.max_steps, depth=0, max_depth=pool.max_depth,
                )
                f.write(json.dumps({"messages": msgs, "completion": plan, "synthetic_smoke": True}) + "\n")


def sh(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="smoke_out")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--sft_epochs", type=int, default=6)
    args = ap.parse_args()

    root = (ROOT / args.root).resolve()
    shutil.rmtree(root, ignore_errors=True)
    data = root / "data"
    py = sys.executable

    sh([py, "scripts/01_build_dataset.py", "--demo", "--output_dir", str(data)])
    demo = [json.loads(l) for l in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    write_templated_sft(demo[::2], data / "sft_smoke.jsonl")  # 3 tasks x 4 plans keeps CPU time low
    sh([py, "scripts/03_train_sft.py", "--model", args.model, "--data", str(data / "sft_smoke.jsonl"),
        "--output_dir", str(root / "sft"), "--epochs", str(args.sft_epochs), "--lr", "1e-3",
        "--grad_accum", "4", "--lora_r", "8", "--lora_alpha", "16"])

    def matrix(total_steps: int) -> None:
        sh([py, "scripts/06_run_matrix.py", "--root", str(root), "--model", args.model, "--pool", POOL,
            "--train_data", str(data / "train.jsonl"), "--eval_data", str(data / "eval_all.jsonl"),
            "--init_adapter", str(root / "sft"), "--only", ",".join(RUN_NAMES),
            "--total_steps", str(total_steps), "--batch_size", "2", "--group_size", "4",
            "--save_every", "1", "--max_new_tokens", "96", "--eval_limit", "3"])

    # Train 1 step, wipe evals, then extend to 2 steps: exercises --resume from `latest`.
    matrix(1)
    shutil.rmtree(root / "eval")
    matrix(2)

    # ---- Assertions on what was really produced ----
    for run in RUN_NAMES:
        rows = [json.loads(l) for l in (root / "runs" / run / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["step"] for r in rows] == [1, 2], f"{run}: resume produced steps {[r['step'] for r in rows]}"
        state = json.loads((root / "runs" / run / "latest" / "trainer_state.json").read_text(encoding="utf-8"))
        assert state["global_step"] == 2
        assert (root / "eval" / run / "eval_summary.json").exists()
        print(f"[smoke] {run}: fmt={[r['format_rate'] for r in rows]} turns_updated={[int(r['num_turns_updated']) for r in rows]} "
              f"rec={[r['recursion_rate'] for r in rows]}")
    norec = [json.loads(l) for l in (root / "runs" / "rq2_no_recursion" / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["recursion_rate"] == 0.0 for r in norec), "No-Recursion run must never recurse"
    for fig in ["rq1_topology_evolution", "rq2_credit_assignment", "rq3_worker_transfer", "rq4_pareto_frontier"]:
        assert (root / "figures" / f"{fig}.png").exists(), fig
    print(f"\n[smoke] PASS - pipeline executes end to end. Outputs under {root} are NOT results.")


if __name__ == "__main__":
    main()
