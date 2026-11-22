"""Does the Conductor call a worker because of WHO it is or WHICH ID it holds?

For each test-time condition, re-derives the per-task ID assignment used by `05_eval_suite.py`
and tabulates, per worker and per ID slot, how often the worker appears in the top-level plan.

    python scripts/07_slot_analysis.py --eval_dir $ROOT/eval/sft_only
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import build_episode_worker_view, load_pool_config


def _acc(rows: list[dict]) -> float:
    return round(sum(x["correct"] for x in rows) / max(len(rows), 1), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--pool", default="configs/pool_local_weak.yaml")
    ap.add_argument("--seed", type=int, default=123, help="Must match the eval run")
    ap.add_argument("--allow_recursion", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    pool = load_pool_config(args.pool)
    held_names = [w.name for w in pool.held_out_workers]
    for cond, held in [("permuted_ids", False), ("plus_heldoutplus_perm", True), ("plus_heldout", True)]:
        path = Path(args.eval_dir) / f"details_{cond}.json"
        if not path.exists():
            continue
        workers = pool.active_workers(include_held_out=held, allow_recursion=args.allow_recursion)
        det = json.loads(path.read_text(encoding="utf-8"))
        print("==", cond)
        for target in [w.name for w in workers if not w.is_self]:
            tab: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0])  # id -> [episodes, used, correct]
            for x in det:
                view = build_episode_worker_view(
                    workers, permute_ids="perm" in cond, rng=random.Random(f"{args.seed}:{x['task_id']}")
                )
                t = tab[view.name_to_id[target]]
                t[0] += 1
                t[1] += target in x["workers_used"]
                t[2] += x["correct"]
            print(f"  {target:13s}", {k: f"used {v[1]}/{v[0]} acc {v[2] / v[0]:.2f}" for k, v in sorted(tab.items())})
        for name in held_names if held else []:
            hk = [x for x in det if name in x["workers_used"]]
            fin = [x for x in hk if x["workers_used"][-1] == name]
            no = [x for x in det if name not in x["workers_used"] and x["format_valid"]]
            print(
                f"  episodes with {name}: {len(hk)} acc {_acc(hk)} | as last step: {len(fin)} acc {_acc(fin)} "
                f"| valid without: {len(no)} acc {_acc(no)}"
            )


if __name__ == "__main__":
    main()
