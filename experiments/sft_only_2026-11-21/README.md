Real-run outputs behind the "Results so far" section of the top-level README.

- Colab A100 (40GB). Conductor = Qwen2.5-7B-Instruct + LoRA SFT warm start (185 Haiku-written
  plans from 150 training tasks, 2 epochs). No RL.
- `*_eval_summary.json`: what `scripts/05_eval_suite.py` wrote, with the `eval_task_ids` list
  left out (it is the default `--limit 150 --seed 123` subset of `data/eval_all.jsonl`).
- `slot_analysis.txt`: per-worker, per-ID-slot usage computed from the per-episode
  `details_*.json` files (about 300KB each; kept with the experiment outputs, not in git).
