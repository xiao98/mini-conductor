# `mini-conductor`

**A minimal, transparent, single-GPU PyTorch reproduction & Agent RL research suite for Sakana AI's [*Learning to Orchestrate Agents in Natural Language with the Conductor*](https://arxiv.org/abs/2512.04388) (ICLR 2026).**

> Built for **Colab H100/A100 (80GB/40GB) + Claude API** (with an optional 100% open-source local specialist pool). No heavy distributed Ray/FSDP stack required — clean `PyTorch + transformers + peft + asyncio` code you can read end-to-end in an afternoon.

---

## Why `mini-conductor`?

Sakana AI commercialized *The Conductor* into **Sakana Fugu** (`arXiv:2606.21228`), so the official RL training pipeline is not open-sourced. This repository provides:

1. **Faithful Core Reproduction**:
   - Chain-of-thought (`<think>`) followed by structured 3-list DAG plans (`model_id`, `subtasks`, `access_list`).
   - True layer-wise parallel execution via `asyncio.gather` (Kahn's topological wavefronts).
   - **Recursive self-referential test-time scaling**: the Conductor can invoke `self` as a worker to inspect intermediate worker outputs and dynamically spawn inner sub-DAGs.
   - End-to-end **GRPO** with format + deterministic verification rewards (MATH / LiveCodeBench / GPQA) and zero-extra-VRAM reference KL via `model.disable_adapter()`.

2. **Four Targeted Agent RL / RSI Studies**:
   - **RQ1 — Emergent Topology Evolution**: Track how parallelism, chain depth, worker routing, and recursion rate evolve across RL steps on Math, Code, and GPQA.
   - **RQ2 — Recursive Credit Assignment (`top_level` vs. `all_depth` GRPO)**: Controlled ablation comparing top-level-only gradient updates against full recursive trajectory credit assignment (masking external worker tokens).
   - **RQ3 — Zero-Shot Worker Transfer & `ID-Permutation Augmentation`**: Train on `{Local-7B, Claude Haiku, Claude Sonnet, self}`, then test zero-shot with held-out `Claude Opus` or swapped IDs — measuring and fixing Worker-ID overfitting.
   - **RQ4 — Cost–Accuracy Pareto Frontier**: Sweep cost penalty $\lambda$ to map dynamic orchestration against single frontier models and static routers.

---

## Project Structure

```text
mini-conductor/
├── configs/
│   ├── pool_claude_hybrid.yaml      # Pool A: Local-7B + Haiku + Sonnet + self (+ held-out Opus)
│   └── pool_open_specialists.yaml   # Pool B: Math-7B + Coder-7B + Base-7B + self (0 API cost)
├── src/
│   ├── config.py                    # Pool/Train configs + ID-Permutation Augmentation
│   ├── prompts.py                   # Outer & recursive Conductor prompts + Worker context injection
│   ├── parser.py                    # Safe ast.literal_eval parser + strict DAG/cycle validator
│   ├── workers.py                   # Async Anthropic & Local HF backends + SQLite response cache
│   ├── executor.py                  # Layered async DAG executor + bounded recursion + trace logger
│   ├── verifiers.py                 # Deterministic Math (sympy), Code (sandbox), and GPQA verifiers
│   ├── grpo.py                      # Pure PyTorch GRPO trainer (top_level & all_depth credit modes)
│   └── metrics.py                   # Topology analytics + publication-ready RQ1–RQ4 plots
├── scripts/
│   ├── 01_build_dataset.py          # Prepare unified train/eval JSONL datasets
│   ├── 02_gen_sft_data.py           # Synthesize & verify diverse warm-start trajectories via Claude
│   ├── 03_train_sft.py              # Fast LoRA SFT warmup (format compliance > 95%)
│   ├── 04_train_grpo.py             # Main GRPO training loop with Drive checkpoint resume
│   └── 05_eval_suite.py             # Run RQ1–RQ4 evaluations and generate figures
├── notebooks/
│   └── colab_h100_quickstart.ipynb  # One-click Colab H100 notebook
└── tests/
    └── test_core.py                 # Comprehensive local unit tests (parser, executor, RL math)
```

---

## Quickstart (Colab H100 / A100)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# 1. Verify all core modules locally
pytest -v

# 2. Build datasets (use --demo for instant offline smoke test)
python scripts/01_build_dataset.py --output_dir data

# 3. Stage 0: Generate verified SFT warmup trajectories & run light LoRA SFT
python scripts/02_gen_sft_data.py \
  --pool configs/pool_claude_hybrid.yaml \
  --dataset data/train.jsonl \
  --output data/sft_warmup.jsonl \
  --num_samples 300

python scripts/03_train_sft.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/sft_warmup.jsonl \
  --output_dir checkpoints/sft_warmup

# 4. Stage 1: End-to-End GRPO Training (e.g., RQ2 all-depth recursive + RQ3 ID permutation)
python scripts/04_train_grpo.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --init_adapter checkpoints/sft_warmup \
  --pool configs/pool_claude_hybrid.yaml \
  --train_data data/train.jsonl \
  --credit_mode all_depth \
  --allow_recursion \
  --permute_ids \
  --cost_lambda 0.05 \
  --save_dir checkpoints/grpo_all_depth_perm

# 5. Evaluate RQ1–RQ4 and export figures
python scripts/05_eval_suite.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter checkpoints/grpo_all_depth_perm/latest \
  --pool configs/pool_claude_hybrid.yaml \
  --eval_data data/eval_all.jsonl \
  --include_held_out \
  --output_dir results/rq_eval
```

---

## Reference

- Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, Yujin Tang. *Learning to Orchestrate Agents in Natural Language with the Conductor*. ICLR 2026 ([arXiv:2512.04388](https://arxiv.org/abs/2512.04388)).
