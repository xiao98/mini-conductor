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

## Status

**No experimental results are checked in.** Every figure and number must come from a real run of
`scripts/06_run_matrix.py`; the aggregator only draws runs whose output files exist. What has
been verified so far: unit tests (`pytest -v`), the real HuggingFace dataset build, and a CPU
plumbing smoke test (`scripts/smoke_cpu.py`: 135M toy model + mock workers through
SFT -> GRPO -> resume -> eval -> figures). The smoke test proves the pipeline executes, nothing more.

---

## Project Structure

```text
mini-conductor/
├── configs/
│   ├── pool_claude_hybrid.yaml      # Pool A: Local-7B + Haiku + Sonnet + self (+ held-out Opus)
│   ├── pool_open_specialists.yaml   # Pool B: Math-7B + Coder-7B + Base-7B + self (0 API cost)
│   └── pool_smoke_mock.yaml         # Mock workers for the CPU smoke test only
├── src/
│   ├── config.py                    # Pool configs + ID-Permutation Augmentation
│   ├── prompts.py                   # Outer & recursive Conductor prompts + worker prompts
│   ├── parser.py                    # ast.literal_eval parser + strict acyclic DAG validator
│   ├── workers.py                   # Claude backend (throttle/backoff/SQLite cache), GPU micro-batcher
│   ├── executor.py                  # Layered async DAG executor + bounded recursion + trace logger
│   ├── verifiers.py                 # Deterministic Math (sympy), Code (subprocess sandbox), MCQ
│   ├── grpo.py                      # Pure PyTorch GRPO (top_level & all_depth credit modes)
│   ├── runtime.py                   # Model loading + generation glue shared by train/eval
│   └── metrics.py                   # Topology analytics + RQ1-RQ4 plots
├── scripts/
│   ├── 00_check_api.py              # Preflight: 1 minimal request per Claude model in the pool
│   ├── 01_build_dataset.py          # MATH L3-5 / MATH-500, MBPP, GPQA-Diamond or MMLU-Pro STEM
│   ├── 02_gen_sft_data.py           # Claude-synthesized, verifier-filtered warm-start trajectories
│   ├── 03_train_sft.py              # LoRA SFT warm start
│   ├── 04_train_grpo.py             # One GRPO run (resumable)
│   ├── 05_eval_suite.py             # One adapter x test-time pool conditions (+ baselines)
│   ├── 06_run_matrix.py             # ALL runs for RQ1-RQ4: train -> eval -> figures (resumable)
│   └── smoke_cpu.py                 # CPU end-to-end plumbing check (no GPU / API key)
├── notebooks/colab_h100_quickstart.ipynb
└── tests/                           # test_core.py, test_pipeline.py
```

---

## Quickstart (Colab H100 / A100)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
export HF_TOKEN="hf_..."          # optional: gated GPQA-Diamond (else MMLU-Pro STEM is used)

pytest -v                                         # 1. unit tests
python scripts/00_check_api.py                    # 2. 1 request per Claude model: IDs + request shape
python scripts/01_build_dataset.py --output_dir data

# 3. Stage 0: Claude-synthesized warm start (canonical worker IDs) + LoRA SFT
python scripts/02_gen_sft_data.py --dataset data/train.jsonl --output data/sft_warmup.jsonl --num_samples 300
python scripts/03_train_sft.py --data data/sft_warmup.jsonl --output_dir checkpoints/sft_warmup

# 4. Stage 1 + evaluation + figures for RQ1-RQ4, resumable (point --root at Google Drive)
python scripts/06_run_matrix.py --root /content/drive/MyDrive/mc_exp --init_adapter checkpoints/sft_warmup
```

Re-running step 4 after a Colab disconnect continues where it stopped. Use `--only base,rq3_perm`
to run a subset, `--figures_only` to re-draw.

### The experiment matrix

All runs share the SFT warm start, seed, data, pool, hyperparameters and eval subset; each
differs from `base` in exactly one switch.

| Run | Differs from `base` by | Used for |
| :-- | :-- | :-- |
| `base` | recursion on, `all_depth` credit, canonical IDs, `cost_lambda=0` | RQ1 curves; shared arm of RQ2/RQ3/RQ4 |
| `rq2_no_recursion` | `self` worker removed | RQ2 |
| `rq2_top_level` | `--credit_mode top_level` | RQ2 |
| `rq3_perm` | `--permute_ids` (ID-Permutation Augmentation) | RQ3 |
| `rq4_lam0.05` / `0.15` / `0.30` | `--cost_lambda` | RQ4 |

Each adapter is evaluated greedily under four test-time pool conditions: `canonical`,
`permuted_ids`, `+heldout` (Claude Opus injected zero-shot) and `+heldout+perm`. Baselines are
single workers answering directly and a static "all solve in parallel, strongest judges" topology.
Accuracies are reported with 95% CI half-widths; with the default 150 eval tasks a single
accuracy is only resolved to roughly +-8 points, so raise `--eval_limit` before claiming small gaps.

### Notes on the Claude backend

- Current Claude models reject non-default sampling parameters, so `temperature` is never sent.
  In the pool YAML, `temperature <= 0.25` only marks a worker as cacheable in SQLite.
- Thinking is on by default and counts toward `max_tokens`, so Claude workers get `max_tokens: 8000`
  and an explicit `effort: low` (model defaults differ, which would confound cost comparisons).
- Only 429 / 5xx / connection errors are retried (exponential backoff + jitter). A 4xx such as a
  wrong model ID fails immediately.
- Server-side model fallbacks are deliberately NOT enabled: a silent model switch would
  contaminate per-worker accuracy and cost. Refusals / truncations are counted in `api_stop_reasons`.

---

## Reference

- Stefan Nielsen, Edoardo Cetin, Peter Schwendeman, Qi Sun, Jinglue Xu, Yujin Tang. *Learning to Orchestrate Agents in Natural Language with the Conductor*. ICLR 2026 ([arXiv:2512.04388](https://arxiv.org/abs/2512.04388)).
