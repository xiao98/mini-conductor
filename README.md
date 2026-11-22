# `mini-conductor`

**A small single-GPU re-implementation of the orchestration idea in Sakana AI's [*Learning to Orchestrate Agents in Natural Language with the Conductor*](https://arxiv.org/abs/2512.04388) (ICLR 2026), built to answer one question the paper leaves open.**

> `PyTorch + transformers + peft + asyncio` only, one Colab GPU + the Claude API. This is **not** a
> faithful reproduction: see [Differences from the paper](#differences-from-the-paper).

---

## The question

> **When a worker the Conductor has never seen is added at test time, does an RL-trained
> Conductor use it, and does it route by what a worker is *described as* or by which *ID slot*
> it occupies?**

The paper passes workers to the Conductor as ordinal numbers and trains on random subsets of a
fixed pool; it evaluates on subsets of that same pool, never on a model that was absent from
training. Here every worker is shown as `ID | name | cost tier | description`, which makes the
question testable:

| | Test-time pool | What it isolates |
| :-- | :-- | :-- |
| `canonical` | training pool, training IDs | reference |
| `permuted_ids` | training pool, IDs shuffled per task | reliance on ID slots |
| `+heldout` | Claude Haiku injected, never seen in training | use of an unseen worker |
| `+heldout+perm` | both | the two combined |

The training pool is two local open models (Qwen2.5-1.5B and Qwen2.5-7B, so RL makes no API
calls); the unseen worker is the much stronger Claude Haiku. Within the training pool the best
policy is simply "use the 7B", which a fixed-ID Conductor can learn as a slot and a shuffled-ID
Conductor can only learn from the descriptions.

Two training arms are compared, differing in one switch: fixed IDs (`base`) versus IDs shuffled
per episode during training (`rq3_perm`). Reported per condition: accuracy with 95% CI, and how
often each worker (including the unseen one) is actually called.

**Gate before any training.** Orchestration can only matter if no single worker already solves
the eval set. Step 3 of the quickstart measures every worker answering directly; if the strongest
training worker is near ceiling, or the held-out worker is not clearly better than it, the pool
or the tasks must change before the experiment means anything.

### Gate results (measured, direct answering, one sample per task)

150-task eval subset (50 MATH-500 / 50 MBPP / 50 MMLU-Pro STEM), accuracy with 95% CI half-width:

| Worker | Overall | Math | Code | MCQ |
| :-- | :-- | :-- | :-- | :-- |
| Qwen2.5-1.5B | 51.3% +-8.0 | 48% | 54% | 52% |
| Qwen2.5-7B | 74.0% +-7.0 | 62% | 86% | 74% |
| Claude Haiku (held out) | 92.7% +-4.2 | 86% | 96% | 96% |
| Claude Sonnet | 93.3% +-4.0 | 88% | 100% | 92% |
| Claude Opus | 94.7% +-3.6 | 90% | 98% | 96% |

This is why the pool is what it is. An earlier design trained on Qwen-7B + Haiku + Sonnet and
held out Opus; it failed this gate because the three Claude models are indistinguishable here.
A harder set (60 AIME 2024/25 + 60 recent LeetCode Hard) did not rescue it (Haiku 76.7%, Sonnet
85.0%, Opus 86.7%, with Opus solving no task that Haiku or Sonnet had not) and is unusable for a
local pool (Qwen-7B 6.7%, Qwen-1.5B 0%). On the 150 tasks the 1.5B solves only 5 that the 7B misses;
Haiku solves 30 that the 7B misses and loses 2. Claude workers ran at `effort: low`, Qwen at
temperature 0.2 with up to 2048 new tokens. All rows were scored before the math-verifier
normalisation fix in `src/verifiers.py`; by hand count that undercounts each Claude model by
about 5 math tasks (the Qwen rows were not re-checked), so step 3 below re-measures the three pool workers.

The code also contains a recursion / credit-assignment ablation, a cost-penalty sweep and
topology-over-training plots (`rq2_*`, `rq4_*` runs). They are implemented and tested but are
secondary and not part of the default workflow.

---

## Differences from the paper

Paper values are taken from arXiv v5; "not stated" means the paper does not say.

| | Paper | This repo |
| :-- | :-- | :-- |
| Worker pool | large closed and open models (GPT-5, Gemini 2.5 Pro, Claude Sonnet 4, DeepSeek-R1-Distill-32B, Gemma3-27B, Qwen3-32B) | local Qwen2.5-1.5B and Qwen2.5-7B (+ held-out Claude Haiku) |
| Worker identity shown to Conductor | ordinal numbers | ID + name + cost tier + description |
| Recursion | a second Conductor round after the workflow finishes, added by 20 extra finetuning iterations | `self` is a worker inside the plan, trained jointly from the start |
| Reward | 0 unparseable / 0.5 parsed but wrong / 1 correct | 0 / 0.2 / 1, optional cost penalty |
| KL penalty | 0 | 0.04 |
| Rollouts per update | 4 questions x 64 rollouts, temperature 1.0 | 8 questions x 8 rollouts, temperature 0.8 |
| Learning rate / length | 1e-6, 200 iterations | 1e-5, 160 steps |
| Warm start | few-shot examples in the prompt | separate LoRA SFT stage on Claude-synthesized plans |
| Conductor finetuning | Qwen2.5-7B (LoRA or full: not stated) | Qwen2.5-7B + LoRA |
| Training data | 960 problems: MMLU, MATH500, LiveCodeBench, RLPR | 1000 problems: MATH L3-5, MBPP, MMLU-Pro STEM |
| Hardware | 2x H100 | 1 GPU |

The 8-rollout groups in particular give a much noisier advantage estimate than the paper's 64.

---

## Results so far: the warm-started Conductor, before any RL

One evaluation run, 150 tasks, one seed, greedy Conductor. The Conductor is Qwen2.5-7B + LoRA
after the SFT warm start only (185 Haiku-written plans, canonical IDs, Haiku never in the pool).
Raw outputs: [`experiments/sft_only_2026-11-21/`](experiments/sft_only_2026-11-21).

| Test-time pool | Accuracy | Valid plans | Worker calls: 7B / 1.5B / `self` / Haiku |
| :-- | :-- | :-- | :-- |
| `canonical` | 69.3% +-7.4 | 94.0% | 87% / 0% / 13% / - |
| `permuted_ids` | 64.7% +-7.6 | 91.3% | 86% / 3% / 11% / - |
| `+heldout` | 69.3% +-7.4 | 90.0% | 79% / 0.4% / 13% / 8% |
| `+heldout+perm` | 65.3% +-7.6 | 86.7% | 69% / 4% / 11% / 16% |

For reference on the same tasks: Qwen-7B alone 76.0%, static parallel+judge 68.7%, Haiku alone 97.3%.

**1. Orchestration does not pay yet.** The Conductor never beats its own best worker answering
directly. It reliably avoids the 1.5B, but 6-13% of its plans are invalid (scored wrong), 33-44% of
episodes recurse, and multi-step plans show no consistent accuracy gain over single-step ones.

**2. A much stronger unseen worker is mostly ignored.** With Haiku available under a new ID,
it appears in the top-level plan of 8 of 150 episodes (all 8 correct) and accuracy does not move.

**3. Known workers are found by identity, the unseen one by slot.** Share of episodes whose
top-level plan calls a worker, split by the ID that worker was assigned in that episode
(`+heldout+perm`; training IDs were 0 = 1.5B, 1 = 7B, 2 = `self`):

| Worker holds ID | 0 | 1 | 2 | 3 |
| :-- | :-- | :-- | :-- | :-- |
| Qwen-7B (seen) | 41/49 | 45/45 | 26/29 | 23/27 |
| Claude Haiku (unseen) | 4/39 | 14/40 | 17/34 | 7/37 |
| Qwen-1.5B (seen, weak) | 1/34 | 5/33 | 4/41 | 0/42 |

The 7B is called 84-100% of the time wherever it sits, although it was only ever seen at ID 1.
Haiku is called 35-50% of the time when it lands on an ID that was *used* in training (1 or 2)
and 10-19% otherwise. In `permuted_ids` the 1.5B is called only when it holds ID 1 (6/44, else 0/106).

Limits: 27-57 episodes per cell, so only the large gaps are meaningful; name and description are
always shown together, so "identity" cannot be split further; the slot table counts top-level
plan steps only, while the worker-call shares above also count calls inside recursive sub-plans;
this is the warm-started policy, and RL may change all of it.

---

## Status

**No RL-trained Conductor results exist yet.** A GRPO pilot on an A100 completed 2 steps at
about 20 minutes per step (8 tasks x 8 rollouts) before it was stopped, which makes the default
160-step runs impractical on one GPU without further speed work; in its second step 4 of the 8
task groups had identical rewards across all 8 rollouts and so carried no signal. Measured so far: the gate numbers and the
SFT-only evaluation above. Every figure and number must come from a real run; the aggregator
only draws runs whose output files exist. Also verified: unit tests (`pytest -v`), the real
HuggingFace dataset build, and a CPU plumbing smoke test (`scripts/smoke_cpu.py`: 135M toy model
+ mock workers through SFT -> GRPO -> resume -> eval -> figures), which proves the pipeline
executes, nothing more.

---

## Project Structure

```text
mini-conductor/
├── configs/
│   ├── pool_local_weak.yaml         # DEFAULT: Qwen-1.5B + Qwen-7B + self (+ held-out Claude Haiku)
│   ├── pool_claude_hybrid.yaml      # Earlier design (failed the gate): Local-7B + Haiku + Sonnet (+ Opus)
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
│   ├── 07_slot_analysis.py          # Per-worker, per-ID-slot usage from an eval's details files
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
ROOT=/content/drive/MyDrive/mc_exp

pytest -v                                         # 1. unit tests
python scripts/00_check_api.py                    # 2. 1 request per Claude model: IDs + request shape
python scripts/01_build_dataset.py --output_dir data

# 3. GATE, no training: every worker answers directly + a static topology (same tasks the matrix evaluates)
python scripts/05_eval_suite.py --baselines --skip_conductor --limit 150 \
    --output_dir $ROOT/eval/baselines --cache_db $ROOT/worker_cache.sqlite
#    Continue only if the strongest training worker is clearly below ceiling AND the held-out worker beats it.

# 4. Stage 0: Claude-synthesized warm start (canonical worker IDs) + LoRA SFT.
#    The only training-side API use: ~1 Sonnet + a few Haiku calls per sample.
python scripts/02_gen_sft_data.py --dataset data/train.jsonl --output data/sft_warmup.jsonl --num_samples 300
python scripts/03_train_sft.py --data data/sft_warmup.jsonl --output_dir checkpoints/sft_warmup

# 5. Stage 1 + evaluation + figures for the two arms, resumable (ROOT on Google Drive)
python scripts/06_run_matrix.py --root $ROOT --init_adapter checkpoints/sft_warmup --only base,rq3_perm
```

Step 3 writes exactly the baseline summary the matrix would compute, so step 5 reuses it.
Re-running step 5 after a Colab disconnect continues where it stopped; `--figures_only` re-draws.
Dropping `--only` also runs the secondary `rq2_*` / `rq4_*` runs.

### The experiment matrix

All runs share the SFT warm start, seed, data, pool, hyperparameters and eval subset; each
differs from `base` in exactly one switch.

| Run | Differs from `base` by | Used for |
| :-- | :-- | :-- |
| `base` | recursion on, `all_depth` credit, canonical IDs, `cost_lambda=0` | RQ1 curves; shared arm of RQ2/RQ3/RQ4 |
| `rq2_no_recursion` | `self` worker removed | RQ2 |
| `rq2_top_level` | `--credit_mode top_level` | RQ2 |
| `rq3_perm` | `--permute_ids` (IDs shuffled per training episode) | **the main question** |
| `rq4_lam0.05` / `0.15` / `0.30` | `--cost_lambda` | RQ4 |

Each adapter is evaluated greedily under four test-time pool conditions: `canonical`,
`permuted_ids`, `+heldout` (Claude Haiku injected zero-shot) and `+heldout+perm`. Baselines are
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
