"""Build unified Math / Code / GPQA JSONL datasets for Conductor training and evaluation.

Usage:
    # Full HuggingFace dataset build (Colab / networked environment):
    python scripts/01_build_dataset.py --output_dir data

    # Instant offline smoke-test dataset:
    python scripts/01_build_dataset.py --output_dir data_demo --demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEMO_EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "demo_math_1",
        "task_type": "math",
        "source": "demo",
        "prompt": "Find the sum of all positive integers n such that n^2 + 2n + 2 divides n^3 + 4n^2 + 4n - 14. Put your final answer in \\boxed{...}.",
        "ground_truth": "4",
        "metadata": {"level": 4},
    },
    {
        "id": "demo_math_2",
        "task_type": "math",
        "source": "demo",
        "prompt": "Compute the value of (1/2) + (1/4) + (1/8) + (1/16). Put your final answer in \\boxed{...}.",
        "ground_truth": "\\frac{15}{16}",
        "metadata": {"level": 1},
    },
    {
        "id": "demo_math_3",
        "task_type": "math",
        "source": "demo",
        "prompt": "How many integers between 1 and 100 inclusive are divisible by 3 or 5? Put your final answer in \\boxed{...}.",
        "ground_truth": "47",
        "metadata": {"level": 2},
    },
    {
        "id": "demo_code_1",
        "task_type": "code",
        "source": "demo",
        "prompt": "Write a Python function `longest_increasing_subseq_len(nums: list[int]) -> int` that returns the length of the strictly longest increasing subsequence in O(n log n) time.",
        "ground_truth": (
            "assert longest_increasing_subseq_len([]) == 0\n"
            "assert longest_increasing_subseq_len([10, 9, 2, 5, 3, 7, 101, 18]) == 4\n"
            "assert longest_increasing_subseq_len([7, 7, 7, 7]) == 1\n"
            "assert longest_increasing_subseq_len([0, 1, 0, 3, 2, 3]) == 4\n"
        ),
        "metadata": {},
    },
    {
        "id": "demo_code_2",
        "task_type": "code",
        "source": "demo",
        "prompt": "Write a Python function `is_balanced(s: str) -> bool` that checks if a string containing only '()[]{}' has properly matched and nested brackets.",
        "ground_truth": (
            "assert is_balanced('') is True\n"
            "assert is_balanced('({[]})') is True\n"
            "assert is_balanced('([)]') is False\n"
            "assert is_balanced('(((') is False\n"
        ),
        "metadata": {},
    },
    {
        "id": "demo_gpqa_1",
        "task_type": "gpqa",
        "source": "demo",
        "prompt": (
            "Which of the following best explains why GRPO (Group Relative Policy Optimization) reduces memory overhead compared to standard PPO in LLM RL?\n"
            "(A) It freezes all attention layers during backpropagation.\n"
            "(B) It estimates the baseline from group-normalized rollout rewards instead of training a separate value/critic network.\n"
            "(C) It replaces softmax sampling with greedy decoding.\n"
            "(D) It eliminates the KL divergence penalty completely."
        ),
        "ground_truth": "B",
        "metadata": {},
    },
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[saved] {path} ({len(rows)} rows)")


def _build_math_splits(
    train_n: int,
    eval_n: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset
    from src.verifiers import extract_boxed

    eval_rows: list[dict[str, Any]] = []
    try:
        ds_eval = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for idx, ex in enumerate(ds_eval):
            eval_rows.append(
                {
                    "id": f"math500_{idx}",
                    "task_type": "math",
                    "source": "MATH-500",
                    "prompt": f"{ex['problem']}\nPut your final answer in \\boxed{{...}}.",
                    "ground_truth": str(ex["answer"]),
                    "metadata": {"level": ex.get("level"), "subject": ex.get("subject")},
                }
            )
    except Exception as exc:
        print(f"[warn] Could not load HuggingFaceH4/MATH-500: {exc}")

    train_rows: list[dict[str, Any]] = []
    try:
        ds_train = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
        candidates: list[dict[str, Any]] = []
        eval_prompts = {r["prompt"] for r in eval_rows}
        for idx, ex in enumerate(ds_train):
            level_str = str(ex.get("level", ""))
            # Focus on Level 3-5 so multi-agent orchestration actually matters
            if any(k in level_str for k in ("3", "4", "5")):
                ans = extract_boxed(ex["solution"]) or ex["solution"]
                prompt = f"{ex['problem']}\nPut your final answer in \\boxed{{...}}."
                if prompt in eval_prompts:
                    continue
                candidates.append(
                    {
                        "id": f"math_train_{idx}",
                        "task_type": "math",
                        "source": "MATH-lighteval",
                        "prompt": prompt,
                        "ground_truth": ans,
                        "metadata": {"level": level_str, "type": ex.get("type")},
                    }
                )
        rng.shuffle(candidates)
        train_rows = candidates[:train_n]
    except Exception as exc:
        print(f"[warn] Could not load MATH-lighteval train split: {exc}")

    rng.shuffle(eval_rows)
    return train_rows, eval_rows[:eval_n]


def _build_code_splits(
    train_n: int,
    eval_n: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train+test+validation")
        for idx, ex in enumerate(ds):
            tests = "\n".join(ex["test_list"])
            prompt = (
                f"{ex['prompt']}\n\n"
                f"Your code must pass tests such as:\n```python\n{ex['test_list'][0]}\n```\n"
                "Output clean executable Python code inside a ```python ... ``` block."
            )
            rows.append(
                {
                    "id": f"mbpp_{ex.get('task_id', idx)}",
                    "task_type": "code",
                    "source": "mbpp_sanitized",
                    "prompt": prompt,
                    "ground_truth": tests,
                    "metadata": {},
                }
            )
    except Exception as exc:
        print(f"[warn] Could not load MBPP sanitized: {exc}")

    rng.shuffle(rows)
    eval_rows = rows[:eval_n]
    train_rows = rows[eval_n : eval_n + train_n]
    return train_rows, eval_rows


def _build_gpqa_splits(
    train_n: int,
    eval_n: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from datasets import load_dataset

    eval_rows: list[dict[str, Any]] = []
    try:
        # Note: Idavidrein/gpqa requires HF access grant (`huggingface-cli login`).
        ds_gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        letters = ["A", "B", "C", "D"]
        for idx, ex in enumerate(ds_gpqa):
            choices = [
                (ex["Correct Answer"], True),
                (ex["Incorrect Answer 1"], False),
                (ex["Incorrect Answer 2"], False),
                (ex["Incorrect Answer 3"], False),
            ]
            rng.shuffle(choices)
            gt_letter = next(letters[i] for i, (_, ok) in enumerate(choices) if ok)
            formatted = "\n".join(f"({letters[i]}) {txt}" for i, (txt, _) in enumerate(choices))
            eval_rows.append(
                {
                    "id": f"gpqa_diamond_{idx}",
                    "task_type": "gpqa",
                    "source": "GPQA-Diamond",
                    "prompt": f"{ex['Question']}\n\n{formatted}",
                    "ground_truth": gt_letter,
                    "metadata": {"subdomain": ex.get("Subdomain")},
                }
            )
    except Exception as exc:
        print(f"[warn] Could not load gated Idavidrein/gpqa ({exc}); falling back to MMLU-Pro STEM.")

    train_rows: list[dict[str, Any]] = []
    try:
        ds_mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        stem_cats = {"physics", "chemistry", "biology", "computer science", "math", "engineering"}
        letters = ["A", "B", "C", "D"]
        candidates: list[dict[str, Any]] = []
        for idx, ex in enumerate(ds_mmlu):
            if str(ex.get("category", "")).lower() not in stem_cats:
                continue
            opts = list(ex["options"])
            ans_idx = int(ex["answer_index"])
            if ans_idx >= len(opts) or len(opts) < 4:
                continue
            correct_text = opts[ans_idx]
            distractors = [o for i, o in enumerate(opts) if i != ans_idx]
            rng.shuffle(distractors)
            four = [(correct_text, True)] + [(d, False) for d in distractors[:3]]
            rng.shuffle(four)
            gt_letter = next(letters[i] for i, (_, ok) in enumerate(four) if ok)
            formatted = "\n".join(f"({letters[i]}) {txt}" for i, (txt, _) in enumerate(four))
            candidates.append(
                {
                    "id": f"mmlu_pro_stem_{idx}",
                    "task_type": "gpqa",
                    "source": "MMLU-Pro-STEM-4way",
                    "prompt": f"{ex['question']}\n\n{formatted}",
                    "ground_truth": gt_letter,
                    "metadata": {"category": ex.get("category")},
                }
            )
        rng.shuffle(candidates)
        train_rows = candidates[:train_n]
        if not eval_rows:
            eval_rows = candidates[train_n : train_n + eval_n]
    except Exception as exc:
        print(f"[warn] Could not load MMLU-Pro: {exc}")

    return train_rows, eval_rows[:eval_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified Conductor train/eval JSONL splits.")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--demo", action="store_true", help="Write tiny offline demo dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--math_train", type=int, default=500)
    parser.add_argument("--math_eval", type=int, default=200)
    parser.add_argument("--code_train", type=int, default=300)
    parser.add_argument("--code_eval", type=int, default=120)
    parser.add_argument("--gpqa_train", type=int, default=200)
    parser.add_argument("--gpqa_eval", type=int, default=198)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    if args.demo:
        _write_jsonl(out_dir / "train.jsonl", DEMO_EXAMPLES)
        _write_jsonl(out_dir / "eval_all.jsonl", DEMO_EXAMPLES)
        return

    m_tr, m_ev = _build_math_splits(args.math_train, args.math_eval, rng)
    c_tr, c_ev = _build_code_splits(args.code_train, args.code_eval, rng)
    g_tr, g_ev = _build_gpqa_splits(args.gpqa_train, args.gpqa_eval, rng)

    train_all = m_tr + c_tr + g_tr
    rng.shuffle(train_all)
    eval_all = m_ev + c_ev + g_ev

    _write_jsonl(out_dir / "train.jsonl", train_all)
    _write_jsonl(out_dir / "eval_math.jsonl", m_ev)
    _write_jsonl(out_dir / "eval_code.jsonl", c_ev)
    _write_jsonl(out_dir / "eval_gpqa.jsonl", g_ev)
    _write_jsonl(out_dir / "eval_all.jsonl", eval_all)


if __name__ == "__main__":
    main()
