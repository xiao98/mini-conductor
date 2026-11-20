"""Comprehensive local unit tests for mini-conductor core modules and PyTorch GRPO math."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import WorkerSpec, build_episode_worker_view, load_pool_config
from src.executor import RecursiveOrchestrator
from src.grpo import (
    ConductorGRPOTrainer,
    GRPOConfig,
    compute_completion_logprobs,
    compute_group_advantages,
    compute_reward,
    schulman_kl_divergence,
)
from src.metrics import plot_pareto_frontier, plot_topology_evolution, summarize_traces
from src.parser import compute_dag_layers, parse_and_validate_plan
from src.verifiers import extract_boxed, verify_answer
from src.workers import MockWorkerBackend, SQLiteResponseCache, WorkerRouter


# ---------------------------------------------------------------------------
# 1. Config & ID-Permutation Tests
# ---------------------------------------------------------------------------

def test_load_pool_configs_and_held_out():
    pool_a = load_pool_config(ROOT / "configs" / "pool_claude_hybrid.yaml")
    assert pool_a.name == "claude_hybrid"
    assert len(pool_a.active_workers(include_held_out=False, allow_recursion=True)) == 4
    assert len(pool_a.active_workers(include_held_out=False, allow_recursion=False)) == 3
    assert len(pool_a.active_workers(include_held_out=True, allow_recursion=True)) == 5

    pool_b = load_pool_config(ROOT / "configs" / "pool_open_specialists.yaml")
    assert pool_b.name == "open_specialists"
    assert len(pool_b.active_workers()) == 4


def test_id_permutation_augmentation():
    pool_a = load_pool_config(ROOT / "configs" / "pool_claude_hybrid.yaml")
    workers = pool_a.active_workers(include_held_out=False, allow_recursion=True)
    canon = build_episode_worker_view(workers, permute_ids=False)
    assert canon.name_to_id["self"] == 3

    seen_self_ids = set()
    for seed in range(25):
        v = build_episode_worker_view(workers, permute_ids=True, rng=random.Random(seed))
        assert v.valid_ids == {0, 1, 2, 3}
        assert set(v.name_to_id.keys()) == set(canon.name_to_id.keys())
        assert v.self_id == v.name_to_id["self"]
        seen_self_ids.add(v.self_id)
    assert len(seen_self_ids) > 1


# ---------------------------------------------------------------------------
# 2. Parser & Strict DAG Validation Tests
# ---------------------------------------------------------------------------

def test_parser_valid_parallel_and_all_keyword():
    raw = """
<think>
Run two solvers in parallel, then verify with step 2 seeing all prior outputs.
</think>
<plan>
model_id = [0, 1, 2]
subtasks = [
    "Solve via method A.",
    "Solve via method B.",
    "Verify both solutions and output final answer."
]
access_list = [[], [], "all"]
</plan>
"""
    res = parse_and_validate_plan(raw, valid_worker_ids={0, 1, 2, 3}, max_steps=5)
    assert res.ok and res.plan is not None
    assert res.plan.model_ids == [0, 1, 2]
    assert res.plan.access_lists == [[], [], [0, 1]]
    assert compute_dag_layers(res.plan.access_lists) == [[0, 1], [2]]


def test_parser_rejects_cycles_bounds_depth_and_unsafe_code():
    # Cycle / self-dependency
    bad_cycle = """
<plan>
model_id = [0, 1]
subtasks = ["a", "b"]
access_list = [[0], [0]]
</plan>
"""
    assert not parse_and_validate_plan(bad_cycle, valid_worker_ids={0, 1}).ok

    # Disallowed recursive self ID at max depth
    blocked_self = """
<plan>
model_id = [0, 3]
subtasks = ["solve", "replan"]
access_list = [[], [0]]
</plan>
"""
    assert not parse_and_validate_plan(
        blocked_self, valid_worker_ids={0, 1, 2, 3}, disallowed_ids={3}
    ).ok

    # Arbitrary code execution inside <plan> MUST fail ast.literal_eval
    unsafe = """
<plan>
model_id = [__import__('os').system('echo hacked')]
subtasks = ["a"]
access_list = [[]]
</plan>
"""
    assert not parse_and_validate_plan(unsafe, valid_worker_ids={0}).ok


# ---------------------------------------------------------------------------
# 3. Async Layered Parallelism & Recursive Orchestration Tests
# ---------------------------------------------------------------------------

def test_async_parallel_layer_execution_and_recursion():
    workers = [
        WorkerSpec(id=0, name="fast-a", backend="mock", model_name="mock-a", description="A", input_cost_per_mtok=1.0, output_cost_per_mtok=2.0),
        WorkerSpec(id=1, name="fast-b", backend="mock", model_name="mock-b", description="B", input_cost_per_mtok=1.0, output_cost_per_mtok=2.0),
        WorkerSpec(id=2, name="self", backend="conductor_self", model_name="self", description="Recursive self"),
    ]
    view = build_episode_worker_view(workers, permute_ids=False)

    def _mock_worker(spec: WorkerSpec, messages: list[dict[str, str]]) -> str:
        user_msg = messages[-1]["content"]
        if "verify" in user_msg.lower():
            return "Verified after checking prior outputs. Final answer: \\boxed{42}"
        return f"Candidate from {spec.name}: 42"

    router = WorkerRouter(mock_backend=MockWorkerBackend(_mock_worker, simulated_delay_s=0.06))
    orch = RecursiveOrchestrator(worker_router=router, max_steps=5, max_depth=2)

    call_depths: list[int] = []

    async def _conductor_gen(messages: list[dict[str, str]]) -> str:
        user_content = messages[-1]["content"]
        if "Intermediate Outputs from Parent Plan" not in user_content:
            call_depths.append(0)
            # 3 parallel steps + 1 recursive self step
            return (
                "<think>Try two parallel workers, then recursively inspect.</think>\n"
                "<plan>\n"
                "model_id = [0, 1, 0, 2]\n"
                "subtasks = ['attempt 1', 'attempt 2', 'attempt 3', 'Check consistency and verify']\n"
                "access_list = [[], [], [], [0, 1, 2]]\n"
                "</plan>"
            )
        call_depths.append(1)
        assert "Candidate from fast-a: 42" in user_content
        assert "Candidate from fast-b: 42" in user_content
        return (
            "<think>Workers agree; dispatch final verification.</think>\n"
            "<plan>\n"
            "model_id = [1]\n"
            "subtasks = ['Verify and produce final answer']\n"
            "access_list = [[]]\n"
            "</plan>"
        )

    t0 = time.perf_counter()
    trace = asyncio.run(
        orch.execute_episode(
            task_prompt="What is 6 * 7?",
            task_type="math",
            worker_view=view,
            conductor_generate_fn=_conductor_gen,
        )
    )
    elapsed = time.perf_counter() - t0

    assert trace.format_valid
    assert trace.recursion_triggered
    assert trace.max_depth_reached == 1
    assert trace.max_parallel_width == 3
    assert len(trace.conductor_turns) == 2
    assert call_depths == [0, 1]
    assert "\\boxed{42}" in trace.final_answer
    # 3 parallel calls of 0.06s + 1 inner call of 0.06s should take ~0.12s, definitely < 0.23s
    assert elapsed < 0.23, f"Expected parallel speedup, got {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# 4. Verifiers & SQLite Cache Tests
# ---------------------------------------------------------------------------

def test_verifiers_math_code_gpqa():
    # Math: nested braces & sympy equivalence
    assert extract_boxed(r"Thus $\boxed{\frac{1}{2}}$ is the answer.") == r"\frac{1}{2}"
    ok, _ = verify_answer("math", r"Final result: \boxed{\frac{2}{4}}", "0.5")
    assert ok
    ok_bad, _ = verify_answer("math", r"\boxed{7}", "8")
    assert not ok_bad

    # Code: pass, fail, and infinite loop timeout
    good_code = "```python\ndef add(a, b):\n    return a + b\n```"
    ok_c, _ = verify_answer("code", good_code, "assert add(2, 3) == 5")
    assert ok_c
    bad_c, _ = verify_answer("code", good_code, "assert add(2, 3) == 99")
    assert not bad_c
    loop_code = "```python\nwhile True:\n    pass\n```"
    ok_t, reason_t = verify_answer("code", loop_code, "assert True")
    assert not ok_t and "timeout" in reason_t

    # GPQA / MCQ
    assert verify_answer("gpqa", "After analysis, Answer: C", "C")[0]
    assert verify_answer("gpqa", r"Therefore \boxed{B}.", "B")[0]
    assert not verify_answer("gpqa", "Answer: A", "D")[0]


def test_sqlite_cache_roundtrip(tmp_path: Path):
    cache = SQLiteResponseCache(tmp_path / "cache.sqlite")
    msgs = [{"role": "user", "content": "hello"}]
    key = cache.make_key("claude-haiku-5-5", msgs, 0.0, 128)
    assert cache.get(key) is None

    from src.workers import WorkerResponse
    cache.put(key, WorkerResponse(text="world", model_name="claude-haiku-5-5", input_tokens=5, output_tokens=3, cost_usd=0.0001))
    hit = cache.get(key)
    assert hit is not None and hit.cached and hit.text == "world"
    cache.close()


# ---------------------------------------------------------------------------
# 5. PyTorch GRPO Math & Recursive Credit Assignment Tests
# ---------------------------------------------------------------------------

class _TinyAdapterLM(nn.Module):
    """Minimal causal LM with LoRA-like `disable_adapter()` for exact GRPO unit testing."""

    def __init__(self, vocab_size: int = 64, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.base_proj = nn.Linear(d_model, vocab_size)
        self.adapter_proj = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.adapter_proj.weight, std=0.02)
        for p in self.embed.parameters():
            p.requires_grad_(False)
        for p in self.base_proj.parameters():
            p.requires_grad_(False)
        self._adapter_enabled = True

    @contextmanager
    def disable_adapter(self):
        prev = self._adapter_enabled
        self._adapter_enabled = False
        try:
            yield
        finally:
            self._adapter_enabled = prev

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.embed(input_ids)
        logits = self.base_proj(h)
        if self._adapter_enabled:
            logits = logits + self.adapter_proj(h)
        return SimpleNamespace(logits=logits)


class _DummyTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True):
        return " | ".join(m["content"] for m in messages)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [(ord(ch) % 60) + 2 for ch in text[:40]]


def test_grpo_math_and_recursive_credit_modes(tmp_path: Path):
    # Advantage edge cases
    assert compute_group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    advs = compute_group_advantages([0.0, 1.0])
    assert advs[0] < 0.0 and advs[1] > 0.0 and abs(sum(advs)) < 1e-5

    # Exact token logprob alignment & non-negative Schulman KL
    torch.manual_seed(0)
    lm = _TinyAdapterLM()
    seq = torch.tensor([[3, 7, 11, 15, 19, 23]], dtype=torch.long)
    lp = compute_completion_logprobs(lm, seq, prompt_len=3)
    assert lp.shape == (1, 3)

    full_logits = lm(seq).logits
    manual = F.log_softmax(full_logits[:, 2:5, :], dim=-1).gather(-1, seq[:, 3:6].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(lp, manual, atol=1e-6)

    with lm.disable_adapter():
        ref_lp = compute_completion_logprobs(lm, seq, prompt_len=3)
    kl = schulman_kl_divergence(lp, ref_lp)
    assert bool((kl >= -1e-7).all())
    assert float(schulman_kl_divergence(lp, lp).detach().abs().max().item()) < 1e-7

    # Build two 2-turn recursive traces (one winning, one losing)
    workers = [
        WorkerSpec(id=0, name="w0", backend="mock", model_name="m0", description="W0"),
        WorkerSpec(id=1, name="self", backend="conductor_self", model_name="self", description="self"),
    ]
    view = build_episode_worker_view(workers)
    router = WorkerRouter(mock_backend=MockWorkerBackend())
    orch = RecursiveOrchestrator(worker_router=router, max_steps=3, max_depth=2)

    async def _rec_gen(messages):
        if "Intermediate Outputs" not in messages[-1]["content"]:
            return "<plan>\nmodel_id = [0, 1]\nsubtasks = ['s0', 'replan']\naccess_list = [[], [0]]\n</plan>"
        return "<plan>\nmodel_id = [0]\nsubtasks = ['final']\naccess_list = [[]]\n</plan>"

    t1 = asyncio.run(orch.execute_episode(task_prompt="q", task_type="math", worker_view=view, conductor_generate_fn=_rec_gen))
    t2 = asyncio.run(orch.execute_episode(task_prompt="q", task_type="math", worker_view=view, conductor_generate_fn=_rec_gen))
    assert len(t1.conductor_turns) == 2

    r1 = compute_reward(t1, True, format_reward=0.2, cost_lambda=0.05, reference_cost_usd=0.01)
    r2 = compute_reward(t2, False, format_reward=0.2, cost_lambda=0.05, reference_cost_usd=0.01)
    assert r1 > r2

    # Compare top_level (2 turns) vs all_depth (4 turns)
    tok = _DummyTokenizer()
    tr_top = ConductorGRPOTrainer(lm, tok, GRPOConfig(credit_mode="top_level", learning_rate=1e-3))
    w_before = lm.adapter_proj.weight.detach().clone()
    stats_top = tr_top.update_from_groups([([t1, t2], [r1, r2])])
    assert stats_top["num_turns_updated"] == 2.0
    assert not torch.allclose(w_before, lm.adapter_proj.weight.detach())

    tr_all = ConductorGRPOTrainer(lm, tok, GRPOConfig(credit_mode="all_depth", learning_rate=1e-3))
    stats_all = tr_all.update_from_groups([([t1, t2], [r1, r2])])
    assert stats_all["num_turns_updated"] == 4.0

    # Save & reload checkpoint
    ckpt = tr_all.save_checkpoint(tmp_path / "ckpts", tag="latest")
    assert (ckpt / "trainer_state.json").exists()
    tr_all.load_optimizer_state(ckpt)
    assert tr_all.global_step == 1

    # Metrics & plotting smoke check
    summary = summarize_traces([t1, t2], [True, False], [r1, r2])
    assert summary["accuracy"] == 0.5
    assert summary["recursion_rate"] == 1.0

    m_file = tmp_path / "metrics.jsonl"
    m_file.write_text(
        json.dumps({"step": 1, **summary}) + "\n" + json.dumps({"step": 2, **summary}) + "\n",
        encoding="utf-8",
    )
    fig1 = plot_topology_evolution(m_file, tmp_path / "rq1.png")
    fig2 = plot_pareto_frontier(
        [
            {"label": "Haiku", "group": "single_model", "accuracy": 0.62, "avg_cost_usd": 0.001},
            {"label": "Sonnet", "group": "single_model", "accuracy": 0.78, "avg_cost_usd": 0.015},
            {"label": "mini-conductor (lam=0.1)", "group": "conductor", "accuracy": 0.81, "avg_cost_usd": 0.008},
        ],
        tmp_path / "rq4.png",
    )
    assert fig1.exists() and fig2.exists()
