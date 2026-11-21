"""Tests for the real-run plumbing: Claude backend request shape & retry policy, GPU micro-batching,
recursion bounds, strict completion masking, resume determinism, and the RQ1-RQ4 experiment matrix."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import WorkerSpec, build_episode_worker_view, load_pool_config
from src.executor import RecursiveOrchestrator
from src.grpo import ConductorGRPOTrainer, GRPOConfig, compute_completion_logprobs, compute_reward
from src.parser import parse_and_validate_plan
from src.verifiers import verify_answer
from src.workers import (
    AnthropicWorkerBackend,
    BatchedGenerator,
    MockWorkerBackend,
    SQLiteResponseCache,
    WorkerRouter,
)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Claude backend: request shape, retry policy, cache
# ---------------------------------------------------------------------------

def _api_error(cls, status: int):
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    resp = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return cls("simulated", response=resp, body=None)


class _FakeMessages:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fake_msg(text: str = "42", stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        stop_reason=stop_reason,
    )


SONNET = WorkerSpec(
    id=2, name="claude-sonnet", backend="anthropic", model_name="claude-sonnet-5-5", description="d",
    input_cost_per_mtok=2.0, output_cost_per_mtok=10.0, max_tokens=8000, temperature=0.2, effort="low",
)
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]


@pytest.fixture
def no_sleep(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _s: real_sleep(0))


def test_anthropic_request_shape_never_sends_sampling_params(tmp_path):
    fake = _FakeMessages([_fake_msg()])
    cache = SQLiteResponseCache(tmp_path / "c.sqlite")
    backend = AnthropicWorkerBackend(client=SimpleNamespace(messages=fake), cache=cache)

    r1 = asyncio.run(backend.generate(SONNET, MSGS))
    kw = fake.calls[0]
    assert "temperature" not in kw and "top_p" not in kw  # rejected (400) by current Claude models
    assert kw["model"] == "claude-sonnet-5-5" and kw["max_tokens"] == 8000
    assert kw["output_config"] == {"effort": "low"}
    assert kw["system"] == "sys" and kw["messages"] == [{"role": "user", "content": "q"}]
    assert r1.text == "42" and r1.cost_usd == pytest.approx(100 * 2e-6 + 50 * 10e-6)
    assert backend.stop_reasons == {"end_turn": 1}

    # Deterministic-intent worker (temperature <= 0.25) is served from SQLite the second time.
    r2 = asyncio.run(backend.generate(SONNET, MSGS))
    assert r2.cached and r2.text == "42" and len(fake.calls) == 1
    cache.close()


def test_anthropic_retries_only_retryable_errors(no_sleep):
    import anthropic

    fake = _FakeMessages(
        [_api_error(anthropic.RateLimitError, 429), _api_error(anthropic.InternalServerError, 529), _fake_msg("ok")]
    )
    backend = AnthropicWorkerBackend(client=SimpleNamespace(messages=fake))
    assert asyncio.run(backend.generate(SONNET, MSGS)).text == "ok"
    assert len(fake.calls) == 3

    # A 400 (e.g. wrong model ID / invalid parameter) must fail immediately, not burn retries.
    fake_bad = _FakeMessages([_api_error(anthropic.BadRequestError, 400), _fake_msg()])
    backend_bad = AnthropicWorkerBackend(client=SimpleNamespace(messages=fake_bad))
    with pytest.raises(anthropic.BadRequestError):
        asyncio.run(backend_bad.generate(SONNET, MSGS))
    assert len(fake_bad.calls) == 1

    # Retry budget is bounded.
    fake_down = _FakeMessages([_api_error(anthropic.RateLimitError, 429)] * 3)
    backend_down = AnthropicWorkerBackend(client=SimpleNamespace(messages=fake_down), max_retries=3)
    with pytest.raises(anthropic.RateLimitError):
        asyncio.run(backend_down.generate(SONNET, MSGS))
    assert len(fake_down.calls) == 3


def test_api_concurrency_is_bounded_by_semaphore():
    live = peak = 0

    class _Slow:
        async def create(self, **kwargs):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1
            return _fake_msg()

    async def _run():
        backend = AnthropicWorkerBackend(client=SimpleNamespace(messages=_Slow()), max_concurrency=3)
        hot = WorkerSpec(id=0, name="t", backend="anthropic", model_name="m", description="d", temperature=0.6)
        await asyncio.gather(*(backend.generate(hot, MSGS) for _ in range(12)))

    asyncio.run(_run())
    assert peak == 3


def test_pool_config_claude_workers_pin_effort_and_headroom():
    pool = load_pool_config(ROOT / "configs" / "pool_claude_hybrid.yaml")
    claude = [w for w in pool.active_workers(include_held_out=True) if w.backend == "anthropic"]
    assert {w.model_name for w in claude} == {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}
    assert all(w.effort == "low" and w.max_tokens >= 4000 for w in claude)
    assert [w.name for w in pool.held_out_workers] == ["claude-opus"]
    # ID permutation must carry every field (incl. effort) over to the remapped spec.
    view = build_episode_worker_view(pool.active_workers(), permute_ids=True, rng=random.Random(0))
    assert view.id_to_worker[view.name_to_id["claude-sonnet"]].effort == "low"


# ---------------------------------------------------------------------------
# GPU micro-batching
# ---------------------------------------------------------------------------

def test_batched_generator_groups_by_key_and_routes_results():
    seen: list[tuple[tuple, int]] = []

    def batch_fn(batch_messages, key):
        seen.append((key, len(batch_messages)))
        return [(f"{key[0]}:{m[0]['content']}", 1, 1) for m in batch_messages]

    async def _run():
        gen = BatchedGenerator(batch_fn, max_batch=4, wait_s=0.01)
        reqs = [(("A" if i % 3 else "B",), [{"role": "user", "content": str(i)}]) for i in range(11)]
        outs = await asyncio.gather(*(gen.generate(m, k) for k, m in reqs))
        return reqs, outs, gen

    reqs, outs, gen = asyncio.run(_run())
    for (key, msgs), (text, _, _) in zip(reqs, outs):
        assert text == f"{key[0]}:{msgs[0]['content']}"  # each caller gets ITS OWN result
    assert sum(n for _, n in seen) == 11 and max(n for _, n in seen) <= 4
    assert len(seen) < 11  # actually batched
    assert gen.batch_sizes == [n for _, n in seen]


def test_batched_generator_propagates_errors():
    def boom(batch_messages, key):
        raise RuntimeError("cuda oom")

    async def _run():
        gen = BatchedGenerator(boom, wait_s=0.0)
        return await asyncio.gather(*(gen.generate([], ("k",)) for _ in range(3)), return_exceptions=True)

    assert all(isinstance(e, RuntimeError) for e in asyncio.run(_run()))


# ---------------------------------------------------------------------------
# Parser / recursion bounds / sandbox
# ---------------------------------------------------------------------------

def _plan(ids, subs, acc) -> str:
    return f"<plan>\nmodel_id = {ids!r}\nsubtasks = {subs!r}\naccess_list = {acc!r}\n</plan>"


@pytest.mark.parametrize(
    "ids,subs,acc",
    [
        ([0, 1], ["a", "b"], [[1], []]),          # forward reference (j >= i)
        ([0, 1], ["a", "b"], [[], [1]]),          # self reference
        ([0, 1], ["a", "b"], [[], [-1]]),         # negative index
        ([0, 1], ["a"], [[], [0]]),               # length mismatch
        ([0] * 6, ["a"] * 6, [[]] * 6),           # exceeds max_steps
        ([True], ["a"], [[]]),                    # bool is not a worker id
        ([9], ["a"], [[]]),                       # unknown worker id
        ([0], [""], [[]]),                        # empty subtask
        ([], [], []),                             # empty plan
    ],
)
def test_parser_rejects_invalid_plans(ids, subs, acc):
    assert not parse_and_validate_plan(_plan(ids, subs, acc), valid_worker_ids={0, 1}, max_steps=5).ok


def test_recursion_is_bounded_by_max_depth():
    workers = [
        WorkerSpec(id=0, name="w0", backend="mock", model_name="m0", description="W0"),
        WorkerSpec(id=1, name="self", backend="conductor_self", model_name="self", description="self"),
    ]
    view = build_episode_worker_view(workers)
    calls = 0

    async def _always_recurse(messages):
        nonlocal calls
        calls += 1
        return _plan([0, 1], ["solve", "replan"], [[], [0]])

    for max_depth in (1, 2, 3):
        calls = 0
        orch = RecursiveOrchestrator(
            worker_router=WorkerRouter(mock_backend=MockWorkerBackend()), max_steps=3, max_depth=max_depth
        )
        trace = asyncio.run(
            orch.execute_episode(task_prompt="q", task_type="math", worker_view=view, conductor_generate_fn=_always_recurse)
        )
        # The deepest allowed level may not call `self`, so an always-recursing policy is invalid
        # there, gets zero reward, and is called at most `max_depth` times (never unbounded).
        assert calls == max_depth
        assert not trace.format_valid
        assert compute_reward(trace, True) == 0.0


def test_code_sandbox_stdin_stdout_mode_and_isolation():
    code = "```python\nimport sys\nprint(sum(int(x) for x in sys.stdin.read().split()))\n```"
    meta = {"io_tests": [{"input": "1 2 3", "output": "6"}, {"input": "10 -4", "output": "6"}]}
    assert verify_answer("code", code, "", metadata=meta)[0]
    assert not verify_answer("code", code, "", metadata={"io_tests": [{"input": "1 2", "output": "4"}]})[0]
    # A crash in the candidate program stays inside the child process.
    ok, reason = verify_answer("code", "```python\nraise SystemExit(3)\n```", "assert True")
    assert not ok and reason.startswith("test_failed")


# ---------------------------------------------------------------------------
# GRPO: strict completion-token masking
# ---------------------------------------------------------------------------

class _CtxLM(torch.nn.Module):
    """Causal LM whose logits at position t depend on ALL tokens <= t (prefix mean)."""

    def __init__(self, vocab: int = 32, d: int = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.drop = torch.nn.Dropout(0.3)
        self.head = torch.nn.Linear(d, vocab)

    def forward(self, input_ids, use_cache=False):
        h = self.embed(input_ids)
        prefix_mean = h.cumsum(1) / torch.arange(1, h.shape[1] + 1).view(1, -1, 1)
        return SimpleNamespace(logits=self.head(self.drop(prefix_mean)))


def test_logprobs_and_gradients_cover_only_completion_tokens():
    torch.manual_seed(0)
    lm = _CtxLM().eval()
    ids = torch.tensor([[3, 7, 11, 15, 19, 23, 5]])
    prompt_len = 4
    lp = compute_completion_logprobs(lm, ids, prompt_len)
    assert lp.shape == (1, ids.shape[1] - prompt_len)

    # Reference: full per-position logprobs; the function must return exactly the completion slice.
    full = torch.log_softmax(lm(ids).logits[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(lp, full[:, prompt_len - 1 :], atol=1e-6)

    # Strict mask: gradient reaches the logits ONLY at the positions that predict completion
    # tokens, i.e. [prompt_len-1, T-2]. Prompt positions and the final position get exactly 0.
    captured = {}

    class _Capture(torch.nn.Module):
        def forward(self, input_ids, use_cache=False):
            out = lm(input_ids)
            out.logits.retain_grad()
            captured["logits"] = out.logits
            return out

    compute_completion_logprobs(_Capture(), ids, prompt_len).sum().backward()
    per_pos = captured["logits"].grad.abs().sum(-1)[0]
    assert torch.all(per_pos[: prompt_len - 1] == 0.0) and per_pos[-1] == 0.0
    assert torch.all(per_pos[prompt_len - 1 : -1] > 0.0)

    with pytest.raises(ValueError):
        compute_completion_logprobs(lm, ids, prompt_len=ids.shape[1])  # no completion tokens


def test_trainer_zeroes_dropout_so_initial_ratio_is_one():
    lm = _CtxLM()

    class _Tok:
        eos_token_id = 1

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "".join(m["content"] for m in messages)

        def encode(self, text, add_special_tokens=False):
            return [(ord(c) % 28) + 2 for c in text[:20]]

    trainer = ConductorGRPOTrainer(lm, _Tok(), GRPOConfig(kl_beta=0.0, learning_rate=0.0))
    assert lm.drop.p == 0.0

    workers = [WorkerSpec(id=0, name="w0", backend="mock", model_name="m0", description="W0")]
    orch = RecursiveOrchestrator(worker_router=WorkerRouter(mock_backend=MockWorkerBackend()), max_depth=1)

    async def _gen(messages):
        return _plan([0], ["s"], [[]])

    traces = [
        asyncio.run(orch.execute_episode(task_prompt="q", task_type="math",
                                         worker_view=build_episode_worker_view(workers), conductor_generate_fn=_gen))
        for _ in range(2)
    ]
    stats = trainer.update_from_groups([(traces, [1.0, 0.2])])
    # pi_theta == pi_old at the first step => ratio == 1 => nothing clipped, pg_loss == -mean(A) == 0
    assert stats["num_turns_updated"] == 2.0
    assert stats["clip_frac"] == 0.0 and abs(stats["pg_loss"]) < 1e-6


# ---------------------------------------------------------------------------
# Resume determinism (Colab disconnects)
# ---------------------------------------------------------------------------

def test_resume_helpers(tmp_path):
    train = load_script("04_train_grpo.py")
    pool = list(range(100))
    assert train.step_rng(42, 7).sample(pool, 8) == train.step_rng(42, 7).sample(pool, 8)
    assert train.step_rng(42, 7).sample(pool, 8) != train.step_rng(42, 8).sample(pool, 8)

    metrics = tmp_path / "train_metrics.jsonl"
    metrics.write_text("".join(json.dumps({"step": s}) + "\n" for s in range(1, 8)), encoding="utf-8")
    train.truncate_metrics(metrics, last_step=5)  # checkpoint was at step 5; 6-7 will be re-run
    assert [json.loads(l)["step"] for l in metrics.read_text(encoding="utf-8").splitlines()] == [1, 2, 3, 4, 5]

    lm = _CtxLM()
    trainer = ConductorGRPOTrainer(lm, SimpleNamespace(), GRPOConfig())
    trainer.global_step = 3
    train.save_latest_atomic(trainer, tmp_path)
    trainer.global_step = 4
    train.save_latest_atomic(trainer, tmp_path)  # overwrite path
    assert not (tmp_path / "latest.tmp").exists()
    assert json.loads((tmp_path / "latest" / "trainer_state.json").read_text())["global_step"] == 4


# ---------------------------------------------------------------------------
# Eval suite helpers & RQ matrix
# ---------------------------------------------------------------------------

def test_eval_stratified_limit_and_static_plan():
    ev = load_script("05_eval_suite.py")
    rows = [{"id": f"{t}{i}", "task_type": t} for t in ("math", "code", "gpqa") for i in range(50)]
    sub = ev.stratified_limit(rows, 30, seed=1)
    assert len(sub) == 30 and {t: sum(r["task_type"] == t for r in sub) for t in ("math", "code", "gpqa")} == {
        "math": 10, "code": 10, "gpqa": 10}
    assert [r["id"] for r in sub] == [r["id"] for r in ev.stratified_limit(list(rows), 30, seed=1)]
    assert ev.stratified_limit(rows, 0, seed=1) == rows

    pool = load_pool_config(ROOT / "configs" / "pool_claude_hybrid.yaml")
    for permute in (False, True):
        view = build_episode_worker_view(
            pool.active_workers(allow_recursion=False), permute_ids=permute, rng=random.Random(3)
        )
        res = parse_and_validate_plan(ev.static_parallel_judge_plan(view), valid_worker_ids=view.valid_ids)
        assert res.ok and res.plan.access_lists[-1] == [0, 1, 2]
        assert view.id_to_worker[res.plan.model_ids[-1]].name == view.ordered_specs[-1].name


def test_matrix_runs_differ_from_base_by_exactly_one_switch():
    mx = load_script("06_run_matrix.py")
    assert set(mx.RUNS) == {
        "base", "rq2_no_recursion", "rq2_top_level", "rq3_perm", "rq4_lam0.05", "rq4_lam0.15", "rq4_lam0.30"}
    base = mx.run_spec("base")
    for name in mx.RUNS:
        diff = {k for k in base if mx.run_spec(name)[k] != base[k]}
        assert len(diff) == (0 if name == "base" else 1), (name, diff)

    args = mx.build_arg_parser().parse_args(["--root", "exp"])
    t = mx.train_cmd("rq2_no_recursion", args)
    assert "--allow_recursion" not in t and "--resume" in t and "--permute_ids" not in t
    assert "--permute_ids" in mx.train_cmd("rq3_perm", args)
    assert mx.train_cmd("rq2_top_level", args)[mx.train_cmd("rq2_top_level", args).index("--credit_mode") + 1] == "top_level"
    # Train/eval conditions must match per run.
    assert "--no-allow_recursion" in mx.eval_cmd("rq2_no_recursion", args)
    assert "--cost_aware" in mx.eval_cmd("rq4_lam0.15", args) and "--cost_aware" not in mx.eval_cmd("base", args)
    assert mx.eval_cmd(None, args)[-2:] == ["--baselines", "--skip_conductor"]
    # The CLI flags the matrix emits are all accepted by the real train/eval parsers.
    load_script("04_train_grpo.py").build_arg_parser().parse_args(t[2:])
    load_script("05_eval_suite.py").build_arg_parser().parse_args(mx.eval_cmd("rq4_lam0.15", args)[2:])
    load_script("05_eval_suite.py").build_arg_parser().parse_args(mx.eval_cmd(None, args)[2:])


def test_matrix_figures_never_invent_missing_runs(tmp_path):
    mx = load_script("06_run_matrix.py")
    assert mx.make_figures(tmp_path) == []  # nothing ran => no figure at all
    assert json.loads((tmp_path / "figures" / "all_results.json").read_text()) == {"baselines": []}

    # Plotting-code fixture (NOT results): only `base` has an eval summary.
    cell = {"label": "base", "group": "conductor", "condition": "canonical", "accuracy": 0.5, "acc_ci95": 0.1,
            "n": 10, "avg_cost_usd": 0.001, "by_task_type": {"math": {"accuracy": 0.5, "acc_ci95": 0.1, "n": 10}}}
    (tmp_path / "eval" / "base").mkdir(parents=True)
    (tmp_path / "eval" / "base" / "eval_summary.json").write_text(json.dumps({"results": [cell]}))
    made = {p.name for p in mx.make_figures(tmp_path)}
    assert made == {"rq2_credit_assignment.png", "rq3_worker_transfer.png", "rq4_pareto_frontier.png"}
    assert set(json.loads((tmp_path / "figures" / "all_results.json").read_text())) == {"baselines", "base"}


def test_math_verifier_accepts_equivalent_surface_forms():
    """Pairs observed as false negatives in a real baseline run (MATH-500, Claude workers)."""
    from src.verifiers import verify_math

    for pred, gt in [
        (r"(3, \frac{\pi}{2})", r"\left( 3, \frac{\pi}{2} \right)"),
        (r"\frac{4}{3}", r"\frac43"),
        ("58500", "58,500"),
        ("5.4", r"5.4 \text{ cents}"),
        ("120", r"120^\circ"),
        ("(15, -29)", "(15,-29)"),
    ]:
        assert verify_math(rf"So the answer is \boxed{{{pred}}}.", gt)[0], (pred, gt)
    for pred, gt in [("58501", "58,500"), (r"\frac{3}{4}", r"\frac43"), ("(15, 29)", "(15,-29)"), (r"\text{west}", r"\text{east}")]:
        assert not verify_math(rf"\boxed{{{pred}}}", gt)[0], (pred, gt)


def test_default_pool_trains_on_local_workers_only():
    pool = load_pool_config(ROOT / "configs" / "pool_local_weak.yaml")
    train = pool.active_workers(include_held_out=False, allow_recursion=False)
    assert {w.backend for w in train} == {"local_hf", "local_conductor_base"}  # RL makes no API calls
    assert [w.name for w in pool.held_out_workers] == ["claude-haiku"]
    ids = [w.id for w in pool.active_workers(include_held_out=True)]
    assert ids == list(range(len(ids)))
