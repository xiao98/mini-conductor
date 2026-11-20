"""Async worker backends (Anthropic Claude API, micro-batched local HuggingFace, Mock) + SQLite cache."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable

from src.config import PoolConfig, WorkerSpec


@dataclass(frozen=True)
class WorkerResponse:
    """Standardized response returned by any worker backend."""

    text: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    cached: bool = False


class SQLiteResponseCache:
    """Lightweight disk cache for deterministic worker calls to save API spend and wall-clock time."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT,
                    text TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL
                )
                """
            )
            self._conn.commit()

    @staticmethod
    def make_key(
        model_name: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            {
                "model": model_name,
                "messages": messages,
                "temperature": round(temperature, 4),
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> WorkerResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT model_name, text, input_tokens, output_tokens, cost_usd FROM responses WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return WorkerResponse(
            model_name=row[0],
            text=row[1],
            input_tokens=int(row[2]),
            output_tokens=int(row[3]),
            cost_usd=float(row[4]),
            latency_s=0.0,
            cached=True,
        )

    def put(self, cache_key: str, resp: WorkerResponse) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO responses
                (cache_key, model_name, text, input_tokens, output_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    resp.model_name,
                    resp.text,
                    resp.input_tokens,
                    resp.output_tokens,
                    resp.cost_usd,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class BaseWorkerBackend:
    """Abstract async interface for executing a worker call."""

    async def generate(
        self,
        spec: WorkerSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResponse:
        raise NotImplementedError


class AnthropicWorkerBackend(BaseWorkerBackend):
    """Async Claude API backend: semaphore throttling, backoff on retryable errors, SQLite cache.

    Current Claude models reject non-default sampling parameters, so `temperature` is never
    sent; `spec.temperature <= 0.25` only marks a worker as "deterministic intent" (cacheable).
    Thinking is on by default and billed inside `output_tokens`, so `max_tokens` must leave
    headroom and depth is controlled with `spec.effort`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_concurrency: int = 16,
        max_retries: int = 6,
        cache: SQLiteResponseCache | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            from anthropic import AsyncAnthropic

            # Retries are handled below (one place, one policy), so disable the SDK's own.
            client = AsyncAnthropic(api_key=api_key, max_retries=0)
        self._client = client
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._cache = cache
        self.stop_reasons: Counter[str] = Counter()  # surfaces silent truncation / refusals

    async def generate(
        self,
        spec: WorkerSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResponse:
        import anthropic

        temp = spec.temperature if temperature is None else temperature
        tok_limit = spec.max_tokens if max_tokens is None else max_tokens

        cache_key = None
        if self._cache is not None and temp <= 0.25:
            cache_key = self._cache.make_key(
                f"{spec.model_name}|effort={spec.effort}", messages, 0.0, tok_limit
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit

        kwargs: dict[str, Any] = {
            "model": spec.model_name,
            "max_tokens": tok_limit,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if system_text:
            kwargs["system"] = system_text
        if spec.effort:
            kwargs["output_config"] = {"effort": spec.effort}

        retryable = (
            anthropic.RateLimitError,
            anthropic.InternalServerError,  # all 5xx, including 529 overloaded
            anthropic.APIConnectionError,   # includes timeouts
        )
        t0 = time.perf_counter()
        for attempt in range(self._max_retries):
            try:
                async with self._sem:
                    msg = await self._client.messages.create(**kwargs)
                break
            except retryable:
                # 4xx (bad model ID, auth, invalid params) are NOT retried: they propagate at once.
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(min(60.0, 2.0**attempt) + random.uniform(0.0, 1.0))

        self.stop_reasons[str(msg.stop_reason)] += 1
        text = "\n".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()
        in_tok = int(msg.usage.input_tokens)
        out_tok = int(msg.usage.output_tokens)  # includes thinking tokens (really billed)
        resp = WorkerResponse(
            text=text,
            model_name=spec.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=spec.estimate_cost_usd(in_tok, out_tok),
            latency_s=time.perf_counter() - t0,
        )
        if cache_key is not None and self._cache is not None:
            self._cache.put(cache_key, resp)
        return resp


class BatchedGenerator:
    """Micro-batches concurrent local generation requests into single GPU `generate()` calls.

    All B*G rollouts (top-level plans, recursive inner plans, free local-worker calls) run
    concurrently under asyncio and submit here; requests sharing a `key` are flushed together.
    A single drain task also serializes GPU access, so no separate GPU lock is needed.
    """

    def __init__(
        self,
        batch_fn: Callable[[list[list[dict[str, str]]], tuple], list[tuple[str, int, int]]],
        *,
        max_batch: int = 16,
        wait_s: float = 0.02,
    ) -> None:
        self._batch_fn = batch_fn
        self._max_batch = max_batch
        self._wait_s = wait_s
        self._pending: list[tuple[tuple, list[dict[str, str]], asyncio.Future]] = []
        self._task: asyncio.Task | None = None
        self.batch_sizes: list[int] = []

    async def generate(self, messages: list[dict[str, str]], key: tuple) -> tuple[str, int, int]:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending.append((key, messages, fut))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())
        return await fut

    async def _drain(self) -> None:
        while self._pending:
            await asyncio.sleep(self._wait_s)  # let concurrent rollouts enqueue
            key = self._pending[0][0]
            idx = [i for i, p in enumerate(self._pending) if p[0] == key][: self._max_batch]
            taken = set(idx)
            batch = [self._pending[i] for i in idx]
            self._pending = [p for i, p in enumerate(self._pending) if i not in taken]
            self.batch_sizes.append(len(batch))
            try:
                outs = await asyncio.to_thread(self._batch_fn, [m for _, m, _ in batch], key)
                for (_, _, fut), out in zip(batch, outs):
                    fut.set_result(out)
            except Exception as exc:  # propagate to every waiting rollout
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)


class HFBatchGenerate:
    """`batch_fn` for BatchedGenerator over one or more local HF causal LMs.

    Key layout: `(model_key, use_adapter, temperature, max_new_tokens)`. With
    `use_adapter=False` the LoRA Conductor's frozen base weights are used through
    `model.disable_adapter()` -- the free local worker costs ZERO extra VRAM.
    """

    def __init__(self) -> None:
        self.models: dict[str, tuple[Any, Any]] = {}

    def add(self, model_key: str, model: Any, tokenizer: Any) -> None:
        tokenizer.padding_side = "left"  # required for batched decoder-only generation
        self.models[model_key] = (model, tokenizer)

    def __call__(
        self, batch_messages: list[list[dict[str, str]]], key: tuple
    ) -> list[tuple[str, int, int]]:
        import torch

        model_key, use_adapter, temperature, max_new_tokens = key
        model, tok = self.models[model_key]
        prompts = [
            tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        # add_special_tokens=False: the chat template already has them, and GRPO re-scores the
        # same prompt with add_special_tokens=False (sampling and scoring contexts must match).
        inputs = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        padded_len = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs.update(temperature=temperature, top_p=0.95)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                if not use_adapter and hasattr(model, "disable_adapter"):
                    with model.disable_adapter():
                        out_ids = model.generate(**inputs, **gen_kwargs)
                else:
                    out_ids = model.generate(**inputs, **gen_kwargs)
        finally:
            if was_training:
                model.train()

        results: list[tuple[str, int, int]] = []
        for i in range(len(prompts)):
            text = tok.decode(out_ids[i, padded_len:], skip_special_tokens=True).strip()
            in_tok = int(inputs["attention_mask"][i].sum())
            results.append((text, in_tok, len(tok.encode(text, add_special_tokens=False))))
        return results


class LocalHFWorkerBackend(BaseWorkerBackend):
    """Local HF worker routed through the shared BatchedGenerator."""

    def __init__(
        self,
        generator: BatchedGenerator,
        model_key: str,
        *,
        disable_adapter: bool = False,
        cache: SQLiteResponseCache | None = None,
    ) -> None:
        self._generator = generator
        self._model_key = model_key
        self._use_adapter = not disable_adapter
        self._cache = cache

    async def generate(
        self,
        spec: WorkerSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResponse:
        temp = spec.temperature if temperature is None else temperature
        tok_limit = spec.max_tokens if max_tokens is None else max_tokens

        cache_key = None
        if self._cache is not None and temp <= 0.25:
            cache_key = self._cache.make_key(spec.model_name, messages, temp, tok_limit)
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit

        t0 = time.perf_counter()
        text, in_tok, out_tok = await self._generator.generate(
            messages, (self._model_key, self._use_adapter, temp, tok_limit)
        )
        resp = WorkerResponse(
            text=text,
            model_name=spec.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=spec.estimate_cost_usd(in_tok, out_tok),
            latency_s=time.perf_counter() - t0,
        )
        if cache_key is not None and self._cache is not None:
            self._cache.put(cache_key, resp)
        return resp


class MockWorkerBackend(BaseWorkerBackend):
    """Deterministic mock backend for fast local unit tests and CI smoke checks."""

    def __init__(
        self,
        responder: Callable[[WorkerSpec, list[dict[str, str]]], str] | None = None,
        *,
        simulated_delay_s: float = 0.0,
    ) -> None:
        self._responder = responder
        self._delay = simulated_delay_s

    async def generate(
        self,
        spec: WorkerSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> WorkerResponse:
        t0 = time.perf_counter()
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._responder is not None:
            text = self._responder(spec, messages)
        else:
            text = f"[{spec.name}] processed subtask."
        in_tok = sum(len(m["content"].split()) for m in messages)
        out_tok = len(text.split())
        return WorkerResponse(
            text=text,
            model_name=spec.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=spec.estimate_cost_usd(in_tok, out_tok),
            latency_s=time.perf_counter() - t0,
            cached=False,
        )


class WorkerRouter:
    """Dispatches a WorkerSpec to the matching registered backend."""

    def __init__(
        self,
        *,
        anthropic_backend: BaseWorkerBackend | None = None,
        conductor_base_backend: BaseWorkerBackend | None = None,
        local_hf_backends: dict[str, BaseWorkerBackend] | None = None,
        mock_backend: BaseWorkerBackend | None = None,
    ) -> None:
        self.anthropic_backend = anthropic_backend
        self.conductor_base_backend = conductor_base_backend
        self.local_hf_backends = local_hf_backends or {}
        self.mock_backend = mock_backend

    async def call(
        self,
        spec: WorkerSpec,
        messages: list[dict[str, str]],
    ) -> WorkerResponse:
        if spec.backend == "anthropic":
            if self.anthropic_backend is None:
                raise RuntimeError("Anthropic backend requested but not configured.")
            return await self.anthropic_backend.generate(spec, messages)
        if spec.backend == "local_conductor_base":
            if self.conductor_base_backend is None:
                raise RuntimeError("Conductor base backend requested but not configured.")
            return await self.conductor_base_backend.generate(spec, messages)
        if spec.backend == "local_hf":
            backend = self.local_hf_backends.get(spec.model_name)
            if backend is None:
                raise RuntimeError(f"Local HF backend for '{spec.model_name}' not loaded.")
            return await backend.generate(spec, messages)
        if spec.backend == "mock":
            if self.mock_backend is None:
                raise RuntimeError("Mock backend requested but not configured.")
            return await self.mock_backend.generate(spec, messages)
        raise ValueError(f"Unsupported concrete worker backend: {spec.backend}")

CONDUCTOR_KEY = "conductor"


def build_router(
    pool_cfg: PoolConfig,
    *,
    generator: BatchedGenerator | None = None,
    hf: HFBatchGenerate | None = None,
    cache: SQLiteResponseCache | None = None,
    max_api_concurrency: int = 16,
) -> WorkerRouter:
    """Instantiate only the backends this pool (including held-out workers) needs."""
    specs = pool_cfg.active_workers(include_held_out=True, allow_recursion=False)
    needed = {w.backend for w in specs}

    local_hf: dict[str, BaseWorkerBackend] = {}
    if "local_hf" in needed:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for w in specs:
            if w.backend == "local_hf" and w.model_name not in local_hf:
                print(f"[worker] Loading local HF specialist: {w.model_name}")
                w_tok = AutoTokenizer.from_pretrained(w.model_name, use_fast=True)
                if w_tok.pad_token is None:
                    w_tok.pad_token = w_tok.eos_token
                w_mod = AutoModelForCausalLM.from_pretrained(
                    w.model_name, torch_dtype=torch.bfloat16, device_map="auto"
                ).eval()
                hf.add(w.model_name, w_mod, w_tok)
                local_hf[w.model_name] = LocalHFWorkerBackend(generator, w.model_name, cache=cache)

    return WorkerRouter(
        anthropic_backend=(
            AnthropicWorkerBackend(max_concurrency=max_api_concurrency, cache=cache)
            if "anthropic" in needed
            else None
        ),
        conductor_base_backend=(
            LocalHFWorkerBackend(generator, CONDUCTOR_KEY, disable_adapter=True, cache=cache)
            if "local_conductor_base" in needed
            else None
        ),
        local_hf_backends=local_hf,
        mock_backend=MockWorkerBackend() if "mock" in needed else None,
    )
