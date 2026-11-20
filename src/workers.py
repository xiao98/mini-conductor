"""Async worker backends (Anthropic Claude API, local HuggingFace, shared LoRA base, Mock) + SQLite cache."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable

from src.config import WorkerSpec


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
    """Async Anthropic Claude API backend with semaphore throttling, retries, prompt caching, and SQLite cache."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_concurrency: int = 16,
        max_retries: int = 5,
        cache: SQLiteResponseCache | None = None,
        enable_prompt_caching: bool = True,
    ) -> None:
        from anthropic import AsyncAnthropic

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self._client = AsyncAnthropic(api_key=resolved_key)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._cache = cache
        self._enable_prompt_caching = enable_prompt_caching

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

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] != "system"
        ]

        system_payload: Any = "\n\n".join(system_parts) if system_parts else None
        if system_payload and self._enable_prompt_caching and len(system_payload) > 512:
            system_payload = [
                {
                    "type": "text",
                    "text": system_payload,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            t0 = time.perf_counter()
            try:
                async with self._sem:
                    kwargs: dict[str, Any] = {
                        "model": spec.model_name,
                        "max_tokens": tok_limit,
                        "temperature": temp,
                        "messages": chat_msgs,
                    }
                    if system_payload is not None:
                        kwargs["system"] = system_payload
                    msg = await self._client.messages.create(**kwargs)

                text_blocks = [
                    b.text for b in msg.content if getattr(b, "type", None) == "text"
                ]
                text = "\n".join(text_blocks).strip()
                in_tok = int(getattr(msg.usage, "input_tokens", 0))
                out_tok = int(getattr(msg.usage, "output_tokens", 0))
                resp = WorkerResponse(
                    text=text,
                    model_name=spec.model_name,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=spec.estimate_cost_usd(in_tok, out_tok),
                    latency_s=time.perf_counter() - t0,
                    cached=False,
                )
                if cache_key is not None and self._cache is not None:
                    self._cache.put(cache_key, resp)
                return resp
            except Exception as exc:
                last_err = exc
                sleep_s = min(30.0, (2.0**attempt) * 0.75)
                await asyncio.sleep(sleep_s)

        raise RuntimeError(
            f"Anthropic call failed for {spec.model_name} after {self._max_retries} attempts: {last_err}"
        )


class LocalHFWorkerBackend(BaseWorkerBackend):
    """Local HuggingFace causal LM backend.

    When `disable_adapter=True`, shares the trainable LoRA Conductor model's frozen base
    weights with ZERO extra GPU VRAM!
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        disable_adapter: bool = False,
        cache: SQLiteResponseCache | None = None,
        gpu_lock: asyncio.Lock | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.disable_adapter = disable_adapter
        self._cache = cache
        self._gpu_lock = gpu_lock or asyncio.Lock()

    def _sync_generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        import torch

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_len = int(inputs["input_ids"].shape[1])

        was_training = self.model.training
        self.model.eval()
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        try:
            with torch.no_grad():
                if self.disable_adapter and hasattr(self.model, "disable_adapter"):
                    with self.model.disable_adapter():
                        out_ids = self.model.generate(**inputs, **gen_kwargs)
                else:
                    out_ids = self.model.generate(**inputs, **gen_kwargs)
        finally:
            if was_training:
                self.model.train()

        new_tokens = out_ids[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, prompt_len, int(new_tokens.shape[0])

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
        async with self._gpu_lock:
            text, in_tok, out_tok = await asyncio.to_thread(
                self._sync_generate, messages, temp, tok_limit
            )

        resp = WorkerResponse(
            text=text,
            model_name=spec.model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=spec.estimate_cost_usd(in_tok, out_tok),
            latency_s=time.perf_counter() - t0,
            cached=False,
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
