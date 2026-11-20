"""Configuration models and Worker ID-Permutation helpers for mini-conductor."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from pathlib import Path
from typing import Literal

import yaml


BackendType = Literal[
    "anthropic",
    "local_hf",
    "local_conductor_base",
    "conductor_self",
    "mock",
]


@dataclass(frozen=True)
class WorkerSpec:
    """Declarative specification for a single worker in the Conductor's pool."""

    id: int
    name: str
    backend: BackendType
    model_name: str
    description: str
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    cost_tier: str = "free"
    max_tokens: int = 1536
    temperature: float = 0.2

    @property
    def is_self(self) -> bool:
        return self.backend == "conductor_self"

    def estimate_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000.0) * self.input_cost_per_mtok
            + (output_tokens / 1_000_000.0) * self.output_cost_per_mtok
        )


@dataclass
class PoolConfig:
    """Complete worker pool configuration loaded from YAML."""

    name: str
    workers: list[WorkerSpec]
    held_out_workers: list[WorkerSpec] = field(default_factory=list)
    reference_cost_usd: float = 0.015
    max_steps: int = 5
    max_depth: int = 2

    def active_workers(
        self,
        *,
        include_held_out: bool = False,
        allow_recursion: bool = True,
    ) -> list[WorkerSpec]:
        pool = list(self.workers)
        if include_held_out:
            pool.extend(self.held_out_workers)
        if not allow_recursion:
            pool = [w for w in pool if not w.is_self]
        return pool


@dataclass(frozen=True)
class EpisodeWorkerView:
    """Per-episode view of workers with either canonical or randomly permuted IDs.

    Supporting ID-Permutation Augmentation (RQ3) forces the Conductor policy to read
    natural-language worker descriptions instead of memorizing fixed integer IDs.
    """

    ordered_specs: list[WorkerSpec]
    id_to_worker: dict[int, WorkerSpec]
    name_to_id: dict[str, int]
    is_permuted: bool = False

    @property
    def valid_ids(self) -> set[int]:
        return set(self.id_to_worker.keys())

    @property
    def self_id(self) -> int | None:
        for wid, spec in self.id_to_worker.items():
            if spec.is_self:
                return wid
        return None


def build_episode_worker_view(
    workers: list[WorkerSpec],
    *,
    permute_ids: bool = False,
    rng: random.Random | None = None,
) -> EpisodeWorkerView:
    """Create a canonical or ID-permuted mapping for a single episode/prompt."""
    if not workers:
        raise ValueError("Worker pool must not be empty.")

    if not permute_ids:
        id_map = {w.id: w for w in workers}
        if len(id_map) != len(workers):
            raise ValueError("Duplicate worker IDs found in pool config.")
        ordered = sorted(workers, key=lambda w: w.id)
        return EpisodeWorkerView(
            ordered_specs=ordered,
            id_to_worker=id_map,
            name_to_id={w.name: w.id for w in ordered},
            is_permuted=False,
        )

    gen = rng if rng is not None else random.Random()
    shuffled = list(workers)
    gen.shuffle(shuffled)
    available_ids = list(range(len(shuffled)))
    gen.shuffle(available_ids)

    remapped: list[WorkerSpec] = []
    for assigned_id, spec in zip(available_ids, shuffled):
        remapped.append(
            WorkerSpec(
                id=assigned_id,
                name=spec.name,
                backend=spec.backend,
                model_name=spec.model_name,
                description=spec.description,
                input_cost_per_mtok=spec.input_cost_per_mtok,
                output_cost_per_mtok=spec.output_cost_per_mtok,
                cost_tier=spec.cost_tier,
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
            )
        )

    remapped.sort(key=lambda w: w.id)
    return EpisodeWorkerView(
        ordered_specs=remapped,
        id_to_worker={w.id: w for w in remapped},
        name_to_id={w.name: w.id for w in remapped},
        is_permuted=True,
    )


def load_pool_config(path: str | Path) -> PoolConfig:
    """Load a PoolConfig from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    workers = [WorkerSpec(**w) for w in raw.get("workers", [])]
    held_out = [WorkerSpec(**w) for w in raw.get("held_out_workers", [])]
    return PoolConfig(
        name=raw["name"],
        workers=workers,
        held_out_workers=held_out,
        reference_cost_usd=float(raw.get("reference_cost_usd", 0.015)),
        max_steps=int(raw.get("max_steps", 5)),
        max_depth=int(raw.get("max_depth", 2)),
    )
