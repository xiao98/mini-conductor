"""Preflight: send ONE minimal request to every Claude model the pool config references.

Goes through the real `AnthropicWorkerBackend` code path (no cache), so a pass here means the
exact request shape used in training is accepted for this account. Exits non-zero on failure.

    python scripts/00_check_api.py --pool configs/pool_claude_hybrid.yaml
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_pool_config
from src.workers import AnthropicWorkerBackend


async def main(pool_path: str) -> int:
    pool = load_pool_config(pool_path)
    specs = [
        w
        for w in pool.active_workers(include_held_out=True, allow_recursion=False)
        if w.backend == "anthropic"
    ]
    backend = AnthropicWorkerBackend(max_retries=2)

    try:
        ids = [m.id async for m in backend._client.models.list()]
        print(f"[models] {len(ids)} model IDs visible to this account:\n  " + "\n  ".join(ids))
    except Exception as exc:  # listing is informational only
        print(f"[models] Could not list models ({type(exc).__name__}: {exc})")

    failures = 0
    msgs = [{"role": "user", "content": "Reply with exactly: OK"}]
    for spec in specs:
        try:
            resp = await backend.generate(spec, msgs, max_tokens=1024)
            print(
                f"[ok]   {spec.name:14s} model={spec.model_name:20s} effort={spec.effort} "
                f"reply={resp.text!r} in={resp.input_tokens} out={resp.output_tokens} "
                f"cost=${resp.cost_usd:.6f} {resp.latency_s:.1f}s"
            )
            if not resp.text:
                failures += 1
                print(f"[FAIL] {spec.name}: empty text (stop_reasons={dict(backend.stop_reasons)})")
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {spec.name:14s} model={spec.model_name}: {type(exc).__name__}: {exc}")
    print(f"[done] {len(specs) - failures}/{len(specs)} Claude workers usable.")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="configs/pool_claude_hybrid.yaml")
    sys.exit(asyncio.run(main(ap.parse_args().pool)))
