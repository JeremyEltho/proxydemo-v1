"""Thread-safe counters for routing and backend behaviour."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict


class Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = 0
        self.errors = 0
        self.fallbacks = 0
        self.by_category: Dict[str, int] = defaultdict(int)
        self.by_model: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"requests": 0, "errors": 0, "prompt_tokens": 0,
                     "completion_tokens": 0, "total_latency_ms": 0.0}
        )

    def record(self, category: str, model: str, latency_ms: float,
               prompt_tokens: int = 0, completion_tokens: int = 0,
               error: bool = False, fallbacks: int = 0) -> None:
        with self._lock:
            self.requests += 1
            self.by_category[category] += 1
            self.fallbacks += fallbacks
            entry = self.by_model[model]
            entry["requests"] += 1
            entry["total_latency_ms"] += latency_ms
            entry["prompt_tokens"] += prompt_tokens
            entry["completion_tokens"] += completion_tokens
            if error:
                self.errors += 1
                entry["errors"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            models = {}
            for name, entry in self.by_model.items():
                requests = entry["requests"] or 1
                models[name] = {
                    **{k: int(v) if k != "total_latency_ms" else round(v, 1)
                       for k, v in entry.items()},
                    "avg_latency_ms": round(entry["total_latency_ms"] / requests, 1),
                }
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "requests": self.requests,
                "errors": self.errors,
                "fallbacks": self.fallbacks,
                "by_category": dict(self.by_category),
                "by_model": models,
            }
