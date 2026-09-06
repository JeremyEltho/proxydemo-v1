"""Routing configuration: the route table, model resolution and overrides.

The route table maps a *category* (what the request looks like) to an ordered
fallback chain of model names. At startup the chains are resolved against the
models actually installed locally, so an unavailable preference is skipped
rather than causing a runtime failure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Ordered preference per category. First installed model wins; the rest of the
# chain stays as the runtime fallback if that model errors out.
DEFAULT_ROUTES: Dict[str, List[str]] = {
    "code": ["qwen2.5-coder:3b", "qwen2.5-coder:7b", "qwen2.5:3b", "phi3:mini", "qwen2.5-coder:14b"],
    "math": ["qwen2.5:3b", "phi3:mini", "dolphin-phi"],
    "reasoning": ["qwen2.5:3b", "phi3:mini", "llama3.2:3b"],
    "summarize": ["qwen2.5:3b", "phi3:mini"],
    "long": ["qwen2.5:3b", "phi3:mini"],
    "chat": ["dolphin-phi", "phi3:mini", "qwen2.5:3b"],
}

# Per-category sampling defaults. Deterministic work gets a cold temperature.
DEFAULT_OPTIONS: Dict[str, Dict[str, Any]] = {
    "code": {"temperature": 0.1, "num_ctx": 8192},
    "math": {"temperature": 0.0, "num_ctx": 4096},
    "reasoning": {"temperature": 0.3, "num_ctx": 4096},
    "summarize": {"temperature": 0.2, "num_ctx": 8192},
    "long": {"temperature": 0.2, "num_ctx": 16384},
    "chat": {"temperature": 0.7, "num_ctx": 4096},
}

CATEGORIES: List[str] = list(DEFAULT_ROUTES)


@dataclass
class RouterConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    host: str = "127.0.0.1"
    port: int = 8080
    # "heuristic" (instant, no inference) or "llm" (a small local model votes).
    classifier: str = "heuristic"
    classifier_model: Optional[str] = None
    # Requests above this estimated token count are forced onto the "long" route.
    long_context_tokens: int = 1500
    request_timeout: int = 300
    routes: Dict[str, List[str]] = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_ROUTES.items()})
    options: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_OPTIONS.items()})

    # Populated by resolve(); category -> chain of installed models.
    resolved: Dict[str, List[str]] = field(default_factory=dict)
    available: List[str] = field(default_factory=list)
    # True when no backend was reachable and chains are declared-but-unverified
    # (the cloud deployment, where routing decisions work but generation cannot).
    offline: bool = False

    @classmethod
    def load(cls, path: Optional[str] = None) -> "RouterConfig":
        """Defaults, overlaid with a JSON config file, overlaid with env vars."""
        cfg = cls()
        path = path or os.environ.get("ROUTER_CONFIG", "router.config.json")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            for key, value in blob.items():
                if key in ("routes", "options") and isinstance(value, dict):
                    getattr(cfg, key).update(value)
                elif hasattr(cfg, key):
                    setattr(cfg, key, value)

        env = os.environ.get
        cfg.ollama_url = env("OLLAMA_URL", cfg.ollama_url).rstrip("/")
        cfg.host = env("ROUTER_HOST", cfg.host)
        cfg.port = int(env("ROUTER_PORT", cfg.port))
        cfg.classifier = env("ROUTER_CLASSIFIER", cfg.classifier)
        cfg.classifier_model = env("ROUTER_CLASSIFIER_MODEL", cfg.classifier_model or "") or None
        cfg.long_context_tokens = int(env("ROUTER_LONG_TOKENS", cfg.long_context_tokens))
        return cfg

    def resolve(self, available: List[Dict[str, Any]]) -> None:
        """Filter each chain down to models that are actually installed.

        `available` is Ollama's /api/tags payload. Chains that end up empty fall
        back to the smallest installed model, so the router still answers on a
        machine that has none of the preferred names.
        """
        names = [m["name"] for m in available]
        self.available = names
        by_size = sorted(available, key=lambda m: m.get("size", 0))
        smallest = [m["name"] for m in by_size[:3]]

        self.offline = False
        self.resolved = {}
        for category, chain in self.routes.items():
            picked = [n for n in chain if n in names]
            # Tolerate a bare family name ("qwen2.5") matching "qwen2.5:3b".
            for want in chain:
                if want in picked or ":" in want:
                    continue
                picked.extend(n for n in names if n.split(":")[0] == want and n not in picked)
            if not picked:
                picked = list(smallest)
            self.resolved[category] = picked

        if not self.classifier_model and smallest:
            self.classifier_model = smallest[0]

    def resolve_static(self) -> None:
        """Resolve without a backend: trust the declared chains, flag offline.

        Routing decisions stay meaningful (you can see which model *would* be
        used); generation is refused with a clear error until a backend exists.
        """
        self.offline = True
        self.available = []
        self.resolved = {c: list(chain) for c, chain in self.routes.items()}
        self.classifier_model = None

    def chain_for(self, category: str) -> List[str]:
        return self.resolved.get(category) or self.resolved.get("chat") or []

    def options_for(self, category: str) -> Dict[str, Any]:
        return dict(self.options.get(category, {}))

    def describe(self) -> Dict[str, Any]:
        return {
            "ollama_url": self.ollama_url,
            "offline": self.offline,
            "classifier": self.classifier,
            "classifier_model": self.classifier_model,
            "long_context_tokens": self.long_context_tokens,
            "routes": self.resolved or self.routes,
            "available_models": self.available,
        }
