"""The router itself: pick a model, then run it with a fallback chain."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .backends import BackendError, OllamaBackend
from .classifier import Decision, classify, prompt_text
from .config import CATEGORIES, RouterConfig
from .stats import Stats


class RoutingError(ValueError):
    """The request cannot be routed (bad model name, no models installed)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Plan:
    decision: Decision
    chain: List[str]
    options: Dict[str, Any]
    mode: str  # "auto" | "pinned" | "category"

    @property
    def model(self) -> str:
        return self.chain[0]


# OpenAI request field -> Ollama option field
PARAM_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "num_predict",
    "max_completion_tokens": "num_predict",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}


class Router:
    def __init__(self, cfg: RouterConfig, backend: Optional[OllamaBackend] = None,
                 stats: Optional[Stats] = None):
        self.cfg = cfg
        self.backend = backend or OllamaBackend(cfg.ollama_url, cfg.request_timeout)
        self.stats = stats or Stats()

    # -- setup ------------------------------------------------------------
    def refresh(self, strict: bool = True) -> Dict[str, Any]:
        """Re-read installed models and re-resolve every route chain.

        With strict=False an unreachable backend degrades to declared-only
        chains instead of raising, so /route keeps working without models.
        """
        try:
            self.cfg.resolve(self.backend.list_models())
        except BackendError:
            if strict:
                raise
            self.cfg.resolve_static()
        return self.cfg.describe()

    # -- decision ---------------------------------------------------------
    def plan(self, messages: List[Dict[str, Any]], requested: Optional[str] = None,
             hint: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Plan:
        requested = (requested or "auto").strip()
        text = prompt_text(messages)

        if requested in ("", "auto", "router", "auto/auto"):
            if hint and hint in CATEGORIES:
                decision = Decision(category=hint, scores={}, reasons=[f"X-Router-Hint: {hint}"],
                                    confidence=1.0, strategy="hint", forced="header hint")
                mode = "category"
            else:
                decision = classify(text, self.cfg, self.backend)
                mode = "auto"
        elif requested in CATEGORIES:
            decision = Decision(category=requested, scores={}, reasons=[f"caller asked for {requested!r}"],
                                confidence=1.0, strategy="explicit-category", forced="explicit category")
            mode = "category"
        elif requested in self.cfg.available or (self.cfg.offline and ":" in requested):
            decision = Decision(category="pinned", scores={}, reasons=[f"caller pinned {requested!r}"],
                                confidence=1.0, strategy="pinned", forced="explicit model")
            return Plan(decision, [requested], self._options("chat", params), "pinned")
        else:
            raise RoutingError(
                f"unknown model {requested!r}. Use 'auto', a category "
                f"({', '.join(CATEGORIES)}), or an installed model: {', '.join(self.cfg.available) or 'none'}"
            )

        chain = self.cfg.chain_for(decision.category)
        if not chain:
            raise RoutingError("no local models are installed; run `ollama pull qwen2.5:3b`", status=503)
        return Plan(decision, chain, self._options(decision.category, params), mode)

    def _options(self, category: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Category defaults, overridden by explicit OpenAI-style params."""
        options = self.cfg.options_for(category)
        for src, dst in PARAM_MAP.items():
            value = (params or {}).get(src)
            if value is not None:
                options[dst] = value
        return options

    # -- execution --------------------------------------------------------
    def complete(self, messages: List[Dict[str, Any]], plan: Plan) -> Dict[str, Any]:
        """Run the chain until one model answers. Returns the result plus trace."""
        attempts: List[Dict[str, Any]] = []
        started = time.time()

        for index, model in enumerate(plan.chain):
            attempt_started = time.time()
            try:
                result = self.backend.chat(model, messages, plan.options)
            except BackendError as exc:
                attempts.append({"model": model, "ok": False, "error": str(exc)})
                self.stats.record(plan.decision.category, model,
                                  (time.time() - attempt_started) * 1000, error=True)
                continue

            latency = (time.time() - started) * 1000
            self.stats.record(plan.decision.category, model, latency,
                              result["prompt_tokens"], result["completion_tokens"],
                              fallbacks=index)
            attempts.append({"model": model, "ok": True})
            result["model"] = model
            result["latency_ms"] = round(latency, 1)
            result["attempts"] = attempts
            result["fallbacks"] = index
            return result

        raise BackendError(
            f"every model in the {plan.decision.category!r} chain failed: "
            + "; ".join(f"{a['model']}: {a.get('error')}" for a in attempts),
            status=502,
        )

    def stream(self, messages: List[Dict[str, Any]], plan: Plan) -> Tuple[str, Iterator[Dict[str, Any]]]:
        """Pick a working model, then yield its chunks.

        Fallback only applies before the first token: once bytes are on the
        wire the client has already committed to a model.
        """
        started = time.time()
        last_error: Optional[BackendError] = None

        for index, model in enumerate(plan.chain):
            stream = self.backend.chat_stream(model, messages, plan.options)
            try:
                first = next(stream)
            except BackendError as exc:
                last_error = exc
                self.stats.record(plan.decision.category, model, 0.0, error=True)
                continue
            except StopIteration:
                first = {"done": True, "prompt_tokens": 0, "completion_tokens": 0}

            def chunks(first=first, stream=stream, model=model, index=index):
                usage = {"prompt_tokens": 0, "completion_tokens": 0}
                for chunk in (first, *stream):
                    if chunk.get("done"):
                        usage["prompt_tokens"] = chunk.get("prompt_tokens", 0)
                        usage["completion_tokens"] = chunk.get("completion_tokens", 0)
                        self.stats.record(plan.decision.category, model,
                                          (time.time() - started) * 1000,
                                          usage["prompt_tokens"], usage["completion_tokens"],
                                          fallbacks=index)
                        yield {"done": True, **usage}
                        return
                    yield chunk

            return model, chunks()

        raise last_error or BackendError("no model could start a stream", status=502)

    # -- introspection ----------------------------------------------------
    def route_table(self) -> Dict[str, Any]:
        return {
            category: {"model": chain[0] if chain else None, "fallbacks": chain[1:],
                       "options": self.cfg.options_for(category)}
            for category, chain in (self.cfg.resolved or {}).items()
        }
