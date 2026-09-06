"""Ollama client built on urllib. No third-party dependencies."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional


class BackendError(RuntimeError):
    """Any failure talking to the model runtime."""

    def __init__(self, message: str, status: int = 502, model: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.model = model


class OllamaBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- plumbing ---------------------------------------------------------
    def _request(self, path: str, payload: Optional[dict] = None, timeout: Optional[int] = None):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url, data=data, method="POST" if data else "GET",
            headers={"Content-Type": "application/json"},
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise BackendError(f"ollama {exc.code}: {detail}", status=502) from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"cannot reach ollama at {self.base_url}: {exc.reason}", status=503) from exc

    # -- api --------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        with self._request("/api/tags", timeout=10) as resp:
            return json.load(resp).get("models", [])

    def health(self) -> Dict[str, Any]:
        try:
            models = self.list_models()
            return {"reachable": True, "model_count": len(models)}
        except BackendError as exc:
            return {"reachable": False, "error": str(exc)}

    def chat(self, model: str, messages: List[Dict[str, Any]],
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Non-streaming completion. Returns content plus token accounting."""
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": options or {}}
        with self._request("/api/chat", payload) as resp:
            body = json.load(resp)
        if body.get("error"):
            raise BackendError(str(body["error"]), model=model)
        return {
            "content": (body.get("message") or {}).get("content", ""),
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "completion_tokens": body.get("eval_count", 0),
            "total_duration_ms": round(body.get("total_duration", 0) / 1e6, 1),
            "done_reason": body.get("done_reason", "stop"),
        }

    def chat_stream(self, model: str, messages: List[Dict[str, Any]],
                    options: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
        """Yield {"delta": str} chunks, then one {"done": True, ...} summary."""
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": options or {}}
        with self._request("/api/chat", payload) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                body = json.loads(line)
                if body.get("error"):
                    raise BackendError(str(body["error"]), model=model)
                if body.get("done"):
                    yield {
                        "done": True,
                        "prompt_tokens": body.get("prompt_eval_count", 0),
                        "completion_tokens": body.get("eval_count", 0),
                        "done_reason": body.get("done_reason", "stop"),
                    }
                    return
                delta = (body.get("message") or {}).get("content", "")
                if delta:
                    yield {"delta": delta}
        yield {"done": True, "prompt_tokens": 0, "completion_tokens": 0, "done_reason": "stop"}
