"""HTTP API. stdlib ThreadingHTTPServer, OpenAI-compatible chat completions.

    GET  /health                 liveness + backend reachability
    GET  /v1/models              routes and installed models, OpenAI shaped
    GET  /routes                 the resolved route table
    GET  /stats                  per-model and per-category counters
    POST /route                  dry run: the decision, no generation
    POST /tokenomics             educational input/output token estimate
    POST /v1/chat/completions    the real thing (set stream=true for SSE)
    POST /admin/reload           re-read config and re-resolve models
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from . import tokenomics
from .backends import BackendError, OllamaBackend
from .classifier import prompt_text
from .config import CATEGORIES, RouterConfig
from .engine import Router, RoutingError

MAX_BODY = 8 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalModelRouter/1.0"
    router: Router = None  # injected on the server class

    # -- helpers ----------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, payload: Any, status: int = 200, extra_headers: Optional[Dict[str, str]] = None) -> None:
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400, kind: str = "invalid_request_error") -> None:
        self._send_json({"error": {"message": message, "type": kind, "code": status}}, status)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._error("request body is required")
            return None
        if length > MAX_BODY:
            self._error("request body too large", 413)
            return None
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            self._error(f"invalid JSON: {exc}")
            return None

    def _messages(self, body: Dict[str, Any]) -> Optional[list]:
        """Accept OpenAI `messages`, or a bare `prompt` for convenience."""
        messages = body.get("messages")
        if messages is None and body.get("prompt"):
            messages = [{"role": "user", "content": body["prompt"]}]
        if not isinstance(messages, list) or not messages:
            self._error("`messages` must be a non-empty array (or provide `prompt`)")
            return None
        for msg in messages:
            if not isinstance(msg, dict) or "content" not in msg:
                self._error("each message needs a `content` field")
                return None
            msg.setdefault("role", "user")
        return messages

    # -- routing ----------------------------------------------------------
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Router-Hint, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"
        router = self.router
        if path in ("/", "/health"):
            health = router.backend.health()
            self._send_json({
                "status": "ok" if health.get("reachable") else "degraded",
                "backend": {"url": router.cfg.ollama_url, **health},
                "classifier": router.cfg.classifier,
                "categories": CATEGORIES,
                "endpoints": ["/health", "/v1/models", "/routes", "/stats",
                              "/route", "/tokenomics", "/v1/chat/completions", "/admin/reload"],
            }, 200 if health.get("reachable") else 503)
        elif path == "/v1/models":
            data = [{"id": "auto", "object": "model", "owned_by": "router",
                     "description": "classify the request, then pick a local model"}]
            data += [{"id": c, "object": "model", "owned_by": "router",
                      "description": f"force the {c} route",
                      "target": (router.cfg.chain_for(c) or [None])[0]} for c in CATEGORIES]
            data += [{"id": m, "object": "model", "owned_by": "ollama"} for m in router.cfg.available]
            self._send_json({"object": "list", "data": data})
        elif path == "/routes":
            self._send_json({"routes": router.route_table(), "config": router.cfg.describe()})
        elif path == "/stats":
            self._send_json(router.stats.snapshot())
        else:
            self._error(f"no such endpoint: {path}", 404, "not_found")

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/route":
            self._handle_route()
        elif path == "/tokenomics":
            self._handle_tokenomics()
        elif path in ("/v1/chat/completions", "/chat/completions", "/v1/completions"):
            self._handle_completion()
        elif path == "/admin/reload":
            self._send_json({"reloaded": True, "config": self.router.refresh()})
        else:
            self._error(f"no such endpoint: {path}", 404, "not_found")

    # -- handlers ---------------------------------------------------------
    def _plan(self, body: Dict[str, Any]) -> Optional[Tuple[list, Any]]:
        messages = self._messages(body)
        if messages is None:
            return None
        try:
            plan = self.router.plan(
                messages,
                requested=body.get("model"),
                hint=self.headers.get("X-Router-Hint"),
                params=body,
            )
        except RoutingError as exc:
            self._error(str(exc), exc.status)
            return None
        return messages, plan

    def _tokenomics(self, messages: list, plan: Any) -> Dict[str, Any]:
        return tokenomics.estimate(
            prompt_text(messages),
            category=plan.decision.category,
            max_output_tokens=plan.options.get("num_predict"),
            num_ctx=plan.options.get("num_ctx"),
        ).to_dict()

    def _handle_route(self) -> None:
        body = self._read_json()
        if body is None:
            return
        started = time.time()
        planned = self._plan(body)
        if planned is None:
            return
        messages, plan = planned
        self._send_json({
            "decision": plan.decision.to_dict(),
            "mode": plan.mode,
            "selected_model": plan.model,
            "fallback_chain": plan.chain[1:],
            "options": plan.options,
            "prompt_preview": prompt_text(messages)[:200],
            "routing_ms": round((time.time() - started) * 1000, 2),
            "tokenomics": self._tokenomics(messages, plan),
        })

    def _handle_tokenomics(self) -> None:
        """Educational token estimate: input size, plus a forecast output range.

        No generation happens here — see `router/tokenomics.py` for what the
        numbers mean and why they are approximations, not measurements.
        """
        body = self._read_json()
        if body is None:
            return
        planned = self._plan(body)
        if planned is None:
            return
        messages, plan = planned
        self._send_json({
            "selected_model": plan.model,
            "mode": plan.mode,
            **self._tokenomics(messages, plan),
        })

    def _handle_completion(self) -> None:
        body = self._read_json()
        if body is None:
            return
        planned = self._plan(body)
        if planned is None:
            return
        messages, plan = planned

        if body.get("stream"):
            self._stream_completion(messages, plan)
            return

        try:
            result = self.router.complete(messages, plan)
        except BackendError as exc:
            self._error(str(exc), exc.status, "backend_error")
            return

        self._send_json({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["content"]},
                "finish_reason": "stop" if result["done_reason"] == "stop" else "length",
            }],
            "usage": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
            },
            "router": {
                "mode": plan.mode,
                "decision": plan.decision.to_dict(),
                "selected_model": result["model"],
                "fallbacks_used": result["fallbacks"],
                "attempts": result["attempts"],
                "latency_ms": result["latency_ms"],
            },
        }, extra_headers={
            "X-Router-Category": plan.decision.category,
            "X-Router-Model": result["model"],
            "X-Router-Strategy": plan.decision.strategy,
        })

    def _stream_completion(self, messages: list, plan: Any) -> None:
        try:
            model, chunks = self.router.stream(messages, plan)
        except BackendError as exc:
            self._error(str(exc), exc.status, "backend_error")
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # No Content-Length on an SSE body, so the client must read to EOF.
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Router-Category", plan.decision.category)
        self.send_header("X-Router-Model", model)
        self.end_headers()

        def emit(payload: Dict[str, Any]) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        def frame(delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
            return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        try:
            emit({**frame({"role": "assistant", "content": ""}),
                  "router": {"mode": plan.mode, "decision": plan.decision.to_dict(),
                             "selected_model": model}})
            for chunk in chunks:
                if chunk.get("done"):
                    emit({**frame({}, "stop"),
                          "usage": {"prompt_tokens": chunk["prompt_tokens"],
                                    "completion_tokens": chunk["completion_tokens"],
                                    "total_tokens": chunk["prompt_tokens"] + chunk["completion_tokens"]}})
                    break
                emit(frame({"content": chunk["delta"]}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.log_message("client disconnected mid-stream")
        except BackendError as exc:
            emit({"error": {"message": str(exc), "type": "backend_error"}})


def build_server(cfg: Optional[RouterConfig] = None) -> Tuple[ThreadingHTTPServer, Router]:
    cfg = cfg or RouterConfig.load()
    router = Router(cfg, OllamaBackend(cfg.ollama_url, cfg.request_timeout))
    try:
        router.refresh()
    except BackendError as exc:
        print(f"!! {exc}\n!! Start it with `ollama serve`; the API will run but return 503s.", flush=True)

    handler = type("BoundHandler", (Handler,), {"router": router})
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    httpd.daemon_threads = True
    return httpd, router


def main() -> None:
    parser = argparse.ArgumentParser(description="Local model routing API")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--classifier", choices=["heuristic", "llm"], default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = RouterConfig.load(args.config)
    cfg.host = args.host or cfg.host
    cfg.port = args.port or cfg.port
    cfg.ollama_url = (args.ollama_url or cfg.ollama_url).rstrip("/")
    cfg.classifier = args.classifier or cfg.classifier

    httpd, router = build_server(cfg)
    table = router.route_table()
    print(f"\n  Local model router  http://{cfg.host}:{cfg.port}")
    print(f"  backend: {cfg.ollama_url}   classifier: {cfg.classifier}")
    print("  route table:")
    for category, entry in table.items():
        fallbacks = f"  (fallback: {', '.join(entry['fallbacks'])})" if entry["fallbacks"] else ""
        print(f"    {category:<10} -> {entry['model']}{fallbacks}")
    print("\n  try: curl -s localhost:%d/health | python3 -m json.tool\n" % cfg.port, flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        httpd.shutdown()


if __name__ == "__main__":
    main()
