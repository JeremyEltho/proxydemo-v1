"""FastAPI app wrapping the routing engine.

Runs in two modes, from the same code:

  local  Ollama is reachable, so routing decisions AND generation work
  cloud  no Ollama (Vercel), so /route still shows the decision and
         /v1/chat/completions returns a clear 503

Every route is mounted twice, bare and under /api, so the same frontend works
against `uvicorn app.main:app` locally and against Vercel's api/ function.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from router import __version__
from router.backends import BackendError, OllamaBackend
from router.classifier import prompt_text
from router.config import CATEGORIES, RouterConfig
from router.engine import Router, RoutingError

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str


class ChatRequest(BaseModel):
    """OpenAI-shaped, plus a `prompt` shorthand for quick testing."""

    model: str = Field("auto", description="'auto', a category, or an installed model id")
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None

    def as_messages(self) -> List[Dict[str, Any]]:
        if self.messages:
            return [m.model_dump() for m in self.messages]
        if self.prompt:
            return [{"role": "user", "content": self.prompt}]
        raise HTTPException(422, "provide either `messages` or `prompt`")

    def params(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump(
            include={"temperature", "top_p", "max_tokens", "seed", "stop"}).items() if v is not None}


class RouteRequest(BaseModel):
    model: str = "auto"
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None

    def as_messages(self) -> List[Dict[str, Any]]:
        return ChatRequest(model=self.model, messages=self.messages, prompt=self.prompt).as_messages()


# --------------------------------------------------------------------------
# app wiring
# --------------------------------------------------------------------------
def build_router() -> Router:
    cfg = RouterConfig.load()
    engine = Router(cfg, OllamaBackend(cfg.ollama_url, cfg.request_timeout))
    if os.environ.get("ROUTER_OFFLINE", "").lower() in ("1", "true", "yes"):
        # Serverless: skip the probe entirely, no runtime is coming.
        cfg.resolve_static()
    else:
        # Never fatal at import time; degrade to routing-only if Ollama is down.
        engine.refresh(strict=False)
    return engine


ENGINE = build_router()
api = APIRouter()


def plan_for(body, hint: Optional[str], params: Optional[Dict[str, Any]] = None):
    try:
        return ENGINE.plan(body.as_messages(), requested=body.model, hint=hint, params=params)
    except RoutingError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


def require_backend() -> None:
    if ENGINE.cfg.offline:
        raise HTTPException(503, (
            "routing works here, but generation needs a local model runtime. "
            "This deployment cannot reach one. Run the router locally "
            "(`./run.sh`) and point the frontend's API base at it."
        ))


@api.get("/health")
def health() -> Dict[str, Any]:
    # In forced-offline mode there is nothing to probe; skip the round trip.
    backend = ({"reachable": False, "error": "no model runtime in this deployment"}
               if ENGINE.cfg.offline else ENGINE.backend.health())
    return {
        "status": "ok" if backend.get("reachable") else "routing-only",
        "version": __version__,
        "mode": "cloud" if ENGINE.cfg.offline else "local",
        "backend": {"url": ENGINE.cfg.ollama_url, **backend},
        "classifier": ENGINE.cfg.classifier,
        # Hybrid needs a runtime to escalate to; without one it is heuristic-only.
        "escalation_available": bool(backend.get("reachable") and ENGINE.cfg.classifier_model),
        "classifier_model": ENGINE.cfg.classifier_model,
        "categories": CATEGORIES,
    }


@api.get("/routes")
def routes() -> Dict[str, Any]:
    return {"routes": ENGINE.route_table(), "config": ENGINE.cfg.describe()}


@api.get("/stats")
def stats() -> Dict[str, Any]:
    return ENGINE.stats.snapshot()


@api.get("/v1/models")
def models() -> Dict[str, Any]:
    data: List[Dict[str, Any]] = [{
        "id": "auto", "object": "model", "owned_by": "router",
        "description": "classify the request, then pick a local model",
    }]
    data += [{"id": c, "object": "model", "owned_by": "router",
              "description": f"force the {c} route",
              "target": (ENGINE.cfg.chain_for(c) or [None])[0]} for c in CATEGORIES]
    data += [{"id": m, "object": "model", "owned_by": "ollama"} for m in ENGINE.cfg.available]
    return {"object": "list", "data": data}


@api.post("/route")
def route(body: RouteRequest, x_router_hint: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dry run: what would happen, without spending a single token."""
    started = time.perf_counter()
    messages = body.as_messages()
    plan = plan_for(body, x_router_hint)
    return {
        "decision": plan.decision.to_dict(),
        "mode": plan.mode,
        "selected_model": plan.model,
        "fallback_chain": plan.chain[1:],
        "options": plan.options,
        "prompt_preview": prompt_text(messages)[:200],
        "routing_ms": round((time.perf_counter() - started) * 1000, 2),
        "can_generate": not ENGINE.cfg.offline,
    }


@api.post("/admin/reload")
def reload_config() -> Dict[str, Any]:
    global ENGINE
    ENGINE = build_router()
    return {"reloaded": True, "config": ENGINE.cfg.describe()}


@api.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, request: Request,
                     x_router_hint: Optional[str] = Header(None)):
    require_backend()
    messages = body.as_messages()
    plan = plan_for(body, x_router_hint, body.params())

    if body.stream:
        return StreamingResponse(
            _sse(messages, plan),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Router-Category": plan.decision.category},
        )

    try:
        result = ENGINE.complete(messages, plan)
    except BackendError as exc:
        raise HTTPException(exc.status, str(exc)) from exc

    return {
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
            "fallback_chain": plan.chain[1:],
            "fallbacks_used": result["fallbacks"],
            "attempts": result["attempts"],
            "latency_ms": result["latency_ms"],
        },
    }


def _sse(messages: List[Dict[str, Any]], plan):
    """Sync generator; Starlette drives it off the event loop in a threadpool."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    try:
        model, chunks = ENGINE.stream(messages, plan)
    except BackendError as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
        yield "data: [DONE]\n\n"
        return

    def frame(delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
        return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield "data: " + json.dumps({
        **frame({"role": "assistant", "content": ""}),
        "router": {"mode": plan.mode, "decision": plan.decision.to_dict(),
                   "selected_model": model, "fallback_chain": plan.chain[1:]},
    }) + "\n\n"

    try:
        for chunk in chunks:
            if chunk.get("done"):
                yield "data: " + json.dumps({
                    **frame({}, "stop"),
                    "usage": {"prompt_tokens": chunk["prompt_tokens"],
                              "completion_tokens": chunk["completion_tokens"],
                              "total_tokens": chunk["prompt_tokens"] + chunk["completion_tokens"]},
                }) + "\n\n"
                break
            yield "data: " + json.dumps(frame({"content": chunk["delta"]})) + "\n\n"
    except BackendError as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
    yield "data: [DONE]\n\n"


app = FastAPI(
    title="Local Model Router",
    version=__version__,
    description="Classifies each request, then routes it to the right small local model.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # the hosted frontend may call a router on localhost
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Router-Category", "X-Router-Model"],
)

# Bare paths for OpenAI clients, /api/* for the Vercel function and frontend.
app.include_router(api)
app.include_router(api, prefix="/api")

if PUBLIC_DIR.exists():
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(PUBLIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")
