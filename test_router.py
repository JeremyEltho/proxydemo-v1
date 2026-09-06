"""Unit and integration tests. No network, no third-party packages.

    python3 test_router.py
"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from router.backends import BackendError
from router.classifier import classify_heuristic
from router.config import RouterConfig
from router.engine import Router, RoutingError
from router.server import Handler
from router.stats import Stats

FAKE_MODELS = [
    {"name": "dolphin-phi:latest", "size": 1_600_000_000},
    {"name": "qwen2.5:3b", "size": 1_900_000_000},
    {"name": "phi3:mini", "size": 2_200_000_000},
    {"name": "qwen2.5-coder:14b", "size": 9_000_000_000},
]


class FakeBackend:
    """Stands in for Ollama. `fail` names models that always error."""

    def __init__(self, fail=(), reply="fake reply"):
        self.fail = set(fail)
        self.reply = reply
        self.calls = []

    def list_models(self):
        return list(FAKE_MODELS)

    def health(self):
        return {"reachable": True, "model_count": len(FAKE_MODELS)}

    def chat(self, model, messages, options=None):
        self.calls.append((model, options))
        if model in self.fail:
            raise BackendError(f"{model} is down", model=model)
        return {"content": self.reply, "prompt_tokens": 11, "completion_tokens": 7,
                "total_duration_ms": 1.0, "done_reason": "stop"}

    def chat_stream(self, model, messages, options=None):
        self.calls.append((model, options))
        if model in self.fail:
            raise BackendError(f"{model} is down", model=model)
        for token in self.reply.split():
            yield {"delta": token + " "}
        yield {"done": True, "prompt_tokens": 11, "completion_tokens": 7, "done_reason": "stop"}


def make_router(**kwargs) -> Router:
    cfg = RouterConfig()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    backend = FakeBackend()
    router = Router(cfg, backend, Stats())
    router.refresh()
    return router


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.cfg = RouterConfig()

    def assert_category(self, text, expected):
        got = classify_heuristic(text, self.cfg)
        self.assertEqual(got.category, expected,
                         f"{text[:60]!r} -> {got.category} (scores {got.to_dict()['scores']})")

    def test_code(self):
        self.assert_category("Write a python function that reverses a linked list", "code")
        self.assert_category("```js\nconst x = () => 1\n```\nwhy does this fail?", "code")
        self.assert_category("I get a traceback when I import my module, help me debug it", "code")
        # A named language is a strong code signal even with no code in the prompt.
        self.assert_category("Write a Python one-liner that sums a list of ints", "code")
        self.assert_category("give me a bash script that rotates logs", "code")

    def test_math(self):
        self.assert_category("Calculate 17 * 43 + 12", "math")
        self.assert_category("What is the derivative of x^3 with respect to x?", "math")

    def test_reasoning(self):
        self.assert_category("Compare the trade-offs of REST versus gRPC for our service", "reasoning")
        self.assert_category("Analyze step-by-step whether we should migrate off Postgres", "reasoning")

    def test_summarize(self):
        self.assert_category("Summarize the following article into three bullet points: " + "blah " * 40,
                             "summarize")

    def test_chat(self):
        self.assert_category("hey there", "chat")
        self.assert_category("how are you today?", "chat")

    def test_long_context_override(self):
        decision = classify_heuristic("word " * 4000, self.cfg)
        self.assertEqual(decision.category, "long")
        self.assertIsNotNone(decision.forced)

    def test_decision_is_serializable(self):
        json.dumps(classify_heuristic("hello", self.cfg).to_dict())


class TestConfigResolution(unittest.TestCase):
    def test_unavailable_preferences_are_skipped(self):
        cfg = RouterConfig()
        cfg.resolve(FAKE_MODELS)
        # No small coder model is installed, so the chain skips to the general
        # small model and keeps the big coder model as a last resort.
        self.assertEqual(cfg.chain_for("code")[0], "qwen2.5:3b")
        self.assertNotIn("qwen2.5-coder:7b", cfg.chain_for("code"))
        self.assertEqual(cfg.chain_for("code")[-1], "qwen2.5-coder:14b")

    def test_empty_chain_falls_back_to_smallest(self):
        cfg = RouterConfig()
        cfg.routes["code"] = ["not-installed:1b"]
        cfg.resolve(FAKE_MODELS)
        self.assertEqual(cfg.chain_for("code")[0], "dolphin-phi:latest")

    def test_classifier_model_defaults_to_smallest(self):
        cfg = RouterConfig()
        cfg.resolve(FAKE_MODELS)
        self.assertEqual(cfg.classifier_model, "dolphin-phi:latest")


class TestPlanning(unittest.TestCase):
    def setUp(self):
        self.router = make_router()

    def msg(self, text):
        return [{"role": "user", "content": text}]

    def test_auto_routes_by_content(self):
        plan = self.router.plan(self.msg("refactor this python class"), "auto")
        self.assertEqual(plan.decision.category, "code")
        self.assertEqual(plan.mode, "auto")

    def test_explicit_category(self):
        plan = self.router.plan(self.msg("hi"), "math")
        self.assertEqual(plan.decision.category, "math")
        self.assertEqual(plan.mode, "category")

    def test_pinned_model_bypasses_routing(self):
        plan = self.router.plan(self.msg("hi"), "phi3:mini")
        self.assertEqual(plan.mode, "pinned")
        self.assertEqual(plan.chain, ["phi3:mini"])

    def test_header_hint(self):
        plan = self.router.plan(self.msg("hi"), "auto", hint="summarize")
        self.assertEqual(plan.decision.category, "summarize")

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(RoutingError):
            self.router.plan(self.msg("hi"), "gpt-4")

    def test_params_override_category_options(self):
        plan = self.router.plan(self.msg("hi"), "code", params={"temperature": 0.9, "max_tokens": 64})
        self.assertEqual(plan.options["temperature"], 0.9)
        self.assertEqual(plan.options["num_predict"], 64)

    def test_category_sets_default_options(self):
        plan = self.router.plan(self.msg("write a function"), "auto")
        self.assertEqual(plan.options["temperature"], 0.1)  # code runs cold


class TestExecution(unittest.TestCase):
    def msg(self, text):
        return [{"role": "user", "content": text}]

    def test_completion(self):
        router = make_router()
        plan = router.plan(self.msg("hello"), "auto")
        result = router.complete(self.msg("hello"), plan)
        self.assertEqual(result["content"], "fake reply")
        self.assertEqual(result["fallbacks"], 0)

    def test_falls_back_when_first_model_fails(self):
        cfg = RouterConfig()
        router = Router(cfg, FakeBackend(fail={"qwen2.5:3b"}), Stats())
        router.refresh()
        plan = router.plan(self.msg("write a python function"), "auto")
        result = router.complete(self.msg("write a python function"), plan)
        self.assertEqual(result["fallbacks"], 1)
        self.assertNotEqual(result["model"], "qwen2.5:3b")
        self.assertFalse(result["attempts"][0]["ok"])

    def test_raises_when_whole_chain_fails(self):
        cfg = RouterConfig()
        router = Router(cfg, FakeBackend(fail={m["name"] for m in FAKE_MODELS}), Stats())
        router.refresh()
        plan = router.plan(self.msg("hello"), "auto")
        with self.assertRaises(BackendError):
            router.complete(self.msg("hello"), plan)

    def test_stream_yields_deltas_then_usage(self):
        router = make_router()
        plan = router.plan(self.msg("hello"), "auto")
        model, chunks = router.stream(self.msg("hello"), plan)
        collected = list(chunks)
        self.assertIn(model, router.cfg.available)
        self.assertTrue(collected[-1]["done"])
        self.assertEqual("".join(c["delta"] for c in collected[:-1]).strip(), "fake reply")

    def test_stats_are_recorded(self):
        router = make_router()
        plan = router.plan(self.msg("hello"), "auto")
        router.complete(self.msg("hello"), plan)
        snapshot = router.stats.snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(sum(snapshot["by_category"].values()), 1)


class TestHTTPAPI(unittest.TestCase):
    """Drives the real handler over a real socket, with a fake backend."""

    @classmethod
    def setUpClass(cls):
        cls.router = make_router()
        handler = type("TestHandler", (Handler,), {"router": cls.router, "log_message": lambda *a: None})
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as resp:
            return resp.status, json.load(resp)

    def post(self, path, payload, headers=None, raw=False):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **(headers or {})})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            return resp.status, (body if raw else json.loads(body)), dict(resp.headers)

    def test_health(self):
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_models_lists_router_and_backend_ids(self):
        _, body = self.get("/v1/models")
        ids = [m["id"] for m in body["data"]]
        self.assertIn("auto", ids)
        self.assertIn("code", ids)
        self.assertIn("qwen2.5:3b", ids)

    def test_routes(self):
        _, body = self.get("/routes")
        self.assertIn("code", body["routes"])

    def test_route_dry_run_does_not_generate(self):
        before = len(self.router.backend.calls)
        _, body, _ = self.post("/route", {"prompt": "fix this failing unit test in python"})
        self.assertEqual(body["decision"]["category"], "code")
        self.assertEqual(len(self.router.backend.calls), before)

    def test_chat_completion_shape(self):
        status, body, headers = self.post("/v1/chat/completions", {
            "model": "auto", "messages": [{"role": "user", "content": "hey"}]})
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "fake reply")
        self.assertEqual(body["usage"]["total_tokens"], 18)
        self.assertEqual(headers["X-Router-Category"], "chat")
        self.assertIn("decision", body["router"])

    def test_hint_header_is_honoured(self):
        _, body, _ = self.post("/v1/chat/completions",
                               {"messages": [{"role": "user", "content": "hey"}]},
                               headers={"X-Router-Hint": "code"})
        self.assertEqual(body["router"]["decision"]["category"], "code")

    def test_streaming_sse(self):
        _, raw, headers = self.post("/v1/chat/completions", {
            "model": "auto", "stream": True,
            "messages": [{"role": "user", "content": "hey"}]}, raw=True)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        self.assertTrue(raw.strip().endswith("data: [DONE]"))
        frames = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith("data: ") and not line.endswith("[DONE]")]
        text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames)
        self.assertEqual(text.strip(), "fake reply")
        self.assertEqual(frames[-1]["usage"]["total_tokens"], 18)

    def test_bad_json(self):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/route", data=b"{nope",
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_model_returns_400(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/v1/chat/completions", {"model": "gpt-5", "prompt": "hi"})
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_endpoint_returns_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_stats_endpoint(self):
        _, body = self.get("/stats")
        self.assertIn("by_model", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
