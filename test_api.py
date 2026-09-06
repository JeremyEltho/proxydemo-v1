"""FastAPI layer tests. Requires the dev extras: pip install -r requirements-dev.txt

    python3 test_api.py
"""

from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

import app.main as main
from router.config import RouterConfig
from router.engine import Router
from router.stats import Stats
from test_router import FAKE_MODELS, FakeBackend


def use_fake_backend(**cfg_kwargs) -> None:
    """Point the live app at a fake runtime so tests never touch Ollama."""
    cfg = RouterConfig()
    for key, value in cfg_kwargs.items():
        setattr(cfg, key, value)
    main.ENGINE = Router(cfg, FakeBackend(), Stats())
    main.ENGINE.refresh()


def use_offline() -> None:
    cfg = RouterConfig()
    cfg.resolve_static()
    main.ENGINE = Router(cfg, FakeBackend(), Stats())


class TestLocalMode(unittest.TestCase):
    def setUp(self):
        use_fake_backend()
        self.client = TestClient(main.app)

    def test_health_reports_local(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["mode"], "local")
        self.assertEqual(body["status"], "ok")

    def test_routes_mounted_bare_and_under_api(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_route_classifies(self):
        cases = {
            "refactor this python class to use dataclasses": "code",
            "calculate 12 * 9 + 4": "math",
            "compare the trade-offs of REST versus gRPC": "reasoning",
            "summarize the following text into bullet points: " + "blah " * 40: "summarize",
            "hey there": "chat",
        }
        for prompt, expected in cases.items():
            body = self.client.post("/api/route", json={"prompt": prompt}).json()
            self.assertEqual(body["decision"]["category"], expected, prompt[:40])
            self.assertTrue(body["selected_model"])
            self.assertTrue(body["can_generate"])

    def test_route_does_not_generate(self):
        before = len(main.ENGINE.backend.calls)
        self.client.post("/api/route", json={"prompt": "hi"})
        self.assertEqual(len(main.ENGINE.backend.calls), before)

    def test_forced_category(self):
        body = self.client.post("/api/route", json={"prompt": "hi", "model": "code"}).json()
        self.assertEqual(body["decision"]["category"], "code")
        self.assertEqual(body["mode"], "category")

    def test_header_hint(self):
        body = self.client.post("/api/route", json={"prompt": "hi"},
                                headers={"X-Router-Hint": "math"}).json()
        self.assertEqual(body["decision"]["category"], "math")

    def test_pinned_model(self):
        body = self.client.post("/api/route", json={"prompt": "hi", "model": "phi3:mini"}).json()
        self.assertEqual(body["mode"], "pinned")
        self.assertEqual(body["selected_model"], "phi3:mini")

    def test_unknown_model_is_400(self):
        res = self.client.post("/api/route", json={"prompt": "hi", "model": "gpt-4o"})
        self.assertEqual(res.status_code, 400)

    def test_missing_prompt_is_422(self):
        self.assertEqual(self.client.post("/api/route", json={}).status_code, 422)

    def test_route_includes_a_tokenomics_estimate(self):
        body = self.client.post("/api/route", json={"prompt": "hey there"}).json()
        tok = body["tokenomics"]
        self.assertGreater(tok["input"]["est_tokens"], 0)
        self.assertGreater(tok["output"]["est_tokens_typical"], 0)
        self.assertEqual(tok["category"], "chat")

    def test_chat_completion(self):
        res = self.client.post("/api/v1/chat/completions", json={
            "model": "auto", "messages": [{"role": "user", "content": "hey"}]})
        body = res.json()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "fake reply")
        self.assertEqual(body["usage"]["total_tokens"], 18)
        self.assertEqual(body["router"]["decision"]["category"], "chat")

    def test_openai_params_reach_the_backend(self):
        self.client.post("/api/v1/chat/completions", json={
            "prompt": "hey", "temperature": 0.42, "max_tokens": 16})
        _, options = main.ENGINE.backend.calls[-1]
        self.assertEqual(options["temperature"], 0.42)
        self.assertEqual(options["num_predict"], 16)

    def test_streaming(self):
        with self.client.stream("POST", "/api/v1/chat/completions",
                                json={"prompt": "hey", "stream": True}) as res:
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.headers["content-type"].startswith("text/event-stream"))
            raw = "".join(res.iter_text())

        frames = [json.loads(l[6:]) for l in raw.splitlines()
                  if l.startswith("data: ") and l.strip() != "data: [DONE]"]
        self.assertIn("router", frames[0])
        text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames)
        self.assertEqual(text.strip(), "fake reply")
        self.assertEqual(frames[-1]["usage"]["total_tokens"], 18)
        self.assertTrue(raw.strip().endswith("data: [DONE]"))

    def test_stats_accumulate(self):
        self.client.post("/api/v1/chat/completions", json={"prompt": "hey"})
        body = self.client.get("/api/stats").json()
        self.assertEqual(body["requests"], 1)

    def test_models_endpoint(self):
        ids = [m["id"] for m in self.client.get("/api/v1/models").json()["data"]]
        self.assertIn("auto", ids)
        self.assertIn("code", ids)
        self.assertIn("qwen2.5:3b", ids)

    def test_cors_is_open(self):
        res = self.client.options("/api/route", headers={
            "Origin": "https://example.vercel.app",
            "Access-Control-Request-Method": "POST"})
        self.assertEqual(res.headers["access-control-allow-origin"], "*")

    def test_frontend_is_served(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Local Model Router", res.text)


class TestTokenomicsEndpoint(unittest.TestCase):
    """Educational input/output token estimate. Never generates."""

    def setUp(self):
        use_fake_backend()
        self.client = TestClient(main.app)

    def test_estimates_input_and_output(self):
        body = self.client.post("/api/tokenomics", json={
            "prompt": "write a python function that merges two sorted lists"}).json()
        self.assertEqual(body["category"], "code")
        self.assertGreater(body["input"]["est_tokens"], 0)
        self.assertGreater(body["output"]["est_tokens_typical"], 0)
        self.assertLessEqual(body["output"]["est_tokens_low"], body["output"]["est_tokens_typical"])
        self.assertLessEqual(body["output"]["est_tokens_typical"], body["output"]["est_tokens_high"])
        self.assertIn("disclaimer", body)
        self.assertTrue(body["selected_model"])

    def test_does_not_generate(self):
        before = len(main.ENGINE.backend.calls)
        self.client.post("/api/tokenomics", json={"prompt": "hi"})
        self.assertEqual(len(main.ENGINE.backend.calls), before)

    def test_respects_forced_category(self):
        body = self.client.post("/api/tokenomics", json={
            "prompt": "cut this down: " + "blah " * 40, "model": "summarize"}).json()
        self.assertEqual(body["category"], "summarize")
        self.assertLess(body["output"]["est_tokens_typical"], body["input"]["est_tokens"])

    def test_max_tokens_caps_the_forecast(self):
        body = self.client.post("/api/tokenomics", json={
            "prompt": "write an essay about the history of computing " * 10,
            "model": "reasoning", "max_tokens": 12}).json()
        self.assertLessEqual(body["output"]["est_tokens_high"], 12)
        self.assertEqual(body["output"]["capped_by"], "max_tokens=12")

    def test_missing_prompt_is_422(self):
        self.assertEqual(self.client.post("/api/tokenomics", json={}).status_code, 422)


class TestCloudMode(unittest.TestCase):
    """What the Vercel deployment does: route, but never generate."""

    def setUp(self):
        use_offline()
        self.client = TestClient(main.app)

    def test_health_reports_routing_only(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["mode"], "cloud")
        self.assertEqual(body["status"], "routing-only")

    def test_routing_still_works_without_a_runtime(self):
        body = self.client.post("/api/route", json={"prompt": "write a python function"}).json()
        self.assertEqual(body["decision"]["category"], "code")
        self.assertTrue(body["selected_model"])
        self.assertFalse(body["can_generate"])

    def test_generation_is_503_with_guidance(self):
        res = self.client.post("/api/v1/chat/completions", json={"prompt": "hi"})
        self.assertEqual(res.status_code, 503)
        self.assertIn("locally", res.json()["detail"])

    def test_tokenomics_works_without_a_runtime(self):
        body = self.client.post("/api/tokenomics", json={"prompt": "hi"}).json()
        self.assertGreater(body["input"]["est_tokens"], 0)


class TestVercelEntrypoint(unittest.TestCase):
    def test_entrypoint_exports_asgi_app(self):
        import api.index as entry
        self.assertTrue(callable(entry.app))


if __name__ == "__main__":
    unittest.main(verbosity=2)
