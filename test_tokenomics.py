"""Unit tests for the educational token estimator. No network, no dependencies.

    python3 test_tokenomics.py
"""

from __future__ import annotations

import unittest

from router.tokenomics import estimate, estimate_input_tokens


class TestInputEstimate(unittest.TestCase):
    def test_empty_text_is_zero(self):
        self.assertEqual(estimate_input_tokens(""), 0)

    def test_scales_with_length(self):
        short = estimate_input_tokens("hello there")
        long = estimate_input_tokens("hello there " * 50)
        self.assertGreater(long, short * 10)

    def test_roughly_matches_the_char_over_four_rule_of_thumb(self):
        text = "the quick brown fox jumps over the lazy dog " * 4
        got = estimate_input_tokens(text)
        rough = len(text) / 4
        self.assertLess(abs(got - rough) / rough, 0.35)

    def test_never_zero_for_nonempty_text(self):
        self.assertGreaterEqual(estimate_input_tokens("hi"), 1)


class TestOutputForecast(unittest.TestCase):
    def test_summarize_forecasts_smaller_than_input(self):
        text = "summarize this: " + "lorem ipsum dolor sit amet consectetur adipiscing. " * 30
        est = estimate(text, category="summarize")
        self.assertLess(est.est_output_typical, est.est_input_tokens)

    def test_code_forecasts_larger_than_input(self):
        text = "write a function that merges two sorted lists"
        est = estimate(text, category="code")
        self.assertGreater(est.est_output_typical, est.est_input_tokens)

    def test_short_generative_asks_still_forecast_a_real_reply(self):
        """A dozen tokens of "write me a function" does not get a dozen back.

        The regression this guards: a purely proportional model forecast ~30
        tokens here, and a real run came back with 780.
        """
        est = estimate("write a python function that merges two sorted lists",
                       category="code")
        self.assertLess(est.est_input_tokens, 30)
        self.assertGreater(est.est_output_typical, 150)
        self.assertGreater(est.est_output_high, 400)

    def test_transformations_still_scale_with_the_input(self):
        """Summaries are the other half: the base must not swamp the ratio."""
        small = estimate("summarize: " + "word " * 20, category="summarize")
        large = estimate("summarize: " + "word " * 2000, category="summarize")
        self.assertGreater(large.est_output_typical, small.est_output_typical * 10)

    def test_low_typical_high_are_ordered(self):
        est = estimate("explain the trade-offs of REST versus gRPC", category="reasoning")
        self.assertLessEqual(est.est_output_low, est.est_output_typical)
        self.assertLessEqual(est.est_output_typical, est.est_output_high)

    def test_unknown_category_falls_back_to_default_shape(self):
        known = estimate("hello", category="chat")
        unknown = estimate("hello", category="not-a-real-category")
        self.assertEqual(known.est_output_typical, unknown.est_output_typical)

    def test_output_never_below_the_floor(self):
        est = estimate("hi", category="summarize")
        self.assertGreaterEqual(est.est_output_low, 1)

    def test_max_tokens_caps_the_forecast(self):
        text = "write a long essay about the history of computing " * 20
        uncapped = estimate(text, category="reasoning")
        capped = estimate(text, category="reasoning", max_output_tokens=10)
        self.assertGreater(uncapped.est_output_high, 10)
        self.assertLessEqual(capped.est_output_high, 10)
        self.assertEqual(capped.capped_by, "max_tokens=10")

    def test_num_ctx_caps_when_input_leaves_little_room(self):
        text = "explain in detail " * 200  # deliberately large input
        ctx = estimate_input_tokens(text) + 20  # only a little headroom past the input
        est = estimate(text, category="reasoning", num_ctx=ctx)
        self.assertLessEqual(est.est_output_high, max(0, ctx - est.est_input_tokens))
        self.assertIsNotNone(est.capped_by)


class TestToDict(unittest.TestCase):
    def test_shape(self):
        est = estimate("hello there, how are you?", category="chat")
        d = est.to_dict()
        self.assertIn("input", d)
        self.assertIn("output", d)
        self.assertIn("est_tokens", d["input"])
        self.assertIn("est_tokens_typical", d["output"])
        self.assertIn("disclaimer", d)
        self.assertEqual(d["est_total_tokens_typical"], est.est_input_tokens + est.est_output_typical)


if __name__ == "__main__":
    unittest.main()
