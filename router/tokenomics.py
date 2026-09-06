"""Educational token estimation: input size, and a rough output-size forecast.

This is deliberately not a real tokenizer. Production tokenizers (BPE,
SentencePiece, etc.) are model-specific and split text into sub-word pieces
in ways that don't map cleanly onto characters or whitespace-separated words.
Building one is out of scope for a demo router with zero third-party
dependencies (see `router/backends.py`).

What this gives you instead is a classroom-grade approximation good enough to
reason about cost and latency *before* a model runs and reports the real
count back (Ollama's `prompt_eval_count` / `eval_count`, surfaced as
`usage.prompt_tokens` / `usage.completion_tokens` elsewhere in this API):

  input tokens   blend of chars/4 and words*1.3 — the two rules of thumb
                 people usually reach for, averaged so dense/code text and
                 loose prose both land in the right ballpark.
  output tokens  no model has run yet, so there is nothing to count. Instead
                 each category carries a fixed base size plus a share of the
                 prompt: a code request returns a function whether you asked
                 in ten words or fifty, while a summary scales with the text
                 you handed over. The result is a low/typical/high spread,
                 not a promise.

Treat every number here as "probably within a factor of 1.5-2x", not ground
truth. Two things it deliberately does not model: the chat template the
runtime wraps around your prompt (which is why a measured `prompt_eval_count`
comes in well above the input estimate), and how verbose one particular model
happens to be.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

_WORD = re.compile(r"\S+")

# Two independent rules of thumb for English/code text, averaged together.
CHARS_PER_TOKEN = 4.0
TOKENS_PER_WORD = 1.3

# Expected completion size as (base, ratio) pairs — a fixed amount for the
# kind of reply, plus a share of the prompt — given for low, typical and high.
#
# A pure ratio badly underestimates generative work: "write a merge function"
# is a dozen tokens in and several hundred out, because the reply's length is
# set by the task, not by how long you spent asking. A pure constant is just as
# wrong for transformations, where the reply really does scale with what you
# fed in. Splitting the two is what makes one table work for both.
#
#   code       a function plus explanation, largely independent of the ask.
#   math       an answer with brief working.
#   reasoning  expands the most: comparisons, step-by-step explanations.
#   summarize  almost entirely proportional — it shrinks what you supplied.
#   long       proportional and heavily compressed.
#   chat       a short reply that grows a little with the message.
Shape = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
OUTPUT_SHAPE: Dict[str, Shape] = {
    #             low          typical       high
    "code":      ((60, 0.4), (180, 1.0), (480, 2.2)),
    "math":      ((20, 0.2), (70, 0.6), (220, 1.4)),
    "reasoning": ((80, 0.5), (220, 1.2), (560, 2.6)),
    "summarize": ((8, 0.15), (20, 0.35), (60, 0.6)),
    "long":      ((10, 0.05), (30, 0.12), (90, 0.3)),
    "chat":      ((15, 0.3), (50, 0.8), (160, 1.8)),
}
DEFAULT_SHAPE: Shape = ((15, 0.3), (50, 0.8), (160, 1.8))  # unknown category

# A response has to start and stop somewhere; nothing gets estimated at zero.
MIN_OUTPUT_TOKENS = 8

DISCLAIMER = (
    "Educational approximation only, not a real tokenizer. Actual "
    "tokenization is model-specific (BPE/SentencePiece) and the true count "
    "can differ from this estimate by 20-40%; the output figure is a forecast "
    "made before any generation happens, not a measurement."
)


@dataclass
class TokenEstimate:
    chars: int
    words: int
    est_input_tokens: int
    est_output_low: int
    est_output_typical: int
    est_output_high: int
    category: str
    capped_by: Optional[str] = None

    @property
    def est_total_typical(self) -> int:
        return self.est_input_tokens + self.est_output_typical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": {
                "chars": self.chars,
                "words": self.words,
                "est_tokens": self.est_input_tokens,
            },
            "output": {
                "est_tokens_low": self.est_output_low,
                "est_tokens_typical": self.est_output_typical,
                "est_tokens_high": self.est_output_high,
                "capped_by": self.capped_by,
            },
            "est_total_tokens_typical": self.est_total_typical,
            "category": self.category,
            "method": ("heuristic: input from blended chars/4 and words*1.3, "
                       "output from a per-category base plus a share of the input; "
                       "no real tokenizer"),
            "disclaimer": DISCLAIMER,
        }


def estimate_input_tokens(text: str) -> int:
    """Blend chars/4 and words*1.3 so both prose and dense/code text land close."""
    if not text:
        return 0
    chars = len(text)
    words = len(_WORD.findall(text))
    by_chars = chars / CHARS_PER_TOKEN
    by_words = words * TOKENS_PER_WORD
    return max(1, round((by_chars + by_words) / 2))


def estimate(text: str, category: str = "chat",
             max_output_tokens: Optional[int] = None,
             num_ctx: Optional[int] = None) -> TokenEstimate:
    """Estimate input size and forecast a plausible output-size range.

    `max_output_tokens` (an explicit `max_tokens` / `num_predict`) and
    `num_ctx` (the context window) are hard ceilings when present — a
    forecast that ignores them would be misleading, not just approximate.
    """
    chars = len(text)
    words = len(_WORD.findall(text))
    input_tokens = estimate_input_tokens(text)

    low, typical, high = (
        max(MIN_OUTPUT_TOKENS, round(base + ratio * input_tokens))
        for base, ratio in OUTPUT_SHAPE.get(category, DEFAULT_SHAPE)
    )

    capped_by = None
    if max_output_tokens and max_output_tokens > 0 and high > max_output_tokens:
        low, typical, high = (min(v, max_output_tokens) for v in (low, typical, high))
        capped_by = f"max_tokens={max_output_tokens}"

    if num_ctx and num_ctx > 0:
        room = max(0, num_ctx - input_tokens)
        if high > room:
            low, typical, high = (min(v, room) for v in (low, typical, high))
            capped_by = capped_by or f"num_ctx={num_ctx}"

    return TokenEstimate(
        chars=chars, words=words, est_input_tokens=input_tokens,
        est_output_low=low, est_output_typical=typical, est_output_high=high,
        category=category, capped_by=capped_by,
    )
