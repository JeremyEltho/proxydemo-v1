"""Decide which category a request belongs to.

Two strategies share one interface:

  heuristic  weighted regex features; ~0ms, fully explainable, the default
  llm        a small local model votes, with the heuristic as the fallback
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import CATEGORIES, RouterConfig

# (compiled pattern, weight, human-readable reason)
Feature = Tuple[re.Pattern, float, str]


def _f(pattern: str, weight: float, reason: str) -> Feature:
    return (re.compile(pattern, re.I | re.M), weight, reason)


# Programming languages, shared by three features below.
LANGUAGES = (r"python|javascript|typescript|rust|golang|java|c\+\+|bash|shell|"
             r"html|css|react|sql")

# An explicit summarization request. Used to exempt a short message from the
# "too short to be a summary request" dampener.
EXPLICIT_SUMMARIZE = re.compile(
    r"\b(summari[sz]e|summary|tl;?dr|condense|outline|bullet points|"
    r"key (points|takeaways)|rewrite|paraphrase|convert)\b", re.I)

FEATURES: Dict[str, List[Feature]] = {
    "code": [
        _f(r"```", 3.0, "fenced code block"),
        _f(r"\b(def|class|import|from\s+\w+\s+import)\b", 2.0, "python syntax"),
        _f(r"(\b(const|let|var|async|await)\b|=>)", 1.5, "javascript syntax"),
        _f(r"\b(SELECT|INSERT|UPDATE|JOIN|WHERE)\b", 1.5, "sql syntax"),
        _f(r"\b(bug|refactor|stack ?trace|traceback|compile|debug|unit test)\b", 2.0, "engineering task"),
        _f(r"\b(null ?pointer|segfault|segmentation fault|memory leak|race condition|"
           r"deadlock|off[- ]by[- ]one|type error|syntax error|runtime error|exception)\b",
           2.0, "fault or error vocabulary"),
        _f(r"\b(regex|api|endpoint|json|yaml|docker|git)\b", 1.0, "technical vocabulary"),
        # A language name on its own is weak — "python is my favorite snake".
        _f(rf"\b({LANGUAGES})\b", 0.6, "names a programming language"),
        # The same name inside an actual task is strong.
        _f(rf"\b(write|writing|fix|fixing|debug|run|compile|refactor|implement|"
           rf"optimi[sz]e|review|port|test)\b[^.!?]{{0,60}}\b({LANGUAGES})\b",
           2.0, "code task in a named language"),
        _f(rf"\b({LANGUAGES})\b[^.!?]{{0,40}}\b"
           rf"(script|code|function|program|snippet|one[- ]?liner|app)\b",
           2.0, "named language applied to code"),
        _f(r"\b(one[- ]?liner|snippet|script|method|algorithm|data structure)\b", 1.5, "code artifact noun"),
        # "my code" / "this bug" names a specific artifact. Bare "code" as a verb
        # ("I used to code in college") deliberately does not match.
        _f(r"\b(this|that|these|those|my|our|your|the)\s+"
           r"(code|script|function|program|module|class|bug|error|method|snippet|query|pointer)\b",
           1.2, "refers to a specific code artifact"),
        _f(r"[{};]\s*$", 0.8, "code punctuation"),
        _f(r"\b(write|fix|implement|optimi[sz]e)\b.{0,30}\b(code|function|script|class)\b", 2.5, "code request"),
    ],
    "math": [
        _f(r"\d+\s*[\+\-\*/\^]\s*\d+", 2.5, "arithmetic expression"),
        _f(r"\b(calculate|compute|solve|evaluate)\b", 2.0, "calculation verb"),
        _f(r"\b(equation|integral|derivative|matrix|probability|theorem|factorial)\b", 2.5, "math vocabulary"),
        _f(r"\b(math|maths|mathematics)\b", 1.5, "math domain noun"),
        _f(r"\b(sum|product|percent|percentage|average|median)\b", 1.2, "quantitative vocabulary"),
        _f(r"[=<>]\s*\d", 0.8, "numeric comparison"),
        _f(r"\b\d+(\.\d+)?%", 1.0, "percentage literal"),
    ],
    "reasoning": [
        _f(r"\b(why|how come|explain|reason|reasoning|rationale|justify)\b", 1.5, "explanatory question"),
        _f(r"\b(compare|contrast|trade ?-?offs?|pros and cons|versus|vs\.?)\b", 2.2, "comparison request"),
        _f(r"\b(step[- ]by[- ]step|analy[sz]e|evaluate|assess)\b", 2.2, "analysis request"),
        # Tolerate the object in between: "think it through", "walk me through".
        _f(r"\b(think|walk|talk)\s+(me\s+|us\s+|it\s+|this\s+|that\s+)?through\b",
           2.2, "asks to be taken through it"),
        _f(r"\b(plan|strategy|architect|design|decide|recommend)\b", 1.8, "planning request"),
        _f(r"\b(if .{0,40} then|implication|consequence|because)\b", 1.2, "causal language"),
    ],
    "summarize": [
        _f(r"\b(summari[sz]e|summary|tl;?dr|condense|abstract)\b", 3.0, "summarization verb"),
        _f(r"\b(key (points|takeaways)|main ideas|bullet points|outline)\b", 2.5, "extraction request"),
        _f(r"\b(extract|rewrite|paraphrase|translate|proofread|convert)\b", 2.0, "transformation verb"),
        _f(r"\b(the (following|text|article|document|transcript)|below)\b", 1.5, "refers to supplied text"),
    ],
    "chat": [
        _f(r"^\s*(hi|hey|hello|yo|sup|thanks|thank you|ok|okay|cool|lol)\b", 3.0, "greeting or ack"),
        _f(r"\b(how are you|what's up|who are you|your name|tell me a joke)\b", 3.0, "small talk"),
        _f(r"\b(chat|talk|speak)\s+(with|to)\s+(me|us)\b|\blet'?s\s+(chat|talk|discuss)\b",
           2.5, "asks for conversation"),
        _f(r"\b(my|your)\s+favou?rite\b", 1.5, "personal preference"),
        _f(r"^\s*\S{1,40}\s*[?.!]?\s*$", 1.0, "very short message"),
    ],
}

# "long" is not scored; it is a hard override applied on estimated size.


@dataclass
class Decision:
    category: str
    scores: Dict[str, float]
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0
    strategy: str = "heuristic"
    est_tokens: int = 0
    forced: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "strategy": self.strategy,
            "est_tokens": self.est_tokens,
            "scores": {k: round(v, 2) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
            "reasons": self.reasons,
            "forced": self.forced,
        }


def estimate_tokens(text: str) -> int:
    """Rough char/4 estimate. Good enough for a size-based routing threshold."""
    return max(1, len(text) // 4)


def prompt_text(messages: List[Dict[str, Any]]) -> str:
    """Flatten a chat payload to the text the classifier looks at."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):  # OpenAI multi-part content
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(str(content))
    return "\n".join(parts)


def classify_heuristic(text: str, cfg: RouterConfig) -> Decision:
    scores: Dict[str, float] = {c: 0.0 for c in CATEGORIES}
    reasons: List[str] = []

    for category, features in FEATURES.items():
        for pattern, weight, reason in features:
            hits = len(pattern.findall(text))
            if hits:
                # Diminishing returns: repeats confirm, they don't dominate.
                scores[category] += weight * (1 + 0.35 * min(hits - 1, 4))
                reasons.append(f"{category}: {reason} (x{hits})")

    # A long message is rarely small talk.
    est = estimate_tokens(text)
    if est > 200:
        scores["chat"] *= 0.4
    # Short messages are usually not summarization requests — unless they say so.
    if est < 60 and not EXPLICIT_SUMMARIZE.search(text):
        scores["summarize"] *= 0.6

    top = max(scores, key=lambda c: scores[c])
    total = sum(scores.values())
    strategy = "heuristic"

    if total == 0:
        # Nothing matched at all. This is a fallthrough, not a classification, and
        # callers need to be able to tell the difference.
        top, confidence, strategy = "chat", 0.0, "fallback"
        reasons = ["chat: no category signal in the text; defaulted to chat"]
    elif scores[top] <= 0.5:
        top, confidence, strategy = "chat", 0.25, "fallback"
    else:
        # Confidence is how far ahead the winner is, discounted by how much
        # evidence there was at all. One weak feature is not certainty.
        share = scores[top] / total
        strength = min(1.0, scores[top] / 3.0)
        confidence = share * strength

    decision = Decision(
        category=top,
        scores=scores,
        reasons=[r for r in reasons if r.startswith(top + ":")][:6],
        confidence=confidence,
        strategy=strategy,
        est_tokens=est,
    )

    if est >= cfg.long_context_tokens:
        decision.forced = f"long-context ({est} est. tokens >= {cfg.long_context_tokens})"
        decision.category = "long"
        decision.confidence = 1.0
    return decision


CLASSIFIER_PROMPT = (
    "You are a request classifier. Reply with exactly one word from this list and "
    "nothing else: code, math, reasoning, summarize, chat.\n\n"
    "Rules: code = programming or debugging. math = calculation. "
    "reasoning = analysis, comparison or planning. summarize = condensing or "
    "rewriting supplied text. chat = greetings and casual conversation.\n\n"
    "Request:\n{text}\n\nOne word:"
)


def classify_llm(text: str, cfg: RouterConfig, backend) -> Decision:
    """Let a small local model vote, falling back to the heuristic on any doubt."""
    fallback = classify_heuristic(text, cfg)
    if fallback.forced:  # size override wins regardless of the vote
        return fallback

    excerpt = text[:1200]
    try:
        result = backend.chat(
            model=cfg.classifier_model,
            messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(text=excerpt)}],
            options={"temperature": 0.0, "num_predict": 5},
        )
        raw = (result.get("content") or "").strip().lower()
    except Exception as exc:  # backend down, model missing, timeout
        fallback.reasons.append(f"llm classifier unavailable ({exc.__class__.__name__}), used heuristic")
        return fallback

    word = re.sub(r"[^a-z]", "", raw.split()[0]) if raw.split() else ""
    if word not in CATEGORIES or word == "long":
        fallback.reasons.append(f"llm classifier returned unusable {raw!r}, used heuristic")
        return fallback

    return Decision(
        category=word,
        scores=fallback.scores,
        reasons=[f"{cfg.classifier_model} voted {word!r}", f"heuristic agreed: {word == fallback.category}"],
        confidence=1.0 if word == fallback.category else 0.7,
        strategy=f"llm:{cfg.classifier_model}",
        est_tokens=fallback.est_tokens,
    )


def classify(text: str, cfg: RouterConfig, backend=None) -> Decision:
    if cfg.classifier == "llm" and backend is not None and cfg.classifier_model:
        return classify_llm(text, cfg, backend)
    return classify_heuristic(text, cfg)
