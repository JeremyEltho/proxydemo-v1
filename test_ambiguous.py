"""Fire the ambiguous prompts at the classifier and report where it whiffs.

    python3 test_ambiguous.py            # writes testcases.md
    python3 test_ambiguous.py --print    # also dump to stdout

`acceptable` is the set of categories a reasonable person would accept for that
prompt. A routed category outside it is a whiff. `margin` is the gap between the
top two scores — a small margin means the call was nearly a coin flip, which is
worth knowing even when the answer is right.
"""

from __future__ import annotations

import sys

from router.classifier import classify_heuristic
from router.config import RouterConfig

CODE_BLOCK = """```python
def merge(a, b):
    out = []
    while a and b:
        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))
    return out + a + b
```"""

PROOF = """Theorem: the sum of the first n integers is n(n+1)/2.
Proof: let S = 1 + 2 + ... + n. Write S backwards and add termwise:
2S = (1+n) + (2+n-1) + ... = n(n+1). Therefore S = n(n+1)/2. QED"""

DERIVATION = ("Starting from d/dx of x^3, apply the power rule to get 3x^2. "
              "Then integrate 3x^2 with respect to x, which returns x^3 + C. "
              "Now evaluate the definite integral from 0 to 2. ") * 3

STUFFED = ("Write a python function to calculate the derivative, then summarize "
           "the tradeoffs step-by-step, and hey how are you? "
           "def solve(): return 17 * 43 + 12  # compare REST versus gRPC, tl;dr please")

# (id, label, prompt, tension, acceptable categories)
CASES = [
    ("A1", "explain how recursion works in java",
     "explain how recursion works in java", "summarize vs code", {"code", "reasoning"}),
    ("A2", "calculate the time complexity of quicksort",
     "calculate the time complexity of quicksort", "math vs code vs reasoning",
     {"math", "code", "reasoning"}),
    ("A3", "why is my python script slow, walk me through the logic",
     "why is my python script slow, walk me through the logic", "code vs reasoning",
     {"code", "reasoning"}),
    ("A4", "tl;dr this function + code block",
     "tl;dr this function for me\n\n" + CODE_BLOCK, "summarize vs code",
     {"summarize", "code"}),
    ("A5", "I have a math problem in my code",
     "I have a math problem in my code", "math vs code", {"math", "code"}),
    ("A6", "chat with me about your favorite algorithm",
     "chat with me about your favorite algorithm", "chat vs code", {"chat", "reasoning"}),
    ("A7", "python is my favorite snake",
     "python is my favorite snake", "language named, zero code intent", {"chat"}),
    ("A8", "what does SQL stand for",
     "what does SQL stand for", "definition question, not a task", {"chat", "code"}),
    ("A9", "I used to code in college",
     "I used to code in college", "mentions coding, no request at all", {"chat"}),
    ("A10", "summarize this proof + real proof",
     "summarize this proof for me\n\n" + PROOF, "summarize vs math", {"summarize", "math"}),
    ("A11", "give me the reasoning behind this bug",
     "give me the reasoning behind this bug", "reasoning vs code", {"reasoning", "code"}),
    ("A12", "explain this like I'm five, what is JavaScript",
     "explain this like I'm five, what is JavaScript", "chat vs code", {"chat", "code", "reasoning"}),
    ("A13", "walk me through this long math derivation, keep it short",
     "walk me through this long math derivation, keep it short\n\n" + DERIVATION,
     "long vs math vs summarize", {"math", "summarize", "reasoning", "long"}),
    ("A14", "every category stuffed equally", STUFFED, "engineered near-tie",
     {"code", "math", "reasoning", "summarize", "chat"}),
    ("A15", "is this code even valid, think it through",
     "is this code even valid, think it through", "code vs reasoning", {"code", "reasoning"}),
    ("A16", "essay to bullets, one bullet has an equation",
     "convert this essay into bullet points, one of the bullets should have an equation",
     "summarize vs math", {"summarize", "math"}),
    ("A17", "route this as chat even though it looks like code",
     "route this as chat even though it looks like code", "meta instruction vs content",
     {"chat", "code"}),
    ("A18a", "single word: python", "python", "one word, no intent", {"chat", "code"}),
    ("A18b", "single word: recursion", "recursion", "one word, no intent", {"chat", "code"}),
    ("A19", "help me summarize this null pointer",
     "help me summarize this null pointer", "summarize vs code — found live in the UI",
     {"summarize", "code"}),
]


def run():
    cfg = RouterConfig()
    cfg.resolve_static()
    rows = []
    for cid, label, prompt, tension, acceptable in CASES:
        d = classify_heuristic(prompt, cfg)
        ranked = sorted(d.scores.items(), key=lambda kv: -kv[1])
        top, second = ranked[0], ranked[1]
        margin = round(top[1] - second[1], 2)
        rows.append({
            "id": cid, "label": label, "prompt": prompt, "tension": tension,
            "acceptable": acceptable, "routed": d.category,
            "model": (cfg.chain_for(d.category) or ["—"])[0],
            "confidence": d.confidence, "est_tokens": d.est_tokens,
            "forced": d.forced, "reasons": d.reasons,
            "ranked": ranked, "margin": margin,
            "ok": d.category in acceptable,
        })
    return rows


def table(rows):
    out = ["| # | prompt | tension | routed | margin | verdict |",
           "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        verdict = "ok" if r["ok"] else "**whiff**"
        if r["ok"] and r["margin"] == 0:
            verdict = "ok, tie"
        elif r["ok"] and r["margin"] < 1.0:
            verdict = "ok, thin"
        label = r["label"].replace("|", "\\|")
        out.append(f'| {r["id"]} | {label} | {r["tension"]} | `{r["routed"]}` '
                   f'| {r["margin"]} | {verdict} |')
    return "\n".join(out)


def detail(rows):
    out = []
    for r in rows:
        out.append(f'### {r["id"]} — {r["label"]}\n')
        out.append(f'*{r["tension"]}* · acceptable: {", ".join(sorted(r["acceptable"]))}\n')
        scored = " · ".join(f"`{c}` {v:.1f}" for c, v in r["ranked"] if v > 0) or "_all zero_"
        out.append(f'- routed **`{r["routed"]}`** → `{r["model"]}` '
                   f'(confidence {r["confidence"]:.2f}, ~{r["est_tokens"]} tok)')
        out.append(f'- scores: {scored}')
        out.append(f'- margin over runner-up: **{r["margin"]}**')
        if r["forced"]:
            out.append(f'- override: {r["forced"]}')
        if r["reasons"]:
            out.append(f'- fired: {"; ".join(x.split(": ", 1)[-1] for x in r["reasons"])}')
        out.append(f'- verdict: {"acceptable" if r["ok"] else "**WHIFF**"}\n')
    return "\n".join(out)


rows = run()
whiffs = [r for r in rows if not r["ok"]]
ties = [r for r in rows if r["ok"] and r["margin"] == 0]
thin = [r for r in rows if r["ok"] and 0 < r["margin"] < 1.0]

summary = (f"{len(rows)} cases · {len(rows) - len(whiffs)} acceptable · "
           f"{len(whiffs)} whiffs · {len(ties)} outright ties · {len(thin)} thin margins")
print(summary)
for r in whiffs:
    print(f'  WHIFF {r["id"]}: routed {r["routed"]}, wanted one of {sorted(r["acceptable"])}')
for r in ties:
    print(f'  TIE   {r["id"]}: {r["routed"]} (all scores {r["ranked"][0][1]})')

with open("/tmp/_ambiguous_report.md", "w") as fh:
    fh.write(f"<!--SUMMARY:{summary}-->\n\n## Results\n\n{table(rows)}\n\n## Case by case\n\n{detail(rows)}")

if "--print" in sys.argv:
    print(table(rows))
