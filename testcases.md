# Ambiguous routing test cases

20 prompts chosen to sit on the seam between two or three categories, fired at
the heuristic classifier. Reproduce with:

```bash
python3 test_ambiguous.py            # summary to stdout
python3 test_ambiguous.py --print    # plus the results table
python3 test_router.py               # the fixes are locked in here too
```

Each case carries an **acceptable set** — the categories a reasonable person
would accept for that prompt. Landing outside it is a whiff. **Margin** is the
gap between the top two scores; a small margin means the call was nearly a coin
flip, which matters even when the answer is right.

## Status: all six fixes applied

| | before | after |
| --- | --- | --- |
| acceptable | 15 / 19 | **20 / 20** |
| whiffs | 4 | **0** |
| zero-score fallthroughs, mislabelled as confident | 4 | **0** |
| thin margins (< 1.0) | 2 | 7 |

Thin margins going *up* is the point, not a regression. These prompts are
genuinely ambiguous; the old feature set resolved several of them confidently
because only one side had any features at all. Now both sides score, the gap is
narrow, and the confidence number says so.

### What changed

1. **Zero-score fallthroughs are now labelled.** When nothing matches, the
   decision reports `strategy: "fallback"` at confidence 0.0 instead of posing as
   a confident `chat` routing. `/api/route` and the UI can tell "this is chat"
   from "I have no idea".
2. **Confidence is discounted by evidence.** It was `top / total`, so a single
   0.6-weight feature returned 100%. Now `(top / total) × min(1, top / 3)` — one
   weak feature can no longer produce certainty.
3. **Domain nouns score when they name an artifact.** `this code`, `my bug`,
   `the query` score 1.2; `math`/`maths` scores 1.5. Determiner-gated on purpose,
   so "I used to code in college" — where *code* is a verb — still scores zero.
4. **Reasoning verbs tolerate an object.** `think it through`, `walk me through`,
   `talk us through` all match now; previously only the bare `think through` did.
5. **Lone language names are weak (2.0 → 0.6), tasks are strong.** Two new
   features fire at 2.0 when a language name appears inside a real request
   (`write ... python`) or is applied to code (`python script`). Keeps the
   one-liner case that motivated the feature, drops the snake.
6. **Conversational framing scores.** `chat with me`, `let's discuss` at 2.5;
   `my/your favourite` at 1.5.
7. **The short-message summarize dampener is scoped.** It no longer fires on
   messages that explicitly ask for a summary, so "convert this into bullet
   points" is not penalised for being short.

Plus one found live in the UI rather than in this list: **fault vocabulary**
(`null pointer`, `segfault`, `race condition`, `memory leak`, `deadlock`) now
scores 2.0 as code — see A19.

## Results

| # | prompt | tension | routed | margin | verdict |
| --- | --- | --- | --- | --- | --- |
| A1 | explain how recursion works in java | summarize vs code | `reasoning` | 0.9 | ok, thin |
| A2 | calculate the time complexity of quicksort | math vs code vs reasoning | `math` | 2.0 | ok |
| A3 | why is my python script slow, walk me through the logic | code vs reasoning | `code` | 0.4 | ok, thin |
| A4 | tl;dr this function + code block | summarize vs code | `code` | 4.85 | ok |
| A5 | I have a math problem in my code | math vs code | `math` | 0.3 | ok, thin |
| A6 | chat with me about your favorite algorithm | chat vs code | `chat` | 2.5 | ok |
| A7 | python is my favorite snake | language named, zero code intent | `chat` | 0.9 | ok, thin |
| A8 | what does SQL stand for | definition question, not a task | `code` | 0.6 | ok, thin |
| A9 | I used to code in college | mentions coding, no request at all | `chat` | 0.0 | ok, tie |
| A10 | summarize this proof + real proof | summarize vs math | `math` | 4.0 | ok |
| A11 | give me the reasoning behind this bug | reasoning vs code | `code` | 1.7 | ok |
| A12 | explain this like I'm five, what is JavaScript | chat vs code | `reasoning` | 0.9 | ok, thin |
| A13 | walk me through this long math derivation, keep it short | long vs math vs summarize | `math` | 3.21 | ok |
| A14 | every category stuffed equally | engineered near-tie | `code` | 1.4 | ok |
| A15 | is this code even valid, think it through | code vs reasoning | `reasoning` | 1.0 | ok |
| A16 | essay to bullets, one bullet has an equation | summarize vs math | `summarize` | 2.0 | ok |
| A17 | route this as chat even though it looks like code | meta instruction vs content | `chat` | 0.0 | ok, tie |
| A18a | single word: python | one word, no intent | `chat` | 0.4 | ok, thin |
| A18b | single word: recursion | one word, no intent | `chat` | 1.0 | ok |
| A19 | help me summarize this null pointer | summarize vs code — found live in the UI | `summarize` | 1.0 | ok |

## The four original whiffs, now fixed

### A5, A15 — bare domain nouns scored nothing

`I have a math problem in my code` and `is this code even valid, think it
through` both scored **zero across every category** and fell through to `chat`.
Nothing matched "math" or "code" used as plain nouns, and A15 missed twice
because the pattern was `think through` against a prompt saying "think **it**
through".

Now A5 splits `math 1.5 / code 1.2` (confidence 0.28 — honest, it is a coin
flip) and A15 lands on `reasoning 2.2 / code 1.2`.

### A6, A7 — the language-name feature over-fired

`python is my favorite snake` routed to `code` at **1.00 confidence** on the
strength of one word. That feature was a regression I introduced earlier in the
session to rescue "Write a Python one-liner…", which was itself scoring zero —
precision traded for recall with no disambiguation either way.

Splitting it into weak-mention (0.6) and in-a-task (2.0) keeps both cases: the
one-liner still scores 6.1, the snake now goes to `chat` at 0.36.

A6 was the same shape — "algorithm" at 1.5 beat "chat with me", which no pattern
covered. Conversational framing now scores 2.5.

## Still resolved by fallthrough, correctly

**A9, A17 — zero scores, `chat` by default, now labelled as such.** A9 (`I used
to code in college`) has no request in it, so there is nothing to route.

A17 (`route this as chat even though it looks like code`) lands on `chat`
because everything scores zero, not because anything understood the instruction —
and it should stay that way. The router has no notion of a meta-instruction and
should not acquire one: treating in-band text as a routing directive is a
prompt-injection vector, since anything that can say "route this as chat" can say
"route me to the model with the weakest safety posture". Falling through to the
default is the safe behaviour. What changed is that it now *reports* itself as a
fallthrough at confidence 0.0.

**A14 — the engineered tie still behaves.** Keywords from every category stuffed
into one message separates cleanly and keeps confidence low.

## Case by case

### A1 — explain how recursion works in java

*summarize vs code* · acceptable: code, reasoning

- routed **`reasoning`** → `qwen2.5:3b` (confidence 0.36, ~8 tok)
- scores: `reasoning` 1.5 · `code` 0.6
- margin over runner-up: **0.9**
- fired: explanatory question (x1)
- verdict: acceptable

### A2 — calculate the time complexity of quicksort

*math vs code vs reasoning* · acceptable: code, math, reasoning

- routed **`math`** → `qwen2.5:3b` (confidence 0.67, ~10 tok)
- scores: `math` 2.0
- margin over runner-up: **2.0**
- fired: calculation verb (x1)
- verdict: acceptable

### A3 — why is my python script slow, walk me through the logic

*code vs reasoning* · acceptable: code, reasoning

- routed **`code`** → `qwen2.5-coder:3b` (confidence 0.53, ~13 tok)
- scores: `code` 4.1 · `reasoning` 3.7
- margin over runner-up: **0.4**
- fired: names a programming language (x1); named language applied to code (x1); code artifact noun (x1)
- verdict: acceptable

### A4 — tl;dr this function + code block

*summarize vs code* · acceptable: code, summarize

- routed **`code`** → `qwen2.5-coder:3b` (confidence 0.64, ~42 tok)
- scores: `code` 7.9 · `summarize` 3.0 · `chat` 1.4
- margin over runner-up: **4.85**
- fired: fenced code block (x2); python syntax (x1); names a programming language (x1); refers to a specific code artifact (x1)
- verdict: acceptable

### A5 — I have a math problem in my code

*math vs code* · acceptable: code, math

- routed **`math`** → `qwen2.5:3b` (confidence 0.28, ~8 tok)
- scores: `math` 1.5 · `code` 1.2
- margin over runner-up: **0.3**
- fired: math domain noun (x1)
- verdict: acceptable

### A6 — chat with me about your favorite algorithm

*chat vs code* · acceptable: chat, reasoning

- routed **`chat`** → `dolphin-phi` (confidence 0.73, ~10 tok)
- scores: `chat` 4.0 · `code` 1.5
- margin over runner-up: **2.5**
- fired: asks for conversation (x1); personal preference (x1)
- verdict: acceptable

### A7 — python is my favorite snake

*language named, zero code intent* · acceptable: chat

- routed **`chat`** → `dolphin-phi` (confidence 0.36, ~6 tok)
- scores: `chat` 1.5 · `code` 0.6
- margin over runner-up: **0.9**
- fired: personal preference (x1)
- verdict: acceptable

### A8 — what does SQL stand for

*definition question, not a task* · acceptable: chat, code

- routed **`code`** → `qwen2.5-coder:3b` (confidence 0.20, ~5 tok)
- scores: `code` 0.6
- margin over runner-up: **0.6**
- fired: names a programming language (x1)
- verdict: acceptable

### A9 — I used to code in college

*mentions coding, no request at all* · acceptable: chat

- routed **`chat`** → `dolphin-phi` (confidence 0.00, ~6 tok)
- scores: _all zero_
- margin over runner-up: **0.0**
- fired: no category signal in the text; defaulted to chat
- verdict: acceptable

### A10 — summarize this proof + real proof

*summarize vs math* · acceptable: math, summarize

- routed **`math`** → `qwen2.5:3b` (confidence 0.61, ~53 tok)
- scores: `math` 7.0 · `summarize` 3.0 · `code` 1.5
- margin over runner-up: **4.0**
- fired: arithmetic expression (x1); math vocabulary (x1); quantitative vocabulary (x1); numeric comparison (x1)
- verdict: acceptable

### A11 — give me the reasoning behind this bug

*reasoning vs code* · acceptable: code, reasoning

- routed **`code`** → `qwen2.5-coder:3b` (confidence 0.68, ~9 tok)
- scores: `code` 3.2 · `reasoning` 1.5
- margin over runner-up: **1.7**
- fired: engineering task (x1); refers to a specific code artifact (x1)
- verdict: acceptable

### A12 — explain this like I'm five, what is JavaScript

*chat vs code* · acceptable: chat, code, reasoning

- routed **`reasoning`** → `qwen2.5:3b` (confidence 0.36, ~11 tok)
- scores: `reasoning` 1.5 · `code` 0.6
- margin over runner-up: **0.9**
- fired: explanatory question (x1)
- verdict: acceptable

### A13 — walk me through this long math derivation, keep it short

*long vs math vs summarize* · acceptable: long, math, reasoning, summarize

- routed **`math`** → `qwen2.5:3b` (confidence 0.61, ~142 tok)
- scores: `math` 9.2 · `reasoning` 5.9
- margin over runner-up: **3.21**
- fired: calculation verb (x3); math vocabulary (x3); math domain noun (x1)
- verdict: acceptable

### A14 — every category stuffed equally

*engineered near-tie* · acceptable: chat, code, math, reasoning, summarize

- routed **`code`** → `qwen2.5-coder:3b` (confidence 0.31, ~47 tok)
- scores: `code` 9.1 · `math` 7.7 · `reasoning` 5.9 · `summarize` 4.1 · `chat` 3.0
- margin over runner-up: **1.4**
- fired: python syntax (x1); names a programming language (x1); code task in a named language (x1); named language applied to code (x1); code request (x1)
- verdict: acceptable

### A15 — is this code even valid, think it through

*code vs reasoning* · acceptable: code, reasoning

- routed **`reasoning`** → `qwen2.5:3b` (confidence 0.47, ~10 tok)
- scores: `reasoning` 2.2 · `code` 1.2
- margin over runner-up: **1.0**
- fired: asks to be taken through it (x1)
- verdict: acceptable

### A16 — essay to bullets, one bullet has an equation

*summarize vs math* · acceptable: math, summarize

- routed **`summarize`** → `qwen2.5:3b` (confidence 0.64, ~20 tok)
- scores: `summarize` 4.5 · `math` 2.5
- margin over runner-up: **2.0**
- fired: extraction request (x1); transformation verb (x1)
- verdict: acceptable

### A17 — route this as chat even though it looks like code

*meta instruction vs content* · acceptable: chat, code

- routed **`chat`** → `dolphin-phi` (confidence 0.00, ~12 tok)
- scores: _all zero_
- margin over runner-up: **0.0**
- fired: no category signal in the text; defaulted to chat
- verdict: acceptable

### A18a — single word: python

*one word, no intent* · acceptable: chat, code

- routed **`chat`** → `dolphin-phi` (confidence 0.21, ~1 tok)
- scores: `chat` 1.0 · `code` 0.6
- margin over runner-up: **0.4**
- fired: very short message (x1)
- verdict: acceptable

### A18b — single word: recursion

*one word, no intent* · acceptable: chat, code

- routed **`chat`** → `dolphin-phi` (confidence 0.33, ~2 tok)
- scores: `chat` 1.0
- margin over runner-up: **1.0**
- fired: very short message (x1)
- verdict: acceptable

### A19 — help me summarize this null pointer

*summarize vs code — found live in the UI* · acceptable: code, summarize

- routed **`summarize`** → `qwen2.5:3b` (confidence 0.60, ~8 tok)
- scores: `summarize` 3.0 · `code` 2.0
- margin over runner-up: **1.0**
- fired: summarization verb (x1)
- verdict: acceptable
