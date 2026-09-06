# Local Model Router

An OpenAI-compatible API that reads each incoming prompt, decides what kind of
request it is, and routes it to the right **small local model** through Ollama.

The point is the decision: `POST /api/route` tells you which model a prompt
*would* go to, and why, without spending a token.

```
prompt ──► classifier ──► category ──► model chain ──► Ollama
                             │              │
                        code/math/…    first installed model wins,
                                       the rest stay as fallbacks
```

## Run it

```bash
./run.sh                      # creates .venv, installs deps, serves on :8000
open http://localhost:8000    # the patch-bay UI
./smoke.sh                    # end-to-end check against a running router
```

Requires [Ollama](https://ollama.com) with at least one small model:

```bash
ollama serve
ollama pull qwen2.5:3b
```

## Endpoints

| Method | Path | What it does |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness, mode, backend reachability |
| `GET` | `/api/routes` | The resolved route table |
| `GET` | `/api/v1/models` | `auto`, every category, every installed model |
| `GET` | `/api/stats` | Per-model requests, latency, tokens, fallbacks |
| `POST` | `/api/route` | The routing decision only — no generation |
| `POST` | `/api/tokenomics` | Educational input/output token estimate — no generation |
| `POST` | `/api/v1/chat/completions` | OpenAI-compatible, `stream: true` supported |
| `POST` | `/api/admin/reload` | Re-read config, re-detect installed models |

Every path is also mounted without the `/api` prefix, so an OpenAI client can
use `http://localhost:8000/v1` as its base URL directly. Interactive docs are at
`/docs`.

```bash
curl -s localhost:8000/api/route -H 'Content-Type: application/json' \
  -d '{"prompt":"fix this failing pytest"}'
# => {"decision":{"category":"code",...},"selected_model":"qwen2.5:3b",...}
```

### Choosing the model yourself

`model` accepts three things:

- `"auto"` — classify the prompt (the default)
- a category (`"code"`, `"math"`, `"reasoning"`, `"summarize"`, `"long"`, `"chat"`) — force that line
- an installed model id (`"phi3:mini"`) — pin it, skipping the router

The `X-Router-Hint` header forces a category too, which is handy for clients
that will not let you change the `model` field.

## How routing works

**Classifier.** The default is a weighted regex scorer: ~30 features across the
categories, with diminishing returns on repeats so a term repeated ten times
confirms rather than dominates. It runs in well under a millisecond and every
decision comes back with the features that fired. Any request over
`long_context_tokens` (default 1500) is forced onto the `long` line regardless
of score.

Set `ROUTER_CLASSIFIER=llm` to have the smallest installed model vote on the
category instead. It is constrained to one word and falls back to the heuristic
on anything unexpected, so a bad vote degrades rather than breaks.

**Fallback chains.** Each category maps to an ordered list of models. At startup
the list is filtered against what Ollama actually has installed, so an
uninstalled preference is skipped instead of failing at request time. If the
chosen model errors mid-request, the router tries the next one and reports the
attempts in `router.attempts`. On a stream, fallback applies only before the
first token — once bytes are on the wire the client is committed.

**Tokenomics.** `POST /api/tokenomics` (and the `tokenomics` field already
included in every `/route` response) estimates how many tokens a prompt costs
*before* anything runs. Input size is a blend of `chars/4` and `words*1.3` —
two familiar rules of thumb, averaged so both prose and dense/code text land
close. Output size has no ground truth yet, so it is a `low`/`typical`/`high`
forecast built from a per-category ratio applied to the input estimate (a
`summarize` reply tends to shrink the input, a `code` reply tends to grow it),
capped by `max_tokens`/`num_ctx` when either is set. It is a classroom-grade
approximation, not a tokenizer — see `router/tokenomics.py` for the full
reasoning and caveats.

## Configuration

Environment variables, or a `router.config.json` next to the app:

| Variable | Default | |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Where the runtime lives |
| `ROUTER_CLASSIFIER` | `heuristic` | or `llm` |
| `ROUTER_LONG_TOKENS` | `1500` | Long-context threshold |
| `ROUTER_OFFLINE` | unset | Skip the backend probe; routing only |

```json
{
  "routes": { "code": ["qwen2.5-coder:3b", "qwen2.5:3b"] },
  "options": { "code": { "temperature": 0.05, "num_ctx": 16384 } }
}
```

## Deploying

The cloud has no Ollama, so the Vercel deployment runs **routing-only**: the UI
and `/api/route` work and show you exactly which model a prompt maps to, while
`/api/v1/chat/completions` returns a 503 explaining that generation needs a
local runtime. Point the UI's *api base* field at `http://localhost:8000` to get
real completions from the hosted page.

- `public/` — the UI, served statically
- `api/index.py` — the FastAPI app as a Vercel Python function
- `.github/workflows/ci.yml` — tests on every push, preview deploys on PRs,
  production on `main`

CI needs three repository secrets: `VERCEL_TOKEN`, `VERCEL_ORG_ID`,
`VERCEL_PROJECT_ID`. The last two come from `vercel link` (they land in
`.vercel/project.json`).

## Tests

```bash
python3 test_router.py       # engine: no dependencies at all
python3 test_tokenomics.py   # token estimator: no dependencies at all
python3 test_api.py          # FastAPI layer, needs requirements-dev.txt
```

83 tests covering classification, chain resolution, fallback behaviour,
streaming, token estimation, the OpenAI response shape, and both local and
cloud modes. None of the suites talk to a real model — they run against a
fake backend.

## Layout

```
router/          the engine, standard library only
  config.py      route table, model resolution
  classifier.py  heuristic + llm classifiers
  engine.py      planning, execution, fallback
  backends.py    Ollama client over urllib
  tokenomics.py  educational input/output token estimator
  server.py      a stdlib-only server, if you want zero dependencies
app/main.py      the FastAPI app
api/index.py     Vercel entrypoint
public/          the UI
```

`router/` has no third-party imports, so `python3 -m router` serves the same API
with nothing installed at all.
