#!/usr/bin/env bash
# End-to-end check against a running router. Usage: ./smoke.sh [base_url]
set -uo pipefail
BASE="${1:-http://localhost:8000}"

post() {  # post <path> <json>
  curl -s -X POST "$BASE$1" -H 'Content-Type: application/json' -d "$2"
}
json_prompt() { python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1]}))' "$1"; }

echo "== health =="
curl -s "$BASE/api/health" | python3 -m json.tool

echo
echo "== route table =="
curl -s "$BASE/api/routes" | python3 -c '
import json, sys
for cat, e in json.load(sys.stdin)["routes"].items():
    tail = "   fallback: " + " -> ".join(e["fallbacks"]) if e["fallbacks"] else ""
    print("  %-10s -> %-20s%s" % (cat, e["model"], tail))
'

echo
echo "== routing decisions (no tokens spent) =="
while IFS= read -r p; do
  [ -z "$p" ] && continue
  post /api/route "$(json_prompt "$p")" | python3 -c '
import json, sys
d = json.load(sys.stdin)
dec = d["decision"]
print("  %-10s -> %-20s conf %.2f  %5.2fms  | %s" % (
    dec["category"], d["selected_model"], dec["confidence"],
    d["routing_ms"], d["prompt_preview"][:46].replace("\n", " ")))
'
done <<'PROMPTS'
Write a Python function that reverses a linked list
Fix this traceback: KeyError when I import my config module
Calculate 17 * 43 + 12
Compare the trade-offs of REST versus gRPC for our service
Summarize the following article into three bullet points: blah blah blah blah blah blah
hey there, how are you?
PROMPTS

echo
echo "== real completion through the router =="
post /api/v1/chat/completions \
  '{"model":"auto","prompt":"Write a Python one-liner that sums a list of ints.","max_tokens":80}' \
| python3 -c '
import json, sys
d = json.load(sys.stdin)
if "detail" in d:
    print("  ", d["detail"]); raise SystemExit
r = d["router"]
print("  routed %s -> %s in %sms, %d tokens" % (
    r["decision"]["category"], d["model"], r["latency_ms"], d["usage"]["total_tokens"]))
print()
for line in d["choices"][0]["message"]["content"].strip().splitlines()[:12]:
    print("  | " + line)
'

echo
echo "== streaming (first bytes) =="
post /api/v1/chat/completions \
  '{"model":"chat","prompt":"Say hello in five words.","stream":true,"max_tokens":30}' \
| head -c 420

echo
echo
echo "== stats =="
curl -s "$BASE/api/stats" | python3 -m json.tool
