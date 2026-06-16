#!/usr/bin/env bash
set -euo pipefail

BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8092}"
LANGGRAPH_URL="${LANGGRAPH_URL:-http://127.0.0.1:8000/v1}"
SEARXNG_URL="${SEARXNG_URL:-http://127.0.0.1:8080/search}"
N8N_SCRAPE_URL="${N8N_SCRAPE_URL:-http://192.168.1.81:5678/webhook/scrape-url}"
API_KEY="${API_KEY:-local-dummy-key}"
MODEL="${MODEL:-onyx-fast}"

echo "=== systemd services ==="
systemctl is-active --quiet hermes-bridge && echo "hermes-bridge: active" || { echo "hermes-bridge: FAILED"; exit 1; }
systemctl is-active --quiet mcp-hermes-bridge && echo "mcp-hermes-bridge: active" || { echo "mcp-hermes-bridge: FAILED"; exit 1; }

echo
echo "=== hermes-bridge health ==="
curl -fsS "$BRIDGE_URL/health" | jq '.'

echo
echo "=== searxng json search ==="
curl -fsS -X POST "$SEARXNG_URL" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'q=n8n production webhook official docs' \
  --data-urlencode 'format=json' \
  | jq -e '.results | length > 0' >/dev/null
echo "searxng: ok"

echo
echo "=== n8n scrape webhook ==="
curl -fsS -X POST "$N8N_SCRAPE_URL" \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com"}' \
  | jq -e '.title // .[0].title' >/dev/null
echo "n8n scrape-url: ok"

echo
echo "=== langgraph chat completions ==="
curl -fsS "$LANGGRAPH_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_KEY" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"Say ok in one word.\"
      }
    ],
    \"temperature\": 0.2,
    \"max_tokens\": 16,
    \"stream\": false
  }" \
  | jq -e '.choices[0].message.content' >/dev/null
echo "langgraph: ok"

echo
echo "=== full bridge research path ==="
curl -fsS -X POST "$BRIDGE_URL/research" \
  -H 'Content-Type: application/json' \
  -d '{
    "task": "Find official documentation for n8n production webhooks. Return a short summary and source URL.",
    "max_results": 1
  }' \
  | jq -e '.mode == "langgraph" and (.answer | length > 0)' >/dev/null
echo "bridge research path: ok"

echo
echo "SMOKE TEST PASSED"
