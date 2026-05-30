#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

echo "== Health =="
curl -fsS "${BASE_URL}/health" | jq

echo
echo "== Readiness =="
curl -fsS "${BASE_URL}/ready" | jq '.status'

echo
echo "== Models =="
curl -fsS "${BASE_URL}/v1/models" | jq -r '.data[].id'

echo
echo "== Route debug =="
curl -fsS "${BASE_URL}/debug/route" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-code",
    "stream": true,
    "max_tokens": 160,
    "messages": [
      {
        "role": "user",
        "content": "Return only a minimal Bash function named check_docker_compose that checks whether Docker Compose is installed."
      }
    ],
    "temperature": 0.1
  }' | jq '.route'

echo
echo "== Streaming test =="
curl -fsSN --max-time 180 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-code",
    "stream": true,
    "max_tokens": 160,
    "messages": [
      {
        "role": "user",
        "content": "Return only a minimal Bash function named check_docker_compose that checks whether Docker Compose is installed."
      }
    ],
    "temperature": 0.1
  }'
