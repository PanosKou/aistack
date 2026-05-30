# Local AI Gateway for Onyx → Ollama

This project is a small local gateway that lets Onyx call local Ollama models through an OpenAI-compatible API while the gateway itself calls Ollama's native `/api/chat` endpoint internally.

It is intentionally simple: one Python file, one Docker Compose service, one environment file, and one smoke-test script.

## Architecture

```text
User / Browser
  ↓
Onyx on PC-B
  ↓
Local AI Gateway on PC-B
  - exposes OpenAI-compatible /v1/models
  - exposes OpenAI-compatible /v1/chat/completions
  - routes model aliases
  - calls native Ollama /api/chat internally
  ↓
Ollama ROCm on PC-A
  ↓
RX 6700 XT 12GB
```

Current model routing:

```text
onyx-auto
  ├── code/devsecops prompts      → qwen2.5-coder:14b
  ├── reasoning/architecture      → deepseek-r1:14b
  └── normal/general prompts      → llama3.1:8b

onyx-fast                         → llama3.1:8b
onyx-code                         → qwen2.5-coder:14b
onyx-reason                       → deepseek-r1:14b
```

## Why this gateway exists

Onyx expects an OpenAI-compatible provider, so the gateway exposes:

```text
GET  /v1/models
POST /v1/chat/completions
```

Ollama has an OpenAI-compatible `/v1/chat/completions` endpoint, but in this environment `qwen2.5-coder:14b` produced corrupted output through that path. The native Ollama `/api/chat` path produced clean output. Therefore, the gateway keeps the Onyx-facing API OpenAI-compatible but calls Ollama's native API internally:

```text
Onyx → Gateway /v1/chat/completions → Ollama /api/chat
```

## Repository layout

Recommended directory:

```text
~/aistack/langgraph/
├── README.md
├── main.py
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .env
└── smoke.sh
```

### `main.py`

The gateway application.

Responsibilities:

- starts a FastAPI service
- exposes `/health`
- exposes `/ready`
- exposes `/v1/models`
- exposes `/v1/chat/completions`
- accepts OpenAI-style chat requests from Onyx
- routes model aliases such as `onyx-code` or `onyx-reason`
- converts OpenAI-style messages into native Ollama `/api/chat` messages
- converts Ollama native responses back into OpenAI-compatible responses
- supports both streaming and non-streaming responses

The code is kept in one file on purpose. This is a personal/local project and the gateway currently has only one real job: model routing plus API adaptation.

### `requirements.txt`

Python dependencies for the gateway container.

Expected contents:

```txt
fastapi
uvicorn[standard]
httpx
pydantic
python-dotenv
```

Dependency purpose:

```text
fastapi            HTTP API framework
uvicorn[standard]  ASGI server for FastAPI
httpx              async HTTP client used to call Ollama
pydantic           request validation / typed request models
python-dotenv      loads .env configuration
```

### `Dockerfile`

Builds the gateway container.

Responsibilities:

- uses `python:3.12-slim`
- installs `curl` for container healthchecks
- installs Python dependencies from `requirements.txt`
- copies `main.py`
- starts the API with Uvicorn

### `compose.yaml`

Runs the gateway service.

Responsibilities:

- builds the local gateway image
- runs the container as `langgraph-gateway`
- loads environment variables from `.env`
- publishes the gateway only on host loopback: `127.0.0.1:8000`
- attaches the container to the external Docker network `ai_backend`
- defines a Docker healthcheck against `/health`

Important detail:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

This means the gateway is reachable from the PC-B host at:

```text
http://127.0.0.1:8000
```

and from other containers on the `ai_backend` Docker network at:

```text
http://langgraph-gateway:8000
```

It is not exposed to the LAN directly.

### `.env`

Runtime configuration.

Example:

```env
LOG_LEVEL=INFO

OLLAMA_BASE_URL=http://192.168.1.27:11434
OLLAMA_NUM_CTX=4096
OLLAMA_KEEP_ALIVE=5m
OLLAMA_HTTP_TIMEOUT=300

LOCAL_FAST_MODEL=llama3.1:8b
LOCAL_CODE_MODEL=qwen2.5-coder:14b
LOCAL_REASON_MODEL=deepseek-r1:14b

FAST_MAX_TOKENS=256
CODE_MAX_TOKENS=512
REASON_MAX_TOKENS=768
```

Important: `OLLAMA_BASE_URL` must not include `/v1`.

Correct:

```env
OLLAMA_BASE_URL=http://192.168.1.27:11434
```

Wrong:

```env
OLLAMA_BASE_URL=http://192.168.1.27:11434/v1
```

### `smoke.sh`

A small local test script.

Responsibilities:

- checks `/health`
- checks `/ready`
- checks `/v1/models`
- checks routing via `/debug/route`
- sends a streaming test request through `/v1/chat/completions`

This replaces ad-hoc commands such as exec-ing Python into unrelated containers.

## Token and context settings

### `OLLAMA_NUM_CTX`

```env
OLLAMA_NUM_CTX=4096
```

This controls the request context window passed to Ollama as:

```json
"num_ctx": 4096
```

This is the input/conversation memory budget. It controls how much prompt and prior context the model can use.

For the RX 6700 XT 12GB, `4096` is the stable baseline. Do not increase this until the full Onyx path is stable.

### `FAST_MAX_TOKENS`, `CODE_MAX_TOKENS`, `REASON_MAX_TOKENS`

```env
FAST_MAX_TOKENS=256
CODE_MAX_TOKENS=512
REASON_MAX_TOKENS=768
```

These are default output-token caps. They are used only when the caller does not provide `max_tokens` or `max_completion_tokens`.

They map to Ollama's native generation option:

```json
"num_predict": ...
```

Meaning:

```text
FAST_MAX_TOKENS    default output cap for llama3.1:8b
CODE_MAX_TOKENS    default output cap for qwen2.5-coder:14b
REASON_MAX_TOKENS  default output cap for deepseek-r1:14b
```

They do not control the input context window. They only control how much text the model may generate by default.

Per-request override example:

```json
{
  "model": "onyx-code",
  "max_tokens": 160,
  "messages": [
    {
      "role": "user",
      "content": "Return a short Bash function."
    }
  ]
}
```

In that case, the gateway sends:

```json
"num_predict": 160
```

instead of using `CODE_MAX_TOKENS=512`.

## Prerequisites

On PC-A / Ollama host:

- Ollama is running in Docker with ROCm
- Ollama is reachable from PC-B over the private LAN
- Ollama is published on the LAN IP, for example:

```text
http://192.168.1.27:11434
```

On PC-B / Onyx and gateway host:

- Docker and Docker Compose are installed
- PC-B can reach PC-A Ollama:

```bash
curl http://192.168.1.27:11434/api/tags
```

Create the local Docker network used by Onyx and the gateway:

```bash
sudo docker network create ai_backend 2>/dev/null || true
```

## Running the gateway

From the gateway directory:

```bash
cd ~/aistack/langgraph
```

Build and start:

```bash
sudo docker compose up -d --build
```

Check container status:

```bash
sudo docker compose ps
```

Follow logs:

```bash
sudo docker logs -f langgraph-gateway
```

Check health:

```bash
curl http://127.0.0.1:8000/health | jq
```

Check readiness, which verifies Ollama is reachable:

```bash
curl http://127.0.0.1:8000/ready | jq
```

List OpenAI-compatible models exposed by the gateway:

```bash
curl http://127.0.0.1:8000/v1/models | jq
```

Run the smoke test:

```bash
./smoke.sh
```

## Docker Compose operations

Start:

```bash
sudo docker compose up -d
```

Rebuild after Python or Dockerfile changes:

```bash
sudo docker compose up -d --build --force-recreate
```

Stop:

```bash
sudo docker compose down
```

View logs:

```bash
sudo docker logs -f langgraph-gateway
```

Inspect healthcheck state:

```bash
sudo docker inspect langgraph-gateway --format '{{json .State.Health}}' | jq
```

## API examples

### Health

```bash
curl http://127.0.0.1:8000/health | jq
```

### Readiness

```bash
curl http://127.0.0.1:8000/ready | jq
```

### Models

```bash
curl http://127.0.0.1:8000/v1/models | jq
```

### Streaming chat completion

```bash
curl -N --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
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
```

### Non-streaming chat completion

```bash
curl --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-code",
    "stream": false,
    "max_tokens": 160,
    "messages": [
      {
        "role": "user",
        "content": "Return only a minimal Bash function named check_docker_compose that checks whether Docker Compose is installed."
      }
    ],
    "temperature": 0.1
  }' | jq
```

### Route debug

```bash
curl http://127.0.0.1:8000/debug/route \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-auto",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python script that validates SHA256 hashes."
      }
    ]
  }' | jq
```

Expected route:

```text
qwen2.5-coder:14b
```

## Onyx integration

Onyx should not call Ollama directly. It should call the gateway.

Inside Onyx, configure a custom/OpenAI-compatible model provider:

```text
Provider type: OpenAI-compatible / Custom
Base URL:      http://langgraph-gateway:8000/v1
API key:       local-dummy-key
```

Add these model IDs:

```text
onyx-auto
onyx-fast
onyx-code
onyx-reason
```

Recommended default:

```text
onyx-auto
```

## Connecting Onyx containers to the gateway

The gateway is on the `ai_backend` Docker network. Onyx backend containers must also be attached to that network.

Example Onyx Compose override:

```yaml
services:
  api_server:
    networks:
      - default
      - ai_backend

  background:
    networks:
      - default
      - ai_backend

networks:
  ai_backend:
    external: true
```

Start Onyx with both files:

```bash
sudo docker compose \
  -f docker-compose.yml \
  -f docker-compose.ai-backend.yml \
  up -d
```

Clean network-level test from PC-B:

```bash
sudo docker run --rm \
  --network ai_backend \
  curlimages/curl:8.20.0 \
  -fsS http://langgraph-gateway:8000/ready
```

This verifies that containers on `ai_backend` can resolve and reach the gateway.

## Troubleshooting

### Gateway is running but `/ready` fails

Check that PC-B can reach Ollama on PC-A:

```bash
curl http://192.168.1.27:11434/api/tags
```

Check `.env`:

```bash
grep OLLAMA_BASE_URL .env
```

Expected:

```text
OLLAMA_BASE_URL=http://192.168.1.27:11434
```

Not:

```text
OLLAMA_BASE_URL=http://192.168.1.27:11434/v1
```

### Onyx cannot reach the gateway

Check gateway container is on `ai_backend`:

```bash
sudo docker inspect langgraph-gateway --format '{{json .NetworkSettings.Networks}}' | jq
```

Check Onyx `api_server` is also on `ai_backend`:

```bash
sudo docker inspect <api_server_container_id> --format '{{json .NetworkSettings.Networks}}' | jq
```

Or run the disposable curl test:

```bash
sudo docker run --rm \
  --network ai_backend \
  curlimages/curl:8.20.0 \
  -fsS http://langgraph-gateway:8000/ready
```

### Qwen output is corrupted

Do not point Onyx or the gateway to Ollama's `/v1/chat/completions` endpoint.

Use this internally:

```text
http://192.168.1.27:11434/api/chat
```

The gateway already does this when `OLLAMA_BASE_URL` has no `/v1`.

### Check GPU usage on the Ollama host

On PC-A:

```bash
ollama ps
```

Expected for code route:

```text
qwen2.5-coder:14b    100% GPU    4096
```

## Current validated behavior

Validated working path:

```text
Gateway /v1/chat/completions
  → native Ollama /api/chat
  → qwen2.5-coder:14b
  → clean streamed OpenAI-compatible chunks
```

Known-bad path in this environment:

```text
Direct Ollama /v1/chat/completions
  → qwen2.5-coder:14b
  → corrupted chunks
```

Keep the gateway between Onyx and Ollama.

## Notes

This service is deliberately not exposed to the LAN directly. It binds to:

```text
127.0.0.1:8000
```

Onyx reaches it through Docker DNS:

```text
http://langgraph-gateway:8000/v1
```

Ollama on PC-A must remain restricted to trusted LAN hosts only.
