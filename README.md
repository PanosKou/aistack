# Local Onyx + Ollama AI Stack

This repository contains a trimmed local AI setup with three parts:

- `onyx/`: Onyx Docker Compose deployment and network override.
- `langgraph/`: a small FastAPI gateway that exposes an OpenAI-compatible API to Onyx.
- `ollama/`: Ollama ROCm Docker Compose service for local AMD GPU inference.

The gateway keeps Onyx configured as an OpenAI-compatible client while sending requests to Ollama's native `/api/chat` endpoint internally.

## Architecture

```text
User / Browser
  -> Onyx
  -> langgraph-gateway /v1/chat/completions
  -> Ollama /api/chat
  -> local model
```

The shared Docker network is named `ainet`.

## Model Routing

The gateway exposes these model IDs to Onyx:

```text
onyx-auto
onyx-fast
onyx-code
onyx-reason
```

Default routing:

```text
onyx-fast    -> LOCAL_FAST_MODEL
onyx-code    -> LOCAL_CODE_MODEL
onyx-reason  -> LOCAL_REASON_MODEL
onyx-auto    -> keyword-routed fast/code/reasoning model
```

Tool/function calling requests are routed to the code/tool model by default, even when the caller asks for `onyx-auto`, `onyx-fast`, or `onyx-reason`.

## Repository Layout

```text
.
├── langgraph/
│   ├── main.py
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── requirements.txt
│   ├── .env.example
│   └── lang_smoke.sh
├── ollama/
│   └── docker-compose.yaml
└── onyx/
    ├── README.md
    ├── data/nginx/
    └── deployment/
        ├── docker-compose.yml
        ├── docker-compose.network.yml
        └── env.template
```

## Prerequisites

- Docker and Docker Compose.
- AMD ROCm-compatible host for the included Ollama service, or a trusted remote Ollama host.
- The Ollama models referenced by `langgraph/.env` must be pulled before use.

Create the shared network once:

```bash
docker network create ainet 2>/dev/null || true
```

## Configure Gateway

Create a local gateway env file:

```bash
cp langgraph/.env.example langgraph/.env
```

If Ollama runs from this repo on the same Docker network, keep:

```env
OLLAMA_BASE_URL=http://ollama:11434
```

If Ollama runs on another trusted machine, set the LAN URL instead:

```env
OLLAMA_BASE_URL=http://192.168.1.27:11434
```

Do not include `/v1` in `OLLAMA_BASE_URL`; the gateway calls Ollama's native `/api/chat`.

## Start Ollama

```bash
cd ollama
docker compose up -d
```

Pull the models you configured:

```bash
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull qwen2.5-coder:14b
docker exec -it ollama ollama pull deepseek-r1:14b
```

## Start Gateway

```bash
cd langgraph
docker compose up -d --build
```

Check it:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/models
```

Run the gateway smoke test:

```bash
cd langgraph
./lang_smoke.sh
```

## Start Onyx

Create a local Onyx env file:

```bash
cp onyx/deployment/env.template onyx/deployment/.env
```

Start Onyx with the network override so the API/background containers can resolve `langgraph-gateway`:

```bash
cd onyx/deployment
docker compose -f docker-compose.yml -f docker-compose.network.yml up -d
```

Inside Onyx, configure a custom/OpenAI-compatible model provider:

```text
Base URL: http://langgraph-gateway:8000/v1
API key:  local-dummy-key
Models:   onyx-auto, onyx-fast, onyx-code, onyx-reason
```

Recommended default model: `onyx-auto`.

## Useful Checks

From the host:

```bash
curl -fsS http://127.0.0.1:8000/ready
```

From the shared Docker network:

```bash
docker run --rm --network ainet curlimages/curl:8.20.0 \
  -fsS http://langgraph-gateway:8000/ready
```

Debug model routing:

```bash
curl -fsS http://127.0.0.1:8000/debug/route \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onyx-auto",
    "messages": [
      {
        "role": "user",
        "content": "Write a Python script that validates SHA256 hashes."
      }
    ]
  }'
```

## Notes

- The gateway is bound to host loopback at `127.0.0.1:8000`.
- Containers on `ainet` reach it at `http://langgraph-gateway:8000`.
- Ollama is exposed by the included compose file on `0.0.0.0:11434`; keep that host restricted to trusted machines.
- Runtime `.env` files are intentionally ignored by git. Use the checked-in examples/templates as starting points.
