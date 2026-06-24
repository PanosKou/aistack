# Onyx Deployment

This directory contains the local Onyx Docker Compose deployment used by this stack.

Use it with the network override so Onyx can reach the local gateway:

```bash
cd onyx/deployment
cp env.template .env
docker compose -f docker-compose.yml -f docker-compose.network.yml up -d
```

The override attaches the `api_server` and `background` services to the external `ainet` network. Configure Onyx with this OpenAI-compatible provider:

```text
Base URL: http://langgraph-gateway:8000/v1
API key:  local-dummy-key
Models:   onyx-auto, onyx-fast, onyx-code, onyx-reason
```

The gateway is defined in `langgraph/` at the repository root.
