import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

# =============================================================================
# Configuration
# =============================================================================

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("local-ai-gateway")


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Using default=%s", name, raw, default)
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r. Using default=%s", name, raw, default)
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

# This gateway calls native Ollama /api/chat internally, not Ollama's /v1 API.
if OLLAMA_BASE_URL.endswith("/v1"):
    OLLAMA_BASE_URL = OLLAMA_BASE_URL.removesuffix("/v1").rstrip("/")

OLLAMA_NUM_CTX = env_int("OLLAMA_NUM_CTX", 4096)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
OLLAMA_HTTP_TIMEOUT = env_float("OLLAMA_HTTP_TIMEOUT", 300.0)

LOCAL_FAST_MODEL = os.getenv("LOCAL_FAST_MODEL", "llama3.2:3b")
LOCAL_CODE_MODEL = os.getenv("LOCAL_CODE_MODEL", "qwen2.5:7b")
LOCAL_REASON_MODEL = os.getenv("LOCAL_REASON_MODEL", "deepseek-r1:7b")

FAST_MAX_TOKENS = env_int("FAST_MAX_TOKENS", 768)
CODE_MAX_TOKENS = env_int("CODE_MAX_TOKENS", 1024)
REASON_MAX_TOKENS = env_int("REASON_MAX_TOKENS", 1024)

# If an OpenAI-compatible client sends tools with a weak alias such as onyx-fast,
# route to the code/tool model. This keeps the UI light for normal chat while
# preserving reliable MCP/tool calling.
ROUTE_TO_CODE_WHEN_TOOLS = env_bool("ROUTE_TO_CODE_WHEN_TOOLS", True)

# Streaming tool calls require OpenAI SSE tool-call delta handling. This gateway
# implements converted SSE chunks by first making a non-stream Ollama tool call.
STREAM_TOOLS_AS_CONVERTED_SSE = env_bool("STREAM_TOOLS_AS_CONVERTED_SSE", True)

# =============================================================================
# Schemas
# =============================================================================


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "onyx-auto"
    messages: list[dict[str, Any]]
    stream: bool | None = False

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None

    max_tokens: int | None = None
    max_completion_tokens: int | None = None

    response_format: dict[str, Any] | None = None

    # OpenAI-compatible tool/function calling input from Onyx.
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class ModelRoute(BaseModel):
    requested_model: str
    target_model: str
    reason: str


# =============================================================================
# App lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        timeout=OLLAMA_HTTP_TIMEOUT,
        connect=10.0,
        read=OLLAMA_HTTP_TIMEOUT,
    )
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=5)
    app.state.http = httpx.AsyncClient(
        base_url=OLLAMA_BASE_URL,
        timeout=timeout,
        limits=limits,
    )
    logger.info("Gateway started. Ollama base URL: %s", OLLAMA_BASE_URL)
    try:
        yield
    finally:
        await app.state.http.aclose()
        logger.info("Gateway stopped")


app = FastAPI(
    title="Local AI Gateway",
    version="0.3.0",
    lifespan=lifespan,
)

# =============================================================================
# Routing
# =============================================================================

CODE_KEYWORDS = [
    "code",
    "python",
    "bash",
    "shell",
    "docker",
    "compose",
    "docker compose",
    "docker-compose",
    "kubernetes",
    "k8s",
    "terraform",
    "ansible",
    "javascript",
    "typescript",
    "golang",
    "rust",
    "cve",
    "devsecops",
    "ci/cd",
    "pipeline",
    "yaml",
    "regex",
    "script",
    "function",
    "debug",
    "stack trace",
    "error log",
    "vulnerability",
    "semgrep",
    "trivy",
    "helm",
]

REASONING_KEYWORDS = [
    "reason",
    "analyze",
    "analysis",
    "architecture",
    "trade-off",
    "tradeoff",
    "root cause",
    "diagnose",
    "plan",
    "strategy",
    "compare",
    "security review",
    "threat model",
    "design",
    "decision",
    "evaluate",
    "investigate",
    "why",
]


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if not isinstance(item, dict):
                continue

            if isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item.get("content"), str):
                parts.append(item["content"])
            elif item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item.get("type") == "image_url":
                parts.append("[image omitted: text-only local gateway]")

        return "\n".join(parts)

    if content is None:
        return ""

    return str(content)


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        extract_text(message.get("content", ""))
        for message in messages
    ).lower()


def has_tools(request: ChatRequest) -> bool:
    return bool(request.tools) and request.tool_choice != "none"


def route_request(request: ChatRequest) -> ModelRoute:
    requested_model = request.model
    text = flatten_messages(request.messages)

    direct_models = {
        LOCAL_FAST_MODEL,
        LOCAL_CODE_MODEL,
        LOCAL_REASON_MODEL,
    }

    if requested_model in direct_models:
        return ModelRoute(
            requested_model=requested_model,
            target_model=requested_model,
            reason="direct local model requested",
        )

    aliases = {
        "onyx-fast": (LOCAL_FAST_MODEL, "fast alias"),
        "onyx-code": (LOCAL_CODE_MODEL, "code alias"),
        "onyx-reason": (LOCAL_REASON_MODEL, "reasoning alias"),
    }

    if has_tools(request) and ROUTE_TO_CODE_WHEN_TOOLS and requested_model in {"onyx-auto", "onyx-fast", "onyx-reason"}:
        return ModelRoute(
            requested_model=requested_model,
            target_model=LOCAL_CODE_MODEL,
            reason="tools present: code/tool model route",
        )

    if requested_model in aliases:
        target_model, reason = aliases[requested_model]
        return ModelRoute(
            requested_model=requested_model,
            target_model=target_model,
            reason=reason,
        )

    if requested_model != "onyx-auto":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model alias: {requested_model}",
        )

    if any(keyword in text for keyword in CODE_KEYWORDS):
        return ModelRoute(
            requested_model=requested_model,
            target_model=LOCAL_CODE_MODEL,
            reason="auto route: code/devsecops task",
        )

    if any(keyword in text for keyword in REASONING_KEYWORDS):
        return ModelRoute(
            requested_model=requested_model,
            target_model=LOCAL_REASON_MODEL,
            reason="auto route: reasoning task",
        )

    return ModelRoute(
        requested_model=requested_model,
        target_model=LOCAL_FAST_MODEL,
        reason="auto route: default fast model",
    )

# =============================================================================
# Ollama request conversion
# =============================================================================


def parse_json_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments


def ollama_tool_calls_from_openai(tool_calls: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tool_calls, list):
        return None

    converted: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue

        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if not name:
            continue

        converted_call: dict[str, Any] = {
            "function": {
                "name": name,
                "arguments": parse_json_arguments(function.get("arguments") or {}),
            }
        }

        if call.get("id"):
            converted_call["id"] = call["id"]

        converted.append(converted_call)

    return converted or None


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user"))

        # OpenAI has developer messages; Ollama does not.
        if role == "developer":
            role = "system"

        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"

        content = extract_text(message.get("content", ""))

        # Tool result messages can be empty in some client flows, but usually are not.
        if not content and role not in {"assistant", "tool"}:
            continue

        normalized_message: dict[str, Any] = {
            "role": role,
            "content": content,
        }

        if role == "assistant":
            converted_tool_calls = ollama_tool_calls_from_openai(message.get("tool_calls"))
            if converted_tool_calls:
                normalized_message["tool_calls"] = converted_tool_calls

        if role == "tool" and message.get("tool_call_id"):
            # Ollama currently does not require this, but preserving it is harmless
            # for clients/models that support it.
            normalized_message["tool_call_id"] = message["tool_call_id"]

        normalized.append(normalized_message)

    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="No valid messages were provided",
        )

    return normalized


def max_tokens_for(route: ModelRoute, request: ChatRequest) -> int:
    if request.max_completion_tokens and request.max_completion_tokens > 0:
        return request.max_completion_tokens

    if request.max_tokens and request.max_tokens > 0:
        return request.max_tokens

    if route.target_model == LOCAL_CODE_MODEL:
        return CODE_MAX_TOKENS

    if route.target_model == LOCAL_REASON_MODEL:
        return REASON_MAX_TOKENS

    return FAST_MAX_TOKENS


def ollama_options(route: ModelRoute, request: ChatRequest) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_ctx": OLLAMA_NUM_CTX,
        "num_predict": max_tokens_for(route, request),
        "temperature": 0.1 if request.temperature is None else request.temperature,
    }

    if request.top_p is not None:
        options["top_p"] = request.top_p

    if request.top_k is not None:
        options["top_k"] = request.top_k

    if request.seed is not None:
        options["seed"] = request.seed

    if isinstance(request.stop, str):
        options["stop"] = [request.stop]
    elif isinstance(request.stop, list):
        options["stop"] = request.stop

    return options


def ollama_format(request: ChatRequest) -> Any | None:
    response_format = request.response_format
    if not isinstance(response_format, dict):
        return None

    if response_format.get("type") == "json_object":
        return "json"

    if response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            schema = json_schema.get("schema")
            if isinstance(schema, dict):
                return schema

    return None


def build_ollama_payload(
    route: ModelRoute,
    request: ChatRequest,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": route.target_model,
        "messages": normalize_messages(request.messages),
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": ollama_options(route, request),
    }

    fmt = ollama_format(request)
    if fmt is not None:
        payload["format"] = fmt

    # Ollama accepts OpenAI-style tools directly in /api/chat.
    # We intentionally do not forward OpenAI's tool_choice field here because
    # the local Ollama API path works correctly with tools alone, and accepting
    # tool_choice on the gateway is enough for OpenAI-compatible clients.
    if has_tools(request):
        payload["tools"] = request.tools

    return payload

# =============================================================================
# OpenAI-compatible response conversion
# =============================================================================


def chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def finish_reason(done_reason: str | None) -> str:
    if not done_reason:
        return "stop"

    if done_reason in {"stop", "length", "tool_calls", "content_filter"}:
        return done_reason

    if done_reason == "unload":
        return "stop"

    return done_reason


def usage_from_ollama(data: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def openai_tool_calls_from_ollama(message: dict[str, Any]) -> list[dict[str, Any]]:
    native_calls = message.get("tool_calls") or []
    converted: list[dict[str, Any]] = []

    if not isinstance(native_calls, list):
        return converted

    for index, call in enumerate(native_calls):
        if not isinstance(call, dict):
            continue

        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        arguments = function.get("arguments") or {}

        if not name:
            continue

        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

        converted.append(
            {
                "id": call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_json,
                },
                # Helpful for streaming deltas. Harmless in non-streaming, but
                # OpenAI's final non-stream tool_call object typically omits it.
                "index": index,
            }
        )

    return converted


def strip_tool_call_index(tool_call: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(tool_call)
    cleaned.pop("index", None)
    return cleaned


def openai_response(data: dict[str, Any], route: ModelRoute) -> dict[str, Any]:
    ollama_message = data.get("message") or {}
    tool_calls = openai_tool_calls_from_ollama(ollama_message)

    if tool_calls:
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [strip_tool_call_index(call) for call in tool_calls],
        }
        choice_finish_reason = "tool_calls"
    else:
        assistant_message = {
            "role": ollama_message.get("role", "assistant"),
            "content": ollama_message.get("content") or "",
        }
        choice_finish_reason = finish_reason(data.get("done_reason"))

    return {
        "id": chat_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": route.requested_model,
        "choices": [
            {
                "index": 0,
                "message": assistant_message,
                "finish_reason": choice_finish_reason,
            }
        ],
        "usage": usage_from_ollama(data),
        "gateway": {
            "backend": "ollama-native-api-chat",
            "target_model": route.target_model,
            "route_reason": route.reason,
        },
    }


def sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


def stream_chunk(
    response_id: str,
    model: str,
    delta: dict[str, Any],
    finish: str | None = None,
) -> str:
    return sse(
        {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }
            ],
        }
    )


def stream_error(response_id: str, model: str, message: str) -> str:
    return stream_chunk(
        response_id=response_id,
        model=model,
        delta={"content": f"\n[Gateway error: {message}]"},
        finish="stop",
    )


async def stream_openai_response_from_non_stream(data: dict[str, Any], route: ModelRoute) -> AsyncIterator[str]:
    response_id = chat_id()
    yield stream_chunk(
        response_id=response_id,
        model=route.requested_model,
        delta={"role": "assistant"},
    )

    message = data.get("message") or {}
    tool_calls = openai_tool_calls_from_ollama(message)

    if tool_calls:
        yield stream_chunk(
            response_id=response_id,
            model=route.requested_model,
            delta={"tool_calls": tool_calls},
        )
        yield stream_chunk(
            response_id=response_id,
            model=route.requested_model,
            delta={},
            finish="tool_calls",
        )
        yield sse_done()
        return

    content = message.get("content") or ""
    if content:
        yield stream_chunk(
            response_id=response_id,
            model=route.requested_model,
            delta={"content": content},
        )

    yield stream_chunk(
        response_id=response_id,
        model=route.requested_model,
        delta={},
        finish=finish_reason(data.get("done_reason")),
    )
    yield sse_done()

# =============================================================================
# Ollama calls
# =============================================================================


async def call_ollama(
    client: httpx.AsyncClient,
    route: ModelRoute,
    request: ChatRequest,
) -> dict[str, Any]:
    payload = build_ollama_payload(route, request, stream=False)
    response = await client.post("/api/chat", json=payload)
    response.raise_for_status()
    return response.json()


async def stream_ollama(
    client: httpx.AsyncClient,
    route: ModelRoute,
    request: ChatRequest,
) -> AsyncIterator[dict[str, Any]]:
    payload = build_ollama_payload(route, request, stream=True)
    async with client.stream("POST", "/api/chat", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            yield json.loads(line)

# =============================================================================
# API routes
# =============================================================================


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "openai-compatible-external-native-ollama-internal",
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_native_chat_url": f"{OLLAMA_BASE_URL}/api/chat",
        "context": {
            "num_ctx": OLLAMA_NUM_CTX,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "timeout_seconds": OLLAMA_HTTP_TIMEOUT,
        },
        "models": {
            "onyx-auto": "keyword-routed local model",
            "onyx-fast": LOCAL_FAST_MODEL,
            "onyx-code": LOCAL_CODE_MODEL,
            "onyx-reason": LOCAL_REASON_MODEL,
        },
        "routing": {
            "route_to_code_when_tools": ROUTE_TO_CODE_WHEN_TOOLS,
            "stream_tools_as_converted_sse": STREAM_TOOLS_AS_CONVERTED_SSE,
        },
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    try:
        response = await app.state.http.get("/api/tags")
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is not ready: {exc}",
        ) from exc

    return {
        "status": "ready",
        "ollama": response.json(),
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    created = int(time.time())
    models = [
        ("onyx-auto", "local-ai-gateway"),
        ("onyx-fast", "local-ai-gateway"),
        ("onyx-code", "local-ai-gateway"),
        ("onyx-reason", "local-ai-gateway"),
        (LOCAL_FAST_MODEL, "ollama"),
        (LOCAL_CODE_MODEL, "ollama"),
        (LOCAL_REASON_MODEL, "ollama"),
    ]

    seen: set[str] = set()
    data: list[dict[str, Any]] = []
    for model, owner in models:
        if model in seen:
            continue
        seen.add(model)
        data.append(
            {
                "id": model,
                "object": "model",
                "created": created,
                "owned_by": owner,
            }
        )

    return {
        "object": "list",
        "data": data,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> Any:
    route = route_request(request)
    tool_names = [
        tool.get("function", {}).get("name")
        for tool in (request.tools or [])
        if isinstance(tool, dict)
    ]

    logger.info(
        "chat request routed: requested=%s target=%s reason=%s stream=%s tools=%s tool_choice=%s tool_names=%s",
        route.requested_model,
        route.target_model,
        route.reason,
        request.stream,
        bool(request.tools),
        request.tool_choice,
        tool_names,
    )

    if request.stream:
        return StreamingResponse(
            openai_stream(request, route),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        data = await call_ollama(app.state.http, route, request)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama HTTP {exc.response.status_code}: {exc.response.text[:1000]}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama backend error: {exc}",
        ) from exc

    return openai_response(data, route)


async def openai_stream(request: ChatRequest, route: ModelRoute) -> AsyncIterator[str]:
    response_id = chat_id()

    if has_tools(request) and STREAM_TOOLS_AS_CONVERTED_SSE:
        try:
            logger.info("stream request contains tools; using non-stream Ollama call and converted SSE response")
            data = await call_ollama(app.state.http, route, request)
            async for chunk in stream_openai_response_from_non_stream(data, route):
                yield chunk
            return
        except httpx.HTTPStatusError as exc:
            yield stream_error(
                response_id=response_id,
                model=route.requested_model,
                message=f"Ollama HTTP {exc.response.status_code}: {exc.response.text[:1000]}",
            )
            yield sse_done()
            return
        except Exception as exc:
            logger.exception("stream tool-call conversion failed")
            yield stream_error(
                response_id=response_id,
                model=route.requested_model,
                message=str(exc),
            )
            yield sse_done()
            return

    role_sent = False

    try:
        async for data in stream_ollama(app.state.http, route, request):
            if data.get("error"):
                yield stream_error(response_id, route.requested_model, str(data["error"]))
                yield sse_done()
                return

            message = data.get("message") or {}
            content = message.get("content") or ""

            if not role_sent:
                delta: dict[str, Any] = {"role": "assistant"}
                if content:
                    delta["content"] = content
                yield stream_chunk(response_id, route.requested_model, delta)
                role_sent = True
            elif content:
                yield stream_chunk(
                    response_id=response_id,
                    model=route.requested_model,
                    delta={"content": content},
                )

            tool_calls = openai_tool_calls_from_ollama(message)
            if tool_calls:
                yield stream_chunk(
                    response_id=response_id,
                    model=route.requested_model,
                    delta={"tool_calls": tool_calls},
                )

            if data.get("done"):
                finish = "tool_calls" if tool_calls else finish_reason(data.get("done_reason"))
                yield stream_chunk(
                    response_id=response_id,
                    model=route.requested_model,
                    delta={},
                    finish=finish,
                )
                yield sse_done()
                return

        yield stream_chunk(
            response_id=response_id,
            model=route.requested_model,
            delta={},
            finish="stop",
        )
        yield sse_done()

    except httpx.HTTPStatusError as exc:
        yield stream_error(
            response_id=response_id,
            model=route.requested_model,
            message=f"Ollama HTTP {exc.response.status_code}: {exc.response.text[:1000]}",
        )
        yield sse_done()
    except Exception as exc:
        logger.exception("stream failed")
        yield stream_error(
            response_id=response_id,
            model=route.requested_model,
            message=str(exc),
        )
        yield sse_done()


@app.post("/debug/route")
async def debug_route(request: ChatRequest) -> dict[str, Any]:
    route = route_request(request)
    return {
        "route": route.model_dump(),
        "ollama_url": f"{OLLAMA_BASE_URL}/api/chat",
        "tools_present": bool(request.tools),
        "tool_choice": request.tool_choice,
        "ollama_payload": build_ollama_payload(
            route=route,
            request=request,
            stream=bool(request.stream),
        ),
    }

