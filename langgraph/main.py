import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ConfigDict


load_dotenv()

app = FastAPI(title="Onyx Local LangGraph Gateway", version="0.1.0")


# -----------------------------
# Environment
# -----------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

LOCAL_FAST_MODEL = os.getenv("LOCAL_FAST_MODEL", "llama3.1:8b")
LOCAL_CODE_MODEL = os.getenv("LOCAL_CODE_MODEL", "qwen2.5-coder:14b")
LOCAL_REASON_MODEL = os.getenv("LOCAL_REASON_MODEL", "deepseek-r1:14b")


# -----------------------------
# Client
# -----------------------------

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
)


# -----------------------------
# API models
# -----------------------------

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "onyx-auto"
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class GraphState(TypedDict, total=False):
    request: Dict[str, Any]
    route: Dict[str, Any]
    response: Dict[str, Any]


# -----------------------------
# Routing rules
# -----------------------------

CODE_KEYWORDS = [
    "code",
    "python",
    "bash",
    "shell",
    "docker",
    "compose",
    "kubernetes",
    "k8s",
    "terraform",
    "ansible",
    "javascript",
    "typescript",
    "golang",
    "go ",
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
    "sast",
    "dast",
    "semgrep",
    "trivy",
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
]


def flatten_messages(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []

    for message in messages:
        content = message.get("content", "")

        if isinstance(content, str):
            parts.append(content)

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)

    return "\n".join(parts).lower()


def route_request(request: Dict[str, Any]) -> Dict[str, Any]:
    requested_model = request.get("model", "onyx-auto")
    messages = request.get("messages", [])
    text = flatten_messages(messages)

    # Direct Ollama model names.
    if requested_model in {
        LOCAL_FAST_MODEL,
        LOCAL_CODE_MODEL,
        LOCAL_REASON_MODEL,
        "llama3.1:8b",
        "qwen2.5-coder:14b",
        "deepseek-r1:14b",
    }:
        return {
            "target_model": requested_model,
            "reason": "direct local model requested",
        }

    # Gateway aliases.
    if requested_model == "onyx-fast":
        return {
            "target_model": LOCAL_FAST_MODEL,
            "reason": "fast alias",
        }

    if requested_model == "onyx-code":
        return {
            "target_model": LOCAL_CODE_MODEL,
            "reason": "code alias",
        }

    if requested_model == "onyx-reason":
        return {
            "target_model": LOCAL_REASON_MODEL,
            "reason": "reasoning alias",
        }

    if requested_model != "onyx-auto":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model alias: {requested_model}",
        )

    # Automatic routing.
    if any(keyword in text for keyword in CODE_KEYWORDS):
        return {
            "target_model": LOCAL_CODE_MODEL,
            "reason": "auto route: code/devsecops task",
        }

    if any(keyword in text for keyword in REASONING_KEYWORDS):
        return {
            "target_model": LOCAL_REASON_MODEL,
            "reason": "auto route: reasoning task",
        }

    return {
        "target_model": LOCAL_FAST_MODEL,
        "reason": "auto route: default fast model",
    }


def clean_payload_for_ollama(request: Dict[str, Any], target_model: str) -> Dict[str, Any]:
    payload = dict(request)
    payload["model"] = target_model

    # Defensive defaults.
    payload.setdefault("temperature", 0.1)

    # Remove fields that are gateway-only or not wanted.
    payload.pop("cloud_allowed", None)
    payload.pop("provider", None)

    return payload


# -----------------------------
# LangGraph nodes
# -----------------------------

def route_node(state: GraphState) -> GraphState:
    request = state["request"]
    route = route_request(request)

    return {
        "request": request,
        "route": route,
    }


def call_ollama_node(state: GraphState) -> GraphState:
    request = state["request"]
    route = state["route"]

    payload = clean_payload_for_ollama(
        request=request,
        target_model=route["target_model"],
    )

    payload["stream"] = False

    try:
        response = ollama_client.chat.completions.create(**payload)
        response_dict = response.model_dump()

        # Keep the model name stable from the gateway perspective.
        # The actual target remains visible via /health logs if needed.
        return {
            "request": request,
            "route": route,
            "response": response_dict,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama backend error: {exc}",
        ) from exc


workflow = StateGraph(GraphState)
workflow.add_node("route", route_node)
workflow.add_node("ollama", call_ollama_node)

workflow.add_edge(START, "route")
workflow.add_edge("route", "ollama")
workflow.add_edge("ollama", END)

graph = workflow.compile()


# -----------------------------
# HTTP routes
# -----------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "mode": "local-only",
        "ollama_base_url": OLLAMA_BASE_URL,
        "models": {
            "onyx-auto": "keyword-routed local model",
            "onyx-fast": LOCAL_FAST_MODEL,
            "onyx-code": LOCAL_CODE_MODEL,
            "onyx-reason": LOCAL_REASON_MODEL,
        },
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    created = int(time.time())

    models = [
        {
            "id": "onyx-auto",
            "object": "model",
            "created": created,
            "owned_by": "langgraph-gateway",
        },
        {
            "id": "onyx-fast",
            "object": "model",
            "created": created,
            "owned_by": "langgraph-gateway",
        },
        {
            "id": "onyx-code",
            "object": "model",
            "created": created,
            "owned_by": "langgraph-gateway",
        },
        {
            "id": "onyx-reason",
            "object": "model",
            "created": created,
            "owned_by": "langgraph-gateway",
        },
        {
            "id": LOCAL_FAST_MODEL,
            "object": "model",
            "created": created,
            "owned_by": "ollama",
        },
        {
            "id": LOCAL_CODE_MODEL,
            "object": "model",
            "created": created,
            "owned_by": "ollama",
        },
        {
            "id": LOCAL_REASON_MODEL,
            "object": "model",
            "created": created,
            "owned_by": "ollama",
        },
    ]

    return {
        "object": "list",
        "data": models,
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest) -> Any:
    request_dict = request.model_dump(exclude_none=True)

    if request.stream:
        route = route_request(request_dict)
        target_model = route["target_model"]

        payload = clean_payload_for_ollama(
            request=request_dict,
            target_model=target_model,
        )
        payload["stream"] = True

        def event_stream():
            try:
                stream = ollama_client.chat.completions.create(**payload)

                for chunk in stream:
                    yield f"data: {chunk.model_dump_json()}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as exc:
                error_payload = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": target_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"\n[Gateway error: {exc}]"
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }

                yield f"data: {json.dumps(error_payload)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
        )

    result = graph.invoke({"request": request_dict})
    return result["response"]
