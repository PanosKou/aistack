import os
import secrets
from typing import Any

import httpx
import uvicorn
from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import JSONResponse

MCP_API_KEY = os.getenv("MCP_API_KEY", "").strip()
RAG_API_KEY = os.getenv("RAG_API_KEY", "").strip()
RAG_API_URL = os.getenv(
    "RAG_API_URL",
    "http://recitals-rag-api:8000",
).rstrip("/")
RECITALS_MODEL_ID = os.getenv(
    "RECITALS_MODEL_ID",
    "recitals-rag",
)
MCP_PUBLIC_URL = os.getenv(
    "MCP_PUBLIC_URL",
    "http://localhost:8020/mcp",
)
MCP_AUTH_ISSUER_URL = os.getenv(
    "MCP_AUTH_ISSUER_URL",
    "http://localhost:8020/",
)

REQUIRED_SCOPE = "recitals:query"
REQUEST_TIMEOUT = 300.0

if not MCP_API_KEY:
    raise RuntimeError("MCP_API_KEY is required")

if not RAG_API_KEY:
    raise RuntimeError("RAG_API_KEY is required")


class StaticTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        if not secrets.compare_digest(token, MCP_API_KEY):
            return None

        return AccessToken(
            token=token,
            client_id="recitals-mcp-client",
            scopes=[REQUIRED_SCOPE],
            resource=MCP_PUBLIC_URL,
            subject="recitals-mcp-client",
        )


mcp = MCPServer(
    "RECITALS MCP",
    version="0.1.0",
    instructions=(
        "Use ask_recitals to query the indexed RECITALS documents."
    ),
    token_verifier=StaticTokenVerifier(),
    auth=AuthSettings(
        issuer_url=AnyHttpUrl(MCP_AUTH_ISSUER_URL),
        resource_server_url=AnyHttpUrl(MCP_PUBLIC_URL),
        required_scopes=[REQUIRED_SCOPE],
        validate_token_resource=True,
    ),
)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "recitals-mcp",
            "rag_api_url": RAG_API_URL,
            "model": RECITALS_MODEL_ID,
        }
    )


@mcp.tool(
    description=(
        "Ask a question using the RECITALS RAG service and return "
        "a grounded answer with source citations."
    )
)
async def ask_recitals(question: str) -> dict[str, Any]:
    question = question.strip()

    if not question:
        raise ValueError("question must not be empty")

    if len(question) > 4000:
        raise ValueError("question must not exceed 4000 characters")

    payload = {
        "model": RECITALS_MODEL_ID,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ],
    }

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
        ) as client:
            response = await client.post(
                f"{RAG_API_URL}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {RAG_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "The RECITALS RAG service is unavailable"
        ) from exc

    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "The RECITALS RAG service returned an invalid response"
        ) from exc

    return {
        "answer": answer,
        "model": RECITALS_MODEL_ID,
    }


app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    json_response=True,
    stateless_http=True,
)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
