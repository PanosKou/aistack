import os
import json
import subprocess
import urllib.parse
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Hermes Bridge")

HERMES_BIN = os.getenv("HERMES_BIN", "/home/pank/.local/bin/hermes")
HERMES_TIMEOUT = int(os.getenv("HERMES_TIMEOUT", "300"))

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8080/search")
SCRAPE_URL = os.getenv("SCRAPE_URL", "http://192.168.1.81:5678/webhook/scrape-url")


class HermesRequest(BaseModel):
    task: str = Field(..., min_length=3)
    mode: str = Field(default="research")
    max_results: int = Field(default=3, ge=1, le=5)
    max_chars_per_page: int = Field(default=6000, ge=1000, le=15000)


@app.get("/health")
def health():
    return {"status": "ok", "service": "hermes-bridge"}

def escape_rich_markup(text: str) -> str:
    return text.replace("[","\\[").replace("]", "\\]")

def run_hermes(prompt: str) -> str:
    safe_prompt = escape_rich_markup(prompt)
    cmd = [
        HERMES_BIN,
        "chat",
        "-q",
        safe_prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=HERMES_TIMEOUT,
            env={
                **os.environ,
                "HOME": "/home/pank",
                "PATH": "/home/pank/.local/bin:/usr/local/bin:/usr/bin:/bin",
            },
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="hermes_timeout")

    if result.returncode != 0:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "hermes_failed",
                "stderr": result.stderr[-4000:],
                "stdout": result.stdout[-4000:],
            },
        )

    return result.stdout.strip()


def search_searxng(query: str, max_results: int):
    response = requests.post(
        SEARXNG_URL,
        data={
            "q": query,
            "format": "json",
            "language": "en",
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    results = []

    for item in data.get("results", [])[:max_results]:
        url = item.get("url")
        if not url:
            continue

        results.append(
            {
                "title": item.get("title"),
                "url": url,
                "content": item.get("content"),
                "engine": item.get("engine"),
                "score": item.get("score"),
            }
        )

    return results


def scrape_url(url: str, max_chars: int):
    try:
        response = requests.post(
            SCRAPE_URL,
            json={"url": url},
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {
            "url": url,
            "error": str(exc),
            "text": "",
        }

    text = data.get("text") or ""
    return {
        "requestedUrl": data.get("requestedUrl") or url,
        "title": data.get("title"),
        "h1": data.get("h1"),
        "text": text[:max_chars],
        "textLength": len(text),
    }


@app.post("/research")
def research(req: HermesRequest):
    if req.mode == "raw":
        answer = run_hermes(req.task)
        return {
            "task": req.task,
            "mode": req.mode,
            "answer": answer,
        }

    search_results = search_searxng(req.task, req.max_results)

    scraped_pages = []
    for result in search_results:
        scraped = scrape_url(result["url"], req.max_chars_per_page)
        scraped_pages.append(
            {
                "searchResult": result,
                "scraped": scraped,
            }
        )

    context_blocks = []

    for index, item in enumerate(scraped_pages, start=1):
        result = item["searchResult"]
        scraped = item["scraped"]

        context_blocks.append(
            f"""
SOURCE {index}
Title: {result.get("title")}
URL: {result.get("url")}
Search snippet: {result.get("content")}

Extracted page title: {scraped.get("title")}
Extracted text:
{scraped.get("text")}
""".strip()
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are answering using local web search and scraped page content.

User task:
{req.task}

Use only the context below. If the context is weak, incomplete, blocked, or irrelevant, say so clearly.

Context:
{context}

Return:
1. Direct answer
2. Key findings
3. Sources used with URLs
4. Caveats
""".strip()

    answer = run_hermes(prompt)

    return {
        "task": req.task,
        "mode": "research",
        "search_results": search_results,
        "scraped_pages": scraped_pages,
        "answer": answer,
    }
