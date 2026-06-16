import asyncio
import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

log = logging.getLogger("hermes-bridge")


SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8080/search")
SCRAPE_URL = os.getenv("SCRAPE_URL", "http://192.168.1.81:5678/webhook/scrape-url")

# This must be LangGraph, not Ollama.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "onyx-fast")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-dummy-key")

DEFAULT_MAX_RESULTS = int(os.getenv("DEFAULT_MAX_RESULTS", "1"))
MAX_RESULTS_HARD = int(os.getenv("MAX_RESULTS_HARD", "2"))

DEFAULT_MAX_CHARS_PER_PAGE = int(os.getenv("DEFAULT_MAX_CHARS_PER_PAGE", "2500"))
MAX_CHARS_PER_PAGE_HARD = int(os.getenv("MAX_CHARS_PER_PAGE_HARD", "5000"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS", "6000"))

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "384"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

SEARCH_TIMEOUT_SECONDS = float(os.getenv("SEARCH_TIMEOUT_SECONDS", "15"))
SCRAPE_TIMEOUT_SECONDS = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "25"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

MAX_CONCURRENT_RESEARCH = int(os.getenv("MAX_CONCURRENT_RESEARCH", "1"))

research_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESEARCH)

app = FastAPI(title="Hermes Bridge", version="0.2.0-langgraph")


class ResearchRequest(BaseModel):
    task: str = Field(..., min_length=3)
    max_results: int | None = None
    max_chars_per_page: int | None = None


def clamp(value: int | None, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        value = default

    try:
        value = int(value)
    except Exception:
        value = default

    return max(minimum, min(value, maximum))


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate(text: str, max_chars: int) -> str:
    text = clean_text(text)

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def is_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def extract_answer(llm_response: dict[str, Any]) -> str:
    choices = llm_response.get("choices") or []

    if not choices:
        return ""

    first = choices[0]

    message = first.get("message") or {}
    content = message.get("content")

    if content:
        return str(content).strip()

    text = first.get("text")

    if text:
        return str(text).strip()

    return ""


async def searxng_search(query: str, max_results: int) -> list[dict[str, str]]:
    timeout = httpx.Timeout(SEARCH_TIMEOUT_SECONDS)
    limits = httpx.Limits(max_connections=3, max_keepalive_connections=1)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        response = await client.post(
            SEARXNG_URL,
            data={
                "q": query,
                "format": "json",
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "SearXNG request failed",
                "status_code": response.status_code,
                "body": response.text[:1000],
            },
        )

    payload = response.json()
    raw_results = payload.get("results") or []

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for item in raw_results:
        url = clean_text(item.get("url"))

        if not url or not is_http_url(url) or url in seen_urls:
            continue

        seen_urls.add(url)

        title = clean_text(item.get("title")) or url
        snippet = clean_text(item.get("content") or item.get("description") or "")

        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
            }
        )

        if len(results) >= max_results:
            break

    return results


async def scrape_url(url: str) -> dict[str, Any]:
    timeout = httpx.Timeout(SCRAPE_TIMEOUT_SECONDS)
    limits = httpx.Limits(max_connections=2, max_keepalive_connections=1)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        response = await client.post(
            SCRAPE_URL,
            json={"url": url},
            headers={"Content-Type": "application/json"},
        )

    if response.status_code >= 400:
        raise RuntimeError(f"n8n scrape failed: status={response.status_code} body={response.text[:500]}")

    payload = response.json()

    if isinstance(payload, list):
        if not payload:
            return {}
        payload = payload[0]

    if not isinstance(payload, dict):
        return {}

    return payload


async def collect_pages(
    search_results: list[dict[str, str]],
    max_chars_per_page: int,
) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []

    for result in search_results:
        url = result["url"]

        try:
            scraped = await scrape_url(url)
            title = clean_text(scraped.get("title")) or result["title"]
            h1 = clean_text(scraped.get("h1"))
            text = clean_text(scraped.get("text"))
            scrape_ok = True
            scrape_error = None
        except Exception as exc:
            log.warning("scrape_failed url=%s error=%s", url, exc)
            title = result["title"]
            h1 = ""
            text = result["snippet"]
            scrape_ok = False
            scrape_error = str(exc)

        pages.append(
            {
                "title": title,
                "h1": h1,
                "url": url,
                "text": truncate(text, max_chars_per_page),
                "scrape_ok": scrape_ok,
                "scrape_error": scrape_error,
            }
        )

    return pages


def build_context(pages: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    remaining = MAX_TOTAL_CONTEXT_CHARS

    for index, page in enumerate(pages, start=1):
        if remaining <= 0:
            break

        title = clean_text(page.get("title"))
        url = clean_text(page.get("url"))
        text = clean_text(page.get("text"))

        header = f"[{index}] {title}\nURL: {url}\nCONTENT:\n"
        available_for_text = max(0, remaining - len(header) - 2)

        if available_for_text <= 0:
            break

        section_text = truncate(text, available_for_text)
        section = header + section_text

        sections.append(section)
        remaining -= len(section)

    return "\n\n---\n\n".join(sections)


async def call_langgraph(task: str, context: str) -> str:
    endpoint = f"{LLM_BASE_URL}/chat/completions"

    system_prompt = (
        "You are the summarisation step of a local web research pipeline. "
        "Use only the supplied source snippets. "
        "Keep the answer short, technical, and precise. "
        "Prefer official documentation when present. "
        "Cite sources inline using [1], [2], etc. "
        "If the sources are insufficient, say that clearly."
    )

    user_prompt = (
        f"Research task:\n{task}\n\n"
        f"Source snippets:\n{context}\n\n"
        "Return a short answer followed by a compact Sources section."
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "stream": False,
    }

    timeout = httpx.Timeout(LLM_TIMEOUT_SECONDS)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        response = await client.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "LangGraph chat completion failed",
                "status_code": response.status_code,
                "endpoint": endpoint,
                "body": response.text[:1000],
            },
        )

    answer = extract_answer(response.json())

    if not answer:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "LangGraph returned an empty answer",
                "endpoint": endpoint,
            },
        )

    return answer


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mode": "langgraph",
        "searxng_url": SEARXNG_URL,
        "scrape_url": SCRAPE_URL,
        "llm_base_url": LLM_BASE_URL,
        "llm_model": LLM_MODEL,
        "max_concurrent_research": MAX_CONCURRENT_RESEARCH,
    }


@app.post("/research")
async def research(request: ResearchRequest) -> dict[str, Any]:
    async with research_semaphore:
        max_results = clamp(
            request.max_results,
            DEFAULT_MAX_RESULTS,
            1,
            MAX_RESULTS_HARD,
        )

        max_chars_per_page = clamp(
            request.max_chars_per_page,
            DEFAULT_MAX_CHARS_PER_PAGE,
            500,
            MAX_CHARS_PER_PAGE_HARD,
        )

        task = clean_text(request.task)

        log.info(
            "research_started task=%r max_results=%s max_chars_per_page=%s llm_base_url=%s model=%s",
            task,
            max_results,
            max_chars_per_page,
            LLM_BASE_URL,
            LLM_MODEL,
        )

        search_results = await searxng_search(task, max_results=max_results)

        if not search_results:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "SearXNG returned no usable HTTP/HTTPS results",
                    "task": task,
                },
            )

        pages = await collect_pages(search_results, max_chars_per_page=max_chars_per_page)
        context = build_context(pages)

        if not context.strip():
            raise HTTPException(
                status_code=502,
                detail={
                    "error": "No usable context after scraping",
                    "task": task,
                },
            )

        answer = await call_langgraph(task=task, context=context)

        log.info(
            "research_finished task=%r results=%s pages=%s",
            task,
            len(search_results),
            len(pages),
        )

        return {
            "mode": "langgraph",
            "model": LLM_MODEL,
            "task": task,
            "answer": answer,
            "search_results": search_results,
            "scraped_pages": [
                {
                    "title": page["title"],
                    "url": page["url"],
                    "chars": len(page["text"]),
                    "scrape_ok": page["scrape_ok"],
                    "scrape_error": page["scrape_error"],
                }
                for page in pages
            ],
            "limits": {
                "max_results": max_results,
                "max_chars_per_page": max_chars_per_page,
                "max_total_context_chars": MAX_TOTAL_CONTEXT_CHARS,
                "llm_max_tokens": LLM_MAX_TOKENS,
                "max_concurrent_research": MAX_CONCURRENT_RESEARCH,
            },
        }
