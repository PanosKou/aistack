import os
import requests
from fastmcp import FastMCP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-hermes-bridge")

mcp = FastMCP(
    "hermes-bridge",
    instructions=(
        "Provides one controlled tool for advanced local web research. "
        "Use this when the user asks for source-backed research, official documentation lookup, "
        "multi-source comparison, or deeper investigation."
    ),
)

HERMES_BRIDGE_URL = os.getenv(
    "HERMES_BRIDGE_URL",
    "http://127.0.0.1:8092/research",
)


@mcp.tool()
def advanced_web_research(task: str, max_results: int = 2) -> dict:
    """
    Perform advanced local web research.

    This calls the local Hermes Bridge, which performs:
    SearXNG search → n8n scrape workflow → Hermes/local LLM summarisation.

    Args:
        task: The research question or instruction.
        max_results: Number of search results to inspect. Recommended range: 1 to 5.

    Returns:
        A JSON object containing the answer, search results, scraped pages, and source URLs.
    """
    if max_results < 1:
        max_results = 1
    if max_results > 5:
        max_results = 5

    logger.info("advanced_web_research CALLED task=%r max_results=%s", task, max_results)

    response = requests.post(
        HERMES_BRIDGE_URL,
        json={
            "task": task,
            "max_results": max_results,
        },
        timeout=420,
    )
    logger.info("hermes-bridge returned status=%s", response.status_code)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8093,
    )
