from __future__ import annotations

import math

import httpx
from pydantic import BaseModel, Field

from .base import BaseTool


class WikipediaInput(BaseModel):
    """Input for Wikipedia summary search."""

    query: str = Field(..., description="The topic to search for on Wikipedia.")


class WikipediaOutput(BaseModel):
    """Structured output for Wikipedia summary."""

    title: str = Field(..., description="The official title of the Wikipedia page.")
    summary: str = Field(..., description="A concise summary of the topic.")
    url: str = Field(..., description="The full URL to the Wikipedia article.")


async def wikipedia_search_fn(input_data: WikipediaInput) -> WikipediaOutput:
    """
    Real Wikipedia search using the REST API.
    First searches for the most relevant page title, then fetches the summary.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Search for the best matching title
        search_url = "https://en.wikipedia.org/w/api.php"
        params_wiki: dict[str, str | int] = {
            "action": "query",
            "list": "search",
            "srsearch": input_data.query,
            "format": "json",
            "srlimit": 1,
        }
        search_resp = await client.get(search_url, params=params_wiki)
        search_resp.raise_for_status()
        search_data = search_resp.json()

        if not search_data.get("query", {}).get("search"):
            raise ValueError(f"No Wikipedia results found for '{input_data.query}'.")

        best_title = search_data["query"]["search"][0]["title"]

        # 2. Fetch the summary for that title
        summary_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title.replace(' ', '_')}"
        )
        summary_resp = await client.get(summary_url)
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()

        return WikipediaOutput(
            title=summary_data.get("title", best_title),
            summary=summary_data.get("extract", "No summary available."),
            url=summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        )


class DuckDuckGoInput(BaseModel):
    """Input for DuckDuckGo instant answer search."""

    query: str = Field(..., description="The search query for DuckDuckGo.")


class DuckDuckGoOutput(BaseModel):
    """Structured output for DuckDuckGo results."""

    answer: str = Field(..., description="The primary answer or snippet found.")
    related_topics: list[str] = Field(default_factory=list, description="Related topic headings.")


async def duckduckgo_instant_answer_fn(input_data: DuckDuckGoInput) -> DuckDuckGoOutput:
    """
    Real DuckDuckGo search using the Instant Answer API.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = "https://api.duckduckgo.com/"
        params_ddg: dict[str, str | int] = {
            "q": input_data.query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
        resp = await client.get(url, params=params_ddg)
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("AbstractText") or data.get("Answer")
        if not answer and data.get("RelatedTopics"):
            # Fallback to first related topic snippet if available
            first_topic = data["RelatedTopics"][0]
            if "Text" in first_topic:
                answer = first_topic["Text"]

        if not answer:
            answer = f"No instant answer found for '{input_data.query}'."

        related = [
            topic["Text"]
            for topic in data.get("RelatedTopics", [])
            if isinstance(topic, dict) and "Text" in topic
        ]

        return DuckDuckGoOutput(
            answer=answer,
            related_topics=related[:5],  # Limit to top 5
        )


# Tool definitions
WIKIPEDIA_TOOL = BaseTool(
    name="wikipedia_search",
    description="Search Wikipedia for concise, factual summaries of topics.",
    function=wikipedia_search_fn,
    input_model=WikipediaInput,
    output_model=WikipediaOutput,
    cost_estimate=lambda _: math.nan,
    risk_estimate=lambda _: math.nan,
)

DUCKDUCKGO_TOOL = BaseTool(
    name="duckduckgo_search",
    description="Get instant answers and snippets from DuckDuckGo for general queries.",
    function=duckduckgo_instant_answer_fn,
    input_model=DuckDuckGoInput,
    output_model=DuckDuckGoOutput,
    cost_estimate=lambda _: math.nan,
    risk_estimate=lambda _: math.nan,
)
