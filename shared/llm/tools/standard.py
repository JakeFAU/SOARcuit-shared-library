from __future__ import annotations

from collections.abc import Mapping

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

        best_title = str(search_data["query"]["search"][0]["title"])

        # 2. Fetch the summary for that title
        summary_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_title.replace(' ', '_')}"
        )
        summary_resp = await client.get(summary_url)
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()

        return WikipediaOutput(
            title=str(summary_data.get("title", best_title)),
            summary=str(summary_data.get("extract", "No summary available.")),
            url=str(summary_data.get("content_urls", {}).get("desktop", {}).get("page", "")),
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

        answer = str(data.get("AbstractText") or data.get("Answer") or "")
        if not answer and data.get("RelatedTopics"):
            first_topic = data["RelatedTopics"][0]
            if isinstance(first_topic, dict) and "Text" in first_topic:
                answer = str(first_topic["Text"])

        if not answer:
            answer = f"No instant answer found for '{input_data.query}'."

        related = [
            str(topic["Text"])
            for topic in data.get("RelatedTopics", [])
            if isinstance(topic, dict) and "Text" in topic
        ]

        return DuckDuckGoOutput(
            answer=answer,
            related_topics=related[:5],
        )


class TavilySearchInput(BaseModel):
    """Input for Tavily high-quality search."""

    query: str = Field(..., description="The search query for Tavily.")
    search_depth: str = Field(default="basic", description="Search depth: 'basic' or 'advanced'.")


class TavilySearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float


class TavilySearchOutput(BaseModel):
    """Structured output for Tavily results."""

    results: list[TavilySearchResult]
    answer: str | None = None


async def tavily_search_fn(input_data: TavilySearchInput) -> TavilySearchOutput:
    """Real Tavily search using their API."""
    from shared.config.config import get_settings

    settings = get_settings()
    if not settings.external_tools.tavily_api_key:
        raise ValueError("Tavily API key is not configured.")

    async with httpx.AsyncClient(timeout=15.0) as client:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": settings.external_tools.tavily_api_key.get_secret_value(),
            "query": input_data.query,
            "search_depth": input_data.search_depth,
            "include_answer": True,
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        results = [TavilySearchResult(**r) for r in data.get("results", [])]
        return TavilySearchOutput(results=results, answer=data.get("answer"))


class ArxivInput(BaseModel):
    """Input for arXiv paper search."""

    query: str = Field(..., description="The search query for arXiv (e.g., 'quantum computing').")
    max_results: int | None = Field(None, description="Optional override for max results.")


class ArxivPaper(BaseModel):
    title: str
    summary: str
    authors: list[str]
    published: str
    link: str


async def arxiv_search_fn(input_data: ArxivInput) -> ArxivOutput:
    """Real arXiv search using their API (Atom feed)."""
    import xml.etree.ElementTree as ET

    from shared.config.config import get_settings

    settings = get_settings()
    limit = input_data.max_results or settings.external_tools.arxiv_max_results

    async with httpx.AsyncClient(timeout=15.0) as client:
        url = "http://export.arxiv.org/api/query"
        params_arxiv: Mapping[str, str | int | float | bool | None] = {
            "search_query": f"all:{input_data.query}",
            "start": 0,
            "max_results": limit,
        }
        resp = await client.get(url, params=params_arxiv)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}

        papers = []
        for entry in root.findall("atom:entry", namespace):
            title_el = entry.find("atom:title", namespace)
            summary_el = entry.find("atom:summary", namespace)
            published_el = entry.find("atom:published", namespace)

            title = (
                (title_el.text or "No Title").strip().replace("\n", " ")
                if title_el is not None
                else "No Title"
            )
            summary = (
                (summary_el.text or "No Summary").strip().replace("\n", " ")
                if summary_el is not None
                else "No Summary"
            )

            authors = []
            for author_el in entry.findall("atom:author", namespace):
                name_el = author_el.find("atom:name", namespace)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text)

            published = (
                published_el.text if published_el is not None and published_el.text else "Unknown"
            )

            link_pdf = entry.find("atom:link[@title='pdf']", namespace)
            if link_pdf is None:
                link_pdf = entry.find("atom:link[@rel='alternate']", namespace)

            pdf_url = link_pdf.attrib["href"] if link_pdf is not None else ""

            papers.append(
                ArxivPaper(
                    title=title,
                    summary=summary,
                    authors=authors,
                    published=published,
                    link=pdf_url,
                )
            )

        return ArxivOutput(papers=papers)


class ArxivOutput(BaseModel):
    """Structured output for arXiv results."""

    papers: list[ArxivPaper]


class GitHubRepoInput(BaseModel):
    """Input for fetching GitHub repository info."""

    owner: str = Field(..., description="The owner of the repository.")
    repo: str = Field(..., description="The name of the repository.")


class GitHubRepoOutput(BaseModel):
    """Structured output for GitHub repository info."""

    name: str
    full_name: str
    description: str | None
    stars: int
    forks: int
    open_issues: int
    language: str | None
    url: str


async def github_repo_fn(input_data: GitHubRepoInput) -> GitHubRepoOutput:
    """Real GitHub API call to fetch repo info."""
    from shared.config.config import get_settings

    settings = get_settings()

    headers = {"Accept": "application/vnd.github+json"}
    if settings.external_tools.github_token:
        headers["Authorization"] = (
            f"Bearer {settings.external_tools.github_token.get_secret_value()}"
        )

    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        url = f"https://api.github.com/repos/{input_data.owner}/{input_data.repo}"
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        return GitHubRepoOutput(
            name=str(data["name"]),
            full_name=str(data["full_name"]),
            description=data.get("description"),
            stars=int(data["stargazers_count"]),
            forks=int(data["forks_count"]),
            open_issues=int(data["open_issues_count"]),
            language=data.get("language"),
            url=str(data["html_url"]),
        )


# Tool definitions
WIKIPEDIA_TOOL = BaseTool(
    name="wikipedia_search",
    description="Search Wikipedia for concise, factual summaries of topics.",
    function=wikipedia_search_fn,
    input_model=WikipediaInput,
    output_model=WikipediaOutput,
    cost_estimate=lambda _: 0.0,
    risk_estimate=lambda _: 0.0,
)

DUCKDUCKGO_TOOL = BaseTool(
    name="duckduckgo_search",
    description="Get instant answers and snippets from DuckDuckGo for general queries.",
    function=duckduckgo_instant_answer_fn,
    input_model=DuckDuckGoInput,
    output_model=DuckDuckGoOutput,
    cost_estimate=lambda _: 0.0,
    risk_estimate=lambda _: 0.0,
)

TAVILY_TOOL = BaseTool(
    name="tavily_search",
    description="Perform high-quality web searches with optional AI-summarized answers.",
    function=tavily_search_fn,
    input_model=TavilySearchInput,
    output_model=TavilySearchOutput,
    cost_estimate=lambda _: 0.0,
    risk_estimate=lambda _: 0.0,
)

ARXIV_TOOL = BaseTool(
    name="arxiv_search",
    description="Search for scientific papers and preprints on arXiv.",
    function=arxiv_search_fn,
    input_model=ArxivInput,
    output_model=ArxivOutput,
    cost_estimate=lambda _: 0.0,
    risk_estimate=lambda _: 0.0,
)

GITHUB_REPO_TOOL = BaseTool(
    name="github_repo_info",
    description="Fetch metadata about a public GitHub repository.",
    function=github_repo_fn,
    input_model=GitHubRepoInput,
    output_model=GitHubRepoOutput,
    cost_estimate=lambda _: 0.0,
    risk_estimate=lambda _: 0.0,
)
