from __future__ import annotations

from urllib.parse import quote

import httpx
from lxml import html


def duckduckgo_tools(*, max_results: int = 10) -> tuple[object, ...]:
    """Return reusable DuckDuckGo tools for agent definitions."""
    from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

    return (duckduckgo_search_tool(max_results=max_results),)


async def wikipedia_search(
    query: str,
    limit: int = 5,
    language: str = "en",
) -> list[dict[str, str | int]]:
    """Search Wikipedia for pages that match a query.

    Args:
        query: Search terms to look up on Wikipedia.
        limit: Maximum number of matches to return.
        language: Wikipedia language edition to search, for example ``en``.

    Returns:
        A list of page matches with title, page id, snippet, and canonical URL.
    """
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    params: dict[str, str | int] = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(api_url, params=params)
        response.raise_for_status()
        payload = response.json()

    results: list[dict[str, str | int]] = []
    for item in payload.get("query", {}).get("search", []):
        title = str(item["title"])
        encoded_title = quote(title.replace(" ", "_"), safe=":/()")
        snippet_html = f"<div>{item.get('snippet', '')}</div>"
        snippet = html.fromstring(snippet_html).text_content().strip()
        results.append(
            {
                "title": title,
                "page_id": int(item["pageid"]),
                "snippet": snippet,
                "url": f"https://{language}.wikipedia.org/wiki/{encoded_title}",
            }
        )
    return results


async def wikipedia_page(
    title: str,
    language: str = "en",
) -> dict[str, str | None]:
    """Fetch the summary payload for a Wikipedia page by title.

    Args:
        title: Exact or near-exact Wikipedia page title.
        language: Wikipedia language edition, for example ``en``.

    Returns:
        The page title, description, summary extract, and canonical URL when available.
    """
    encoded_title = quote(title.replace(" ", "_"), safe=":/()")
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()

    content_urls = payload.get("content_urls", {})
    desktop = content_urls.get("desktop", {})

    return {
        "title": payload.get("title"),
        "description": payload.get("description"),
        "summary": payload.get("extract"),
        "url": desktop.get("page") or payload.get("content_urls"),
    }


async def fetch_page_text(url: str, max_chars: int = 8_000) -> dict[str, str]:
    """Fetch a URL and extract a compact text representation.

    Args:
        url: The page URL to fetch.
        max_chars: Maximum number of text characters to return.

    Returns:
        The source URL, detected page title, and extracted body text.
    """
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "SOARcuit/agent-registry"})
        response.raise_for_status()

    document = html.fromstring(response.text)
    title = " ".join(document.xpath("//title/text()")).strip() or url
    body_text = " ".join(document.text_content().split())

    return {
        "url": str(response.url),
        "title": title,
        "text": body_text[:max_chars],
    }


def wikipedia_tools() -> tuple[object, ...]:
    """Return reusable Wikipedia lookup tools for agent definitions."""
    return (wikipedia_search, wikipedia_page)


def research_tools() -> tuple[object, ...]:
    """Return a compact generic research tool bundle."""
    return duckduckgo_tools() + wikipedia_tools() + (fetch_page_text,)
