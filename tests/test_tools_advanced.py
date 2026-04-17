import pytest
import math
import httpx
from pydantic import BaseModel, Field
from unittest.mock import AsyncMock, MagicMock
from shared.llm.tools.base import (
    BaseTool, ToolExecutionResult, ToolInputError, ToolOutputError,
    ToolConfigurationError, ToolExecutionError
)
from shared.llm.tools.standard import (
    WikipediaInput, WikipediaOutput, wikipedia_search_fn,
    DuckDuckGoInput, DuckDuckGoOutput, duckduckgo_instant_answer_fn,
    TavilySearchInput, TavilySearchOutput, tavily_search_fn,
    ArxivInput, ArxivOutput, arxiv_search_fn,
    GitHubRepoInput, GitHubRepoOutput, github_repo_fn,
    WIKIPEDIA_TOOL, DUCKDUCKGO_TOOL, TAVILY_TOOL, ARXIV_TOOL, GITHUB_REPO_TOOL
)

class MockInput(BaseModel):
    query: str

class MockOutput(BaseModel):
    result: str

async def mock_fn(input_data: MockInput) -> MockOutput:
    return MockOutput(result=f"processed {input_data.query}")

@pytest.mark.anyio
async def test_wikipedia_search_mocked(respx_mock):
    respx_mock.get("https://en.wikipedia.org/w/api.php").mock(return_value=httpx.Response(
        200, json={"query": {"search": [{"title": "Python (programming language)"}]}}
    ))
    respx_mock.get("https://en.wikipedia.org/api/rest_v1/page/summary/Python_(programming_language)").mock(return_value=httpx.Response(
        200, json={
            "title": "Python",
            "extract": "Python is a language.",
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Python"}}
        }
    ))
    
    result = await wikipedia_search_fn(WikipediaInput(query="Python"))
    assert result.title == "Python"
    assert "language" in result.summary

@pytest.mark.anyio
async def test_duckduckgo_search_mocked(respx_mock):
    respx_mock.get("https://api.duckduckgo.com/").mock(return_value=httpx.Response(
        200, json={
            "AbstractText": "Python is a high-level programming language.",
            "RelatedTopics": []
        }
    ))
    
    result = await duckduckgo_instant_answer_fn(DuckDuckGoInput(query="Python"))
    assert "high-level" in result.answer

@pytest.mark.anyio
async def test_tavily_search_mocked(respx_mock, monkeypatch):
    from shared.config.config import AppSettings, LLMSettings, GCPSettings, ModelNames
    from pydantic import SecretStr
    
    # Mock settings
    mock_settings = AppSettings.model_construct(
        external_tools=MagicMock(tavily_api_key=SecretStr("tav-test"))
    )
    monkeypatch.setattr("shared.config.config.get_settings", lambda: mock_settings)

    respx_mock.post("https://api.tavily.com/search").mock(return_value=httpx.Response(
        200, json={
            "results": [{"title": "T", "url": "U", "content": "C", "score": 0.9}],
            "answer": "AI Answer"
        }
    ))
    
    result = await tavily_search_fn(TavilySearchInput(query="test"))
    assert result.answer == "AI Answer"
    assert result.results[0].title == "T"

@pytest.mark.anyio
async def test_arxiv_search_mocked(respx_mock, monkeypatch):
    from shared.config.config import AppSettings
    mock_settings = AppSettings.model_construct(
        external_tools=MagicMock(arxiv_max_results=5)
    )
    monkeypatch.setattr("shared.config.config.get_settings", lambda: mock_settings)

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Quantum Computing</title>
        <summary>A summary of quantum computing.</summary>
        <author><name>John Doe</name></author>
        <published>2023-01-01T00:00:00Z</published>        <link title="pdf" href="http://arxiv.org/pdf/1234.5678" rel="related" type="application/pdf"/>
      </entry>
    </feed>"""
    respx_mock.get("http://export.arxiv.org/api/query").mock(return_value=httpx.Response(200, content=xml_content))
    
    result = await arxiv_search_fn(ArxivInput(query="quantum"))
    assert len(result.papers) == 1
    assert result.papers[0].title == "Quantum Computing"

@pytest.mark.anyio
async def test_github_repo_mocked(respx_mock, monkeypatch):
    from shared.config.config import AppSettings
    from pydantic import SecretStr
    
    mock_settings = AppSettings.model_construct(
        external_tools=MagicMock(github_token=SecretStr("git-test"))
    )
    monkeypatch.setattr("shared.config.config.get_settings", lambda: mock_settings)

    respx_mock.get("https://api.github.com/repos/owner/repo").mock(return_value=httpx.Response(
        200, json={
            "name": "repo",
            "full_name": "owner/repo",
            "description": "desc",
            "stargazers_count": 100,
            "forks_count": 50,
            "open_issues_count": 5,
            "language": "Python",
            "html_url": "url"
        }
    ))
    
    result = await github_repo_fn(GitHubRepoInput(owner="owner", repo="repo"))
    assert result.name == "repo"
    assert result.stars == 100
