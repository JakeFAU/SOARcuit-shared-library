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
    WIKIPEDIA_TOOL, DUCKDUCKGO_TOOL
)

class MockInput(BaseModel):
    query: str

class MockOutput(BaseModel):
    result: str

async def mock_fn(input_data: MockInput) -> MockOutput:
    return MockOutput(result=f"processed {input_data.query}")

def test_base_tool_initialization():
    tool = BaseTool(
        name="test_tool",
        description="A test tool",
        function=mock_fn,
        input_model=MockInput,
        output_model=MockOutput
    )
    assert tool.name == "test_tool"
    assert "test_tool" in tool.to_instruction()

def test_base_tool_invalid_function():
    def sync_fn(input_data: MockInput): pass
    with pytest.raises(ToolConfigurationError):
        BaseTool(
            name="sync_tool",
            description="Sync",
            function=sync_fn,
            input_model=MockInput
        )

@pytest.mark.anyio
async def test_base_tool_execution_success():
    tool = BaseTool(
        name="test_tool",
        description="A test tool",
        function=mock_fn,
        input_model=MockInput,
        output_model=MockOutput
    )
    result = await tool.execute(query="hello")
    assert isinstance(result, MockOutput)
    assert result.result == "processed hello"

@pytest.mark.anyio
async def test_base_tool_execution_input_error():
    tool = BaseTool(
        name="test_tool",
        description="A test tool",
        function=mock_fn,
        input_model=MockInput
    )
    with pytest.raises(ToolInputError):
        await tool.execute(wrong_param="hello")

@pytest.mark.anyio
async def test_base_tool_execution_output_error():
    async def bad_output_fn(input_data: MockInput):
        return {"not": "matching"}
    
    tool = BaseTool(
        name="test_tool",
        description="A test tool",
        function=bad_output_fn,
        input_model=MockInput,
        output_model=MockOutput
    )
    with pytest.raises(ToolOutputError):
        await tool.execute(query="hello")

@pytest.mark.anyio
async def test_base_tool_execute_with_metadata_failure():
    async def fail_fn(input_data: MockInput):
        raise ToolExecutionError("Intentional failure")
    
    tool = BaseTool(
        name="fail_tool",
        description="Fails",
        function=fail_fn,
        input_model=MockInput
    )
    res = await tool.execute_with_metadata(query="hello")
    assert res.success is False
    assert "Intentional failure" in res.error

@pytest.mark.anyio
async def test_base_tool_execute_with_metadata_input_error():
    tool = BaseTool(
        name="input_fail_tool",
        description="Fails",
        function=mock_fn,
        input_model=MockInput
    )
    res = await tool.execute_with_metadata(wrong="param")
    assert res.success is False
    assert "Invalid input" in res.error

def test_base_tool_to_instruction_complex():
    class ComplexInput(BaseModel):
        required_val: str = Field(..., description="Required")
        optional_val: int = Field(default=10, description="Optional")
    
    tool = BaseTool(
        name="complex_tool",
        description="Complex",
        function=mock_fn, # type: ignore
        input_model=ComplexInput
    )
    instr = tool.to_instruction()
    assert "required_val" in instr
    assert "(required)" in instr
    assert "optional_val" in instr
    assert "Required" in instr

def test_base_tool_placeholders():
    tool = BaseTool(
        name="tmpl_tool",
        description="Template",
        function=mock_fn,
        input_model=MockInput,
        input_text="Searching for {query}"
    )
    assert tool.replacements == {"query"}
    rendered = tool.render_input_text(MockInput(query="test"))
    assert rendered == "Searching for test"

def test_base_tool_cost_risk_estimators():
    tool = BaseTool(
        name="est_tool",
        description="Estimators",
        function=mock_fn,
        input_model=MockInput,
        cost_estimate=lambda x: 0.5,
        risk_estimate=lambda x: 0.1
    )
    inp = MockInput(query="test")
    assert tool.estimate_cost(inp) == 0.5
    assert tool.estimate_risk(inp) == 0.1

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
