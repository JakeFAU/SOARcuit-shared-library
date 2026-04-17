"""
SOARcuit Tool Abstraction Layer.

This module defines the foundational `BaseTool` class, which provides a production-grade
interface for LLM-callable functions. It enforces strict input/output contracts
using Pydantic models and provides first-class support for observability,
cost estimation, and template rendering.

Key Features:
- Contract Enforcement: Validates LLM-provided arguments against a Pydantic schema.
- Automatic Instruction Generation: Introspects the input model to generate
  technically precise tool definitions for the LLM.
- Detailed Telemetry: Instruments every execution step with OpenTelemetry spans.
- Result Normalization: Automatically converts raw function returns into
  validated Pydantic models or JSON strings.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import regex as re
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from shared.infrastructure.logging import get_logger
from shared.observability.tracer import get_tracer

logger = get_logger("tool")
tracer = get_tracer("tool")

InputModelT = TypeVar("InputModelT", bound=BaseModel)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)

PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")


class ToolException(Exception):
    """Base exception for all tool-related errors in the SOARcuit ecosystem."""


class ToolConfigurationError(ToolException):
    """Raised when a tool is instantiated with invalid parameters or signatures."""


class ToolExecutionError(ToolException):
    """Raised when the underlying tool function fails during execution."""


class ToolInputError(ToolException):
    """Raised when the LLM-provided arguments fail validation against the input model."""


class ToolOutputError(ToolException):
    """Raised when the tool's return value fails validation against the output model."""


class ToolExecutionResult(BaseModel):
    """
    Normalized metadata for a tool execution episode.

    Used for orchestration, logging, and downstream routing where the full
    telemetry of the tool call is required.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., description="The unique identifier of the tool.")
    success: bool = Field(..., description="Whether the execution completed without error.")
    raw_output: Any | None = Field(None, description="The unparsed result of the tool function.")
    structured_output: BaseModel | None = Field(
        None, description="The Pydantic-validated output model."
    )
    error: str | None = Field(None, description="Error message if success is False.")
    estimated_cost: float | None = Field(None, description="Direct dollar cost of the operation.")
    estimated_risk: float | None = Field(None, description="Calculated operational risk score.")


class BaseTool(BaseModel):
    """
    Production-oriented Tool Abstraction.

    BaseTool wraps an async function with a strict Pydantic-defined contract.
    It ensures that the LLM only interacts with the tool via its validated
    schema and that every call is observable and costed.

    Args:
        name: Unique tool identifier.
        description: Human-readable description for the LLM.
        function: The async callable that performs the actual work.
        input_model: The Pydantic model defining the expected arguments.
        output_model: Optional Pydantic model for the validated result.
        input_text: Optional template for rendering text-based inputs.
        cost_estimate: Optional callable to estimate execution cost.
        risk_estimate: Optional callable to estimate operational risk.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(..., min_length=1, description="Unique tool name")
    description: str = Field(..., min_length=1, description="Human-readable description")
    function: Callable[[Any], Awaitable[Any]] = Field(
        ...,
        description="Async callable accepting validated input model instance",
    )
    system_instructions: str | None = Field(
        default=None,
        description="Optional system instructions associated with the tool",
    )
    input_text: str | None = Field(
        default=None,
        description=r"Optional input template supporting {placeholder} interpolation",
    )
    input_model: type[BaseModel] = Field(
        ...,
        description="Pydantic model class describing tool input",
    )
    output_model: type[BaseModel] | None = Field(
        default=None,
        description="Optional Pydantic model class describing structured tool output",
    )
    cost_estimate: Callable[[BaseModel], float] | None = Field(
        default=None,
        description="Optional estimator for execution cost",
    )
    risk_estimate: Callable[[BaseModel], float] | None = Field(
        default=None,
        description="Optional estimator for execution risk",
    )

    _replacements_cache: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """Performs static validation of the tool configuration at startup."""
        self._validate_function_signature()
        self._validate_template_placeholders()

    @property
    def replacements(self) -> set[str]:
        """Returns the set of placeholder keys found in the input_text template."""
        return set(self._replacements_cache)

    def render_input_text(self, validated_input: BaseModel) -> str | None:
        """Interpolates placeholders in input_text using validated model fields."""
        if not self.input_text:
            return None
        return self.input_text.format(**validated_input.model_dump())

    def estimate_cost(self, validated_input: BaseModel) -> float | None:
        """Calculates the estimated cost for the specific input provided."""
        if self.cost_estimate is None:
            return None
        try:
            return float(self.cost_estimate(validated_input))
        except Exception as exc:
            raise ToolConfigurationError(
                f"Cost estimator failed for tool '{self.name}': {exc}"
            ) from exc

    def estimate_risk(self, validated_input: BaseModel) -> float | None:
        """Calculates the operational risk for the specific input provided."""
        if self.risk_estimate is None:
            return None
        try:
            return float(self.risk_estimate(validated_input))
        except Exception as exc:
            raise ToolConfigurationError(
                f"Risk estimator failed for tool '{self.name}': {exc}"
            ) from exc

    def to_instruction(self) -> str:
        """
        Generates a technically precise instruction block for the LLM.

        Introspects the input_model and its field descriptions to provide the
        LLM with a clear understanding of the tool's required and optional arguments.
        """
        schema = self.input_model.model_json_schema()
        props = schema.get("properties", {})
        required = schema.get("required", [])

        arg_desc = []
        for name, info in props.items():
            req_str = " (required)" if name in required else ""
            desc = info.get("description", "No description.")
            arg_desc.append(f"  - {name}: {info.get('type')}{req_str}. {desc}")

        args_block = "\n".join(arg_desc)
        return f"TOOL: {self.name}\nDESCRIPTION: {self.description}\nARGUMENTS:\n{args_block}\n"

    async def execute(self, **kwargs: Any) -> Any:
        """
        Main execution entry point for the tool.

        Validates inputs, captures telemetry, executes the underlying function,
        and normalizes the output.

        Returns:
            The validated result (either the raw function return or the output_model).

        Raises:
            ToolInputError: If arguments fail validation.
            ToolExecutionError: If the function itself fails.
            ToolOutputError: If the result fails validation.
        """
        with tracer.start_as_current_span(f"tool.{self.name}") as span:
            span.set_attribute("tool.name", self.name)
            span.set_attribute("tool.description", self.description)

            try:
                validated_input = self._validate_input(kwargs, span)
                rendered_input = self.render_input_text(validated_input)

                if rendered_input is not None:
                    span.set_attribute("tool.rendered_input", rendered_input)

                estimated_cost = self.estimate_cost(validated_input)
                estimated_risk = self.estimate_risk(validated_input)

                if estimated_cost is not None:
                    span.set_attribute("tool.estimated_cost", estimated_cost)
                if estimated_risk is not None:
                    span.set_attribute("tool.estimated_risk", estimated_risk)

                logger.info(
                    "Executing tool",
                    tool_name=self.name,
                    estimated_cost=estimated_cost,
                    estimated_risk=estimated_risk,
                )

                raw_result = await self._execute_function(validated_input, span)
                normalized_result = self._normalize_output(raw_result, span)

                span.set_status(Status(StatusCode.OK))
                span.add_event("tool.execution.completed")

                return normalized_result

            except ToolException:
                # Re-raise known tool exceptions
                raise
            except Exception as exc:
                wrapped = ToolExecutionError(
                    f"Unexpected error executing tool '{self.name}': {exc}"
                )
                self._record_error(span, wrapped)
                raise wrapped from exc

    async def execute_with_metadata(self, **kwargs: Any) -> ToolExecutionResult:
        """
        Executes the tool and returns a comprehensive ToolExecutionResult.

        Unlike `execute()`, this method does not raise on business-logic failures;
        it captures them in the `success` and `error` fields of the result model.
        Useful for orchestration where failure is a valid reasoning branch.
        """
        try:
            try:
                validated_input = self.input_model.model_validate(kwargs)
            except ValidationError as exc:
                raise ToolInputError(f"Invalid input for tool '{self.name}': {exc}") from exc

            estimated_cost = self.estimate_cost(validated_input)
            estimated_risk = self.estimate_risk(validated_input)

            result = await self.execute(**kwargs)

            if isinstance(result, BaseModel):
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=True,
                    raw_output=result.model_dump(),
                    structured_output=result,
                    error=None,
                    estimated_cost=estimated_cost,
                    estimated_risk=estimated_risk,
                )

            return ToolExecutionResult(
                tool_name=self.name,
                success=True,
                raw_output=result,
                structured_output=None,
                error=None,
                estimated_cost=estimated_cost,
                estimated_risk=estimated_risk,
            )
        except ToolException as exc:
            validated_input = None
            try:
                validated_input = self.input_model.model_validate(kwargs)
            except Exception:
                pass

            estimated_cost = (
                self.estimate_cost(validated_input)
                if validated_input is not None and self.cost_estimate is not None
                else None
            )
            estimated_risk = (
                self.estimate_risk(validated_input)
                if validated_input is not None and self.risk_estimate is not None
                else None
            )

            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                raw_output=None,
                structured_output=None,
                error=str(exc),
                estimated_cost=estimated_cost,
                estimated_risk=estimated_risk,
            )

    def _validate_function_signature(self) -> None:
        """Ensures the wrapped function is async and accepts the correct arguments."""
        if not inspect.iscoroutinefunction(self.function):
            raise ToolConfigurationError(f"Tool '{self.name}' function must be async.")

        sig = inspect.signature(self.function)
        params = list(sig.parameters.values())

        if len(params) != 1:
            raise ToolConfigurationError(
                f"Tool '{self.name}' function must accept exactly one argument "
                "(the validated input model)."
            )

    def _validate_template_placeholders(self) -> None:
        """Ensures that any placeholders in input_text map to fields on the input model."""
        if not self.input_text:
            self._replacements_cache = set()
            return

        placeholders = set(PLACEHOLDER_PATTERN.findall(self.input_text))
        model_fields = set(self.input_model.model_fields.keys())
        missing = placeholders - model_fields

        if missing:
            raise ToolConfigurationError(
                f"Tool '{self.name}' input_text references unknown placeholders: {sorted(missing)}"
            )

        self._replacements_cache = placeholders

    def _validate_input(self, kwargs: dict[str, Any], span: Span) -> BaseModel:
        """Validates raw dictionary input against the input_model schema."""
        span.add_event("tool.input.validation.started")
        try:
            validated = self.input_model.model_validate(kwargs)
            span.add_event("tool.input.validation.completed")
            return validated
        except ValidationError as exc:
            raise ToolInputError(f"Invalid input for tool '{self.name}': {exc}") from exc

    async def _execute_function(self, validated_input: BaseModel, span: Span) -> Any:
        """Invokes the underlying async function."""
        span.add_event("tool.function.execution.started")
        try:
            result = await self.function(validated_input)
            span.add_event("tool.function.execution.completed")
            return result
        except Exception as exc:
            raise ToolExecutionError(f"Error executing tool '{self.name}': {exc}") from exc

    def _normalize_output(self, result: Any, span: Span) -> Any:
        """Validates and normalizes the tool return value against the output_model."""
        if self.output_model is None:
            return result

        span.add_event("tool.output.validation.started")

        try:
            if isinstance(result, self.output_model):
                span.add_event("tool.output.validation.completed")
                return result

            if isinstance(result, BaseModel):
                candidate = result.model_dump()
                model = self.output_model.model_validate(candidate)
                span.add_event("tool.output.validation.completed")
                return model

            if isinstance(result, dict):
                model = self.output_model.model_validate(result)
                span.add_event("tool.output.validation.completed")
                return model

            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                except json.JSONDecodeError as exc:
                    raise ToolOutputError(
                        f"Tool '{self.name}' returned a string that is not valid JSON."
                    ) from exc

                model = self.output_model.model_validate(parsed)
                span.add_event("tool.output.validation.completed")
                return model

            raise ToolOutputError(
                f"Unsupported output type for tool '{self.name}': {type(result).__name__}"
            )

        except ValidationError as exc:
            raise ToolOutputError(f"Invalid output for tool '{self.name}': {exc}") from exc

    def _record_error(self, span: Span, exc: Exception) -> None:
        """Instruments and logs an execution failure."""
        message = str(exc)
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, message))
        logger.exception(
            "Tool execution failed",
            tool_name=self.name,
            error=message,
        )
