from typing import Any

from google import genai
from google.genai import types

from ...config.config import LLMSettings
from ..base import LLMProvider
from ..types import ChatMessage, ChatResponse, Role, T, TokenUsage


class GeminiProvider(LLMProvider):
    """
    Gemini provider implementation.
    Handles SDK client as a singleton per provider instance.
    """

    def __init__(self, settings: LLMSettings):
        self._settings = settings
        self._client: genai.Client | None = None
        self._default_model = settings.default_gemini_model

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self._settings.gemini_api_key:
                raise ValueError("Gemini API key is missing in configuration.")
            self._client = genai.Client(api_key=self._settings.gemini_api_key.get_secret_value())
        return self._client

    def _prepare_contents(
        self, messages: list[ChatMessage]
    ) -> tuple[types.Content | None, list[types.Content]]:
        system_instruction = None
        contents = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = types.Content(parts=[types.Part(text=msg.content)])
            else:
                role = "user" if msg.role == Role.USER else "model"
                contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        return system_instruction, contents

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        model = model or self._default_model
        system_instruction, contents = self._prepare_contents(messages)

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instruction, **kwargs),
        )

        usage_meta = response.usage_metadata
        usage = TokenUsage(
            prompt_tokens=usage_meta.prompt_token_count or 0 if usage_meta else 0,
            completion_tokens=usage_meta.candidates_token_count or 0 if usage_meta else 0,
            total_tokens=usage_meta.total_token_count or 0 if usage_meta else 0,
        )

        return ChatResponse(
            content=response.text or "",
            usage=usage,
            model=model,
        )

    async def chat_structured(
        self,
        messages: list[ChatMessage],
        response_model: type[T],
        model: str | None = None,
        **kwargs: Any,
    ) -> T:
        model = model or self._default_model
        system_instruction, contents = self._prepare_contents(messages)

        # Prepare schema and strip additionalProperties as it's not supported by Gemini API.
        # We convert the Pydantic model to a raw JSON schema dict first.
        schema = response_model.model_json_schema()
        self._strip_additional_properties(schema)

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                **kwargs,
            ),
        )

        if not response.text:
            raise ValueError(
                f"Gemini failed to return text for structured output with model {model}"
            )

        return response_model.model_validate_json(response.text)

    def _strip_additional_properties(self, schema: Any) -> None:
        """
        Recursively strip 'additionalProperties' from a JSON schema.

        The Gemini API structured output implementation (via the Google GenAI SDK)
        raises a ValueError if 'additionalProperties' is present in the schema,
        even if set to True/False.

        If a schema element indicates it is an object only via 'additionalProperties'
        (with no fixed properties defined), we also strip the 'type' to make it
        an 'Any' schema, allowing the model to produce arbitrary keys as requested
        in the system prompt (e.g., for tool arguments).
        """
        if not isinstance(schema, dict):
            return

        has_additional = "additionalProperties" in schema
        schema.pop("additionalProperties", None)

        # If it was defined as an object primarily to support arbitrary properties,
        # strip the type to allow any structure (equivalent to 'Any' schema).
        if has_additional and "properties" not in schema and schema.get("type") == "object":
            schema.pop("type", None)

        if "properties" in schema:
            for prop in schema["properties"].values():
                self._strip_additional_properties(prop)

        if "items" in schema:
            self._strip_additional_properties(schema["items"])

        if "$defs" in schema:
            for d in schema["$defs"].values():
                self._strip_additional_properties(d)

        # Handle standard JSON schema combinators
        for key in ("allOf", "anyOf", "oneOf"):
            if key in schema:
                for sub_schema in schema[key]:
                    self._strip_additional_properties(sub_schema)
