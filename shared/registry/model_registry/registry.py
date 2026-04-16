from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from shared.config.config import AppSettings, LLMSettings


class RegistryError(Exception):
    """Raised when a registry lookup or registration fails."""


class LLMProvider(StrEnum):
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ModelTier(StrEnum):
    HIGH_END = "high_end"
    NORMAL = "normal"
    FAST = "fast"


@dataclass(slots=True, frozen=True)
class LLMModel:
    """
    Metadata and factory helpers for a registered LLM.

    Token costs are expressed per million tokens when provided.
    """

    provider: LLMProvider
    model_id: str
    model_name: str | None = None
    input_token_cost: float | None = None
    output_token_cost: float | None = None
    input_token_limit: int | None = None
    output_token_limit: int | None = None

    def __post_init__(self) -> None:
        if self.model_name is None:
            object.__setattr__(self, "model_name", self.model_id)

    def to_dict(self) -> dict[str, str | float | int | None]:
        return {
            "provider": self.provider.value,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "input_token_cost": self.input_token_cost,
            "output_token_cost": self.output_token_cost,
            "input_token_limit": self.input_token_limit,
            "output_token_limit": self.output_token_limit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | float | int | None]) -> LLMModel:
        provider = data.get("provider")
        model_id = data.get("model_id")

        if not provider:
            raise RegistryError(
                "LLMModel dict must contain a 'provider' field to instantiate an LLMModel."
            )
        if not model_id:
            raise RegistryError(
                "LLMModel dict must contain a 'model_id' field to instantiate an LLMModel."
            )

        return cls(
            provider=LLMProvider(provider),
            model_id=str(model_id),
            model_name=data.get("model_name") if isinstance(data.get("model_name"), str) else None,  # type: ignore
            input_token_cost=_as_float(data.get("input_token_cost")),
            output_token_cost=_as_float(data.get("output_token_cost")),
            input_token_limit=_as_int(data.get("input_token_limit")),
            output_token_limit=_as_int(data.get("output_token_limit")),
        )

    def is_enabled(self, settings: LLMSettings) -> bool:
        if self.provider == LLMProvider.GOOGLE:
            return settings.gemini_enabled
        if self.provider == LLMProvider.OPENAI:
            return settings.open_ai_enabled
        return settings.anthropic_enabled

    def build_model(self, settings: LLMSettings) -> Model:
        if not self.is_enabled(settings):
            raise RegistryError(
                f"{self.provider.value} is not enabled, cannot build model '{self.model_id}'."
            )

        if self.provider == LLMProvider.GOOGLE:
            api_key = _require_google_api_key(settings)
            return GoogleModel(self.model_id, provider=GoogleProvider(api_key=api_key))

        if self.provider == LLMProvider.OPENAI:
            api_key = _require_openai_api_key(settings)
            return OpenAIChatModel(
                self.model_id,
                provider=OpenAIProvider(api_key=api_key),
            )

        api_key = _require_anthropic_api_key(settings)
        _ensure_anthropic_compatibility()
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(
            self.model_id, provider=AnthropicProvider(api_key=api_key)
        )


@dataclass(slots=True)
class ModelRegistry:
    """Provider-aware registry for available LLM definitions."""

    llm_settings: LLMSettings
    default_provider: LLMProvider | None = None
    _models: dict[str, LLMModel] = field(default_factory=dict, init=False, repr=False)
    _aliases: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _defaults: dict[LLMProvider, dict[ModelTier, str]] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def from_settings(
        cls,
        settings: LLMSettings | AppSettings | None = None,
        *,
        default_provider: LLMProvider | None = None,
    ) -> ModelRegistry:
        llm_settings = _resolve_llm_settings(settings)
        registry = cls(llm_settings=llm_settings, default_provider=default_provider)
        registry.register_enabled_defaults()
        if registry.default_provider is None:
            registry.default_provider = registry.first_enabled_provider()
        registry._register_default_provider_aliases()
        return registry

    def register(self, model: LLMModel, *, aliases: tuple[str, ...] = ()) -> LLMModel:
        existing = self._models.get(model.model_id)
        if existing is not None and existing != model:
            raise RegistryError(f"Model '{model.model_id}' is already registered.")

        self._models[model.model_id] = model
        self.register_alias(f"{model.provider.value}:{model.model_id}", model.model_id)
        for alias in aliases:
            self.register_alias(alias, model.model_id)
        return model

    def register_alias(self, alias: str, model_id: str) -> None:
        if model_id not in self._models:
            raise RegistryError(
                f"Cannot register alias '{alias}' for unknown model '{model_id}'."
            )

        existing = self._aliases.get(alias)
        if existing is not None and existing != model_id:
            raise RegistryError(
                f"Alias '{alias}' is already registered for model '{existing}'."
            )

        self._aliases[alias] = model_id

    def register_default(
        self, provider: LLMProvider, tier: ModelTier, model: LLMModel
    ) -> None:
        self.register(model)
        self._defaults.setdefault(provider, {})[tier] = model.model_id
        self.register_alias(f"{provider.value}:{tier.value}", model.model_id)

    def register_enabled_defaults(self) -> None:
        if self.llm_settings.gemini_enabled:
            for model in GOOGLE_MODELS:
                self.register(model)
            self.register_default(
                LLMProvider.GOOGLE, ModelTier.HIGH_END, GEMINI_HIGH_END
            )
            self.register_default(LLMProvider.GOOGLE, ModelTier.NORMAL, GEMINI_NORMAL)
            self.register_default(LLMProvider.GOOGLE, ModelTier.FAST, GEMINI_FAST)

        if self.llm_settings.open_ai_enabled:
            for model in OPENAI_MODELS:
                self.register(model)
            self.register_default(
                LLMProvider.OPENAI, ModelTier.HIGH_END, OPENAI_HIGH_END
            )
            self.register_default(LLMProvider.OPENAI, ModelTier.NORMAL, OPENAI_NORMAL)
            self.register_default(LLMProvider.OPENAI, ModelTier.FAST, OPENAI_FAST)

        if self.llm_settings.anthropic_enabled:
            for model in ANTHROPIC_MODELS:
                self.register(model)
            self.register_default(
                LLMProvider.ANTHROPIC, ModelTier.HIGH_END, ANTHROPIC_HIGH_END
            )
            self.register_default(
                LLMProvider.ANTHROPIC, ModelTier.NORMAL, ANTHROPIC_NORMAL
            )
            self.register_default(LLMProvider.ANTHROPIC, ModelTier.FAST, ANTHROPIC_FAST)

    def first_enabled_provider(self) -> LLMProvider | None:
        for provider in PROVIDER_PRIORITY:
            if provider == LLMProvider.GOOGLE and self.llm_settings.gemini_enabled:
                return provider
            if provider == LLMProvider.OPENAI and self.llm_settings.open_ai_enabled:
                return provider
            if (
                provider == LLMProvider.ANTHROPIC
                and self.llm_settings.anthropic_enabled
            ):
                return provider
        return None

    def get(self, key: str) -> LLMModel:
        model_id = self._aliases.get(key, key)
        model = self._models.get(model_id)
        if model is None:
            raise RegistryError(f"Model '{key}' is not registered.")
        return model

    def get_default(
        self, tier: ModelTier = ModelTier.NORMAL, *, provider: LLMProvider | None = None
    ) -> LLMModel:
        target_provider = provider or self.default_provider
        if target_provider is None:
            raise RegistryError("No default provider is configured.")

        provider_defaults = self._defaults.get(target_provider)
        if provider_defaults is None or tier not in provider_defaults:
            raise RegistryError(
                f"No '{tier.value}' default model is registered for provider '{target_provider.value}'."
            )
        return self.get(provider_defaults[tier])

    def build_model(self, key: str) -> Model:
        return self.get(key).build_model(self.llm_settings)

    def enabled_providers(self) -> tuple[LLMProvider, ...]:
        providers: list[LLMProvider] = []
        for provider in PROVIDER_PRIORITY:
            if provider in self._defaults:
                providers.append(provider)
        return tuple(providers)

    def list_models(
        self, *, provider: LLMProvider | None = None
    ) -> tuple[LLMModel, ...]:
        models = tuple(self._models.values())
        if provider is None:
            return models
        return tuple(model for model in models if model.provider == provider)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self._models or key in self._aliases)

    def _register_default_provider_aliases(self) -> None:
        if self.default_provider is None:
            return

        for tier, model_id in self._defaults.get(self.default_provider, {}).items():
            self.register_alias(f"default:{tier.value}", model_id)


def _resolve_llm_settings(settings: LLMSettings | AppSettings | None) -> LLMSettings:
    if settings is None:
        return LLMSettings()
    if isinstance(settings, AppSettings):
        return settings.llm_settings
    return settings


def _require_google_api_key(settings: LLMSettings) -> str:
    api_key = settings.gemini_api_key
    if api_key is None:
        raise RegistryError("Gemini is not configured.")
    return api_key.get_secret_value()


def _require_openai_api_key(settings: LLMSettings) -> str:
    api_key = settings.open_ai_api_key
    if api_key is None:
        raise RegistryError("OpenAI is not configured.")
    return api_key.get_secret_value()


def _require_anthropic_api_key(settings: LLMSettings) -> str:
    api_key = settings.anthropic_api_key
    if api_key is None:
        raise RegistryError("Anthropic is not configured.")
    return api_key.get_secret_value()


def _ensure_anthropic_compatibility() -> None:
    import anthropic.types.beta.beta_web_search_tool_20250305_param as compat

    if not hasattr(compat, "UserLocation") and hasattr(compat, "BetaUserLocationParam"):
        compat.UserLocation = compat.BetaUserLocationParam  # type: ignore


def _as_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_int(value: str | float | int | None) -> int | None:
    if value is None:
        return None
    return int(value)


GEMINI3_1_PRO_PREVIEW = LLMModel(
    provider=LLMProvider.GOOGLE,
    model_id="gemini-3.1-pro-preview",
    model_name="Gemini 3.1 Pro Preview",
    input_token_cost=2,
    output_token_cost=12,
    input_token_limit=1_048_576,
    output_token_limit=65_536,
)
GEMINI3_0_FLASH_PREVIEW = LLMModel(
    provider=LLMProvider.GOOGLE,
    model_id="gemini-3.0-flash-preview",
    model_name="Gemini 3.0 Flash Preview",
    input_token_cost=0.5,
    output_token_cost=2,
    input_token_limit=1_048_576,
    output_token_limit=65_536,
)
GEMINI3_1_FLASH_LITE_PREVIEW = LLMModel(
    provider=LLMProvider.GOOGLE,
    model_id="gemini-3.1-flash-lite-preview",
    model_name="Gemini 3.1 Flash Lite Preview",
    input_token_cost=0.25,
    output_token_cost=1.5,
    input_token_limit=1_048_576,
    output_token_limit=65_536,
)
GEMINI2_5_PRO = LLMModel(
    provider=LLMProvider.GOOGLE,
    model_id="gemini-2.5-pro",
    model_name="Gemini 2.5 Pro",
    input_token_cost=1.25,
    output_token_cost=10,
    input_token_limit=1_048_576,
    output_token_limit=65_536,
)
GEMINI2_5_FLASH = LLMModel(
    provider=LLMProvider.GOOGLE,
    model_id="gemini-2.5-flash",
    model_name="Gemini 2.5 Flash",
    input_token_cost=0.3,
    output_token_cost=2.5,
    input_token_limit=1_048_576,
    output_token_limit=65_536,
)
GEMINI2_5_FLASH_LITE = LLMModel(
    provider=LLMProvider.GOOGLE,
    model_id="gemini-2.5-flash-lite",
    model_name="Gemini 2.5 Flash Lite",
    input_token_cost=0.1,
    output_token_cost=0.4,
    input_token_limit=1_048_576,
    output_token_limit=65_536,
)

GPT_5_4 = LLMModel(
    provider=LLMProvider.OPENAI,
    model_id="gpt-5.4",
    model_name="GPT-5.4",
)
GPT_5_4_MINI = LLMModel(
    provider=LLMProvider.OPENAI,
    model_id="gpt-5.4-mini",
    model_name="GPT-5.4 Mini",
)
GPT_5_4_NANO = LLMModel(
    provider=LLMProvider.OPENAI,
    model_id="gpt-5.4-nano",
    model_name="GPT-5.4 Nano",
)

CLAUDE_OPUS_4_6 = LLMModel(
    provider=LLMProvider.ANTHROPIC,
    model_id="claude-opus-4-6",
    model_name="Claude Opus 4.6",
)
CLAUDE_SONNET_4_6 = LLMModel(
    provider=LLMProvider.ANTHROPIC,
    model_id="claude-sonnet-4-6",
    model_name="Claude Sonnet 4.6",
)
CLAUDE_HAIKU_4_5 = LLMModel(
    provider=LLMProvider.ANTHROPIC,
    model_id="claude-haiku-4-5",
    model_name="Claude Haiku 4.5",
)

GEMINI_HIGH_END = GEMINI3_1_PRO_PREVIEW
GEMINI_NORMAL = GEMINI2_5_PRO
GEMINI_FAST = GEMINI2_5_FLASH_LITE

OPENAI_HIGH_END = GPT_5_4
OPENAI_NORMAL = GPT_5_4_MINI
OPENAI_FAST = GPT_5_4_NANO

ANTHROPIC_HIGH_END = CLAUDE_OPUS_4_6
ANTHROPIC_NORMAL = CLAUDE_SONNET_4_6
ANTHROPIC_FAST = CLAUDE_HAIKU_4_5

GOOGLE_MODELS = (
    GEMINI3_1_PRO_PREVIEW,
    GEMINI3_0_FLASH_PREVIEW,
    GEMINI3_1_FLASH_LITE_PREVIEW,
    GEMINI2_5_PRO,
    GEMINI2_5_FLASH,
    GEMINI2_5_FLASH_LITE,
)

OPENAI_MODELS = (
    GPT_5_4,
    GPT_5_4_MINI,
    GPT_5_4_NANO,
)

ANTHROPIC_MODELS = (
    CLAUDE_OPUS_4_6,
    CLAUDE_SONNET_4_6,
    CLAUDE_HAIKU_4_5,
)

PROVIDER_PRIORITY = (
    LLMProvider.GOOGLE,
    LLMProvider.OPENAI,
    LLMProvider.ANTHROPIC,
)
