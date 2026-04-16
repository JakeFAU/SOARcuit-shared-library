import pytest
from shared.config.config import LLMSettings
from shared.registry import (
    AgentDefinition,
    AgentRegistry,
    LLMModel,
    LLMProvider,
    ModelRegistry,
    ModelTier,
    RegistryError,
)


def test_model_registry_registers_only_enabled_providers():
    settings = LLMSettings(gemini_api_key="gemini-key", open_ai_api_key="openai-key")

    registry = ModelRegistry.from_settings(settings)

    assert registry.enabled_providers() == (LLMProvider.GOOGLE, LLMProvider.OPENAI)
    assert "google:high_end" in registry
    assert "openai:normal" in registry
    assert "anthropic:normal" not in registry

    with pytest.raises(RegistryError):
        registry.get("anthropic:normal")


def test_model_registry_builds_models_for_each_provider():
    settings = LLMSettings(
        gemini_api_key="gemini-key",
        open_ai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
    )

    registry = ModelRegistry.from_settings(settings)

    assert registry.build_model("google:fast").system == "google-gla"
    assert registry.build_model("openai:normal").system == "openai"
    assert registry.build_model("anthropic:normal").system == "anthropic"


def test_model_registry_exposes_provider_defaults_and_round_trips_models():
    settings = LLMSettings(open_ai_api_key="openai-key")

    registry = ModelRegistry.from_settings(settings, default_provider=LLMProvider.OPENAI)
    model = registry.get_default(ModelTier.NORMAL)
    cloned = LLMModel.from_dict(model.to_dict())

    assert model.model_id == "gpt-5.4-mini"
    assert registry.get("default:normal") == model
    assert cloned == model


def test_agent_registry_builds_agents_from_registered_definitions():
    settings = LLMSettings(open_ai_api_key="openai-key")
    agent_registry = AgentRegistry.from_settings(
        settings,
        default_provider=LLMProvider.OPENAI,
    )

    assert {
        "assistant",
        "planner",
        "summarizer",
        "reviewer",
        "researcher",
        "duckduckgo_researcher",
        "wikipedia_researcher",
        "fact_checker",
    } <= set(agent_registry.list_agents())

    agent_registry.register(
        AgentDefinition(
            name="task-planner",
            model="default:normal",
            instructions="Write concise plans.",
            retries=2,
        ),
        aliases=("custom-planner",),
    )

    starter_agent = agent_registry.build("summarizer")
    research_agent = agent_registry.build("research")
    wikipedia_agent = agent_registry.build("wikipedia")
    duckduckgo_agent = agent_registry.build("duckduckgo")
    agent = agent_registry.build("custom-planner")

    assert starter_agent.name == "summarizer"
    assert getattr(starter_agent.model, "model_name", None) == "gpt-5.4-nano"
    assert research_agent.name == "researcher"
    assert wikipedia_agent.name == "wikipedia_researcher"
    assert duckduckgo_agent.name == "duckduckgo_researcher"
    assert agent.name == "task-planner"
    assert getattr(agent.model, "model_name", None) == "gpt-5.4-mini"
    assert "task-planner" in agent_registry
