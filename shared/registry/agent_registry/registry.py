from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent

from shared.config.config import AppSettings, LLMSettings
from shared.registry.model_registry.registry import LLMProvider, ModelRegistry, RegistryError

type AgentFactory = Callable[[ModelRegistry], Agent[Any, Any]]
type AgentCapabilitiesFactory = Callable[[], tuple[Any, ...] | list[Any]]
type AgentToolsFactory = Callable[[], tuple[Any, ...] | list[Any]]


@dataclass(slots=True)
class AgentDefinition:
    """Declarative agent registration backed by a model registry lookup."""

    name: str
    model: str
    instructions: str | tuple[str, ...] | None = None
    system_prompt: str | tuple[str, ...] = ()
    output_type: Any = str
    description: str | None = None
    model_settings: dict[str, Any] | None = None
    retries: int = 1
    output_retries: int | None = None
    defer_model_check: bool = False
    instrument: bool | None = None
    tools: tuple[Any, ...] | list[Any] | None = None
    tools_factory: AgentToolsFactory | None = None
    capabilities: tuple[Any, ...] | list[Any] | None = None
    capabilities_factory: AgentCapabilitiesFactory | None = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, model_registry: ModelRegistry) -> Agent[Any, Any]:
        if "name" in self.agent_kwargs or "model" in self.agent_kwargs:
            raise RegistryError("AgentDefinition.agent_kwargs cannot override 'name' or 'model'.")
        if "tools" in self.agent_kwargs or "capabilities" in self.agent_kwargs:
            raise RegistryError(
                "AgentDefinition.agent_kwargs cannot override 'tools' or 'capabilities'."
            )

        agent_kwargs: dict[str, Any] = {
            "output_type": self.output_type,
            "instructions": self.instructions,
            "system_prompt": self.system_prompt,
            "description": self.description,
            "retries": self.retries,
            "output_retries": self.output_retries,
            "defer_model_check": self.defer_model_check,
        }
        if self.model_settings is not None:
            agent_kwargs["model_settings"] = self.model_settings
        if self.instrument is not None:
            agent_kwargs["instrument"] = self.instrument
        tools: list[Any] = []
        if self.tools is not None:
            tools.extend(self.tools)
        if self.tools_factory is not None:
            tools.extend(self.tools_factory())
        if tools:
            agent_kwargs["tools"] = tuple(tools)
        capabilities: list[Any] = []
        if self.capabilities is not None:
            capabilities.extend(self.capabilities)
        if self.capabilities_factory is not None:
            capabilities.extend(self.capabilities_factory())
        if capabilities:
            agent_kwargs["capabilities"] = tuple(capabilities)
        agent_kwargs.update(self.agent_kwargs)

        return Agent(
            name=self.name,
            model=model_registry.build_model(self.model),
            **agent_kwargs,
        )


@dataclass(slots=True)
class AgentRegistry:
    """Registry for named `pydantic_ai.Agent` definitions and factories."""

    model_registry: ModelRegistry
    _definitions: dict[str, AgentDefinition] = field(default_factory=dict, init=False, repr=False)
    _factories: dict[str, AgentFactory] = field(default_factory=dict, init=False, repr=False)
    _aliases: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_settings(
        cls,
        settings: LLMSettings | AppSettings | None = None,
        *,
        default_provider: LLMProvider | None = None,
        include_starter_agents: bool = True,
    ) -> AgentRegistry:
        from shared.registry.agent_registry.starter_bench import register_starter_agents

        model_registry = ModelRegistry.from_settings(
            settings=settings,
            default_provider=default_provider,
        )
        registry = cls(model_registry=model_registry)
        if include_starter_agents:
            register_starter_agents(registry)
        return registry

    def register(
        self, definition: AgentDefinition, *, aliases: tuple[str, ...] = ()
    ) -> AgentDefinition:
        if definition.name in self._factories:
            raise RegistryError(f"Agent '{definition.name}' is already registered.")

        self._definitions[definition.name] = definition
        self._factories[definition.name] = definition.build
        for alias in aliases:
            self.register_alias(alias, definition.name)
        return definition

    def register_many(
        self, definitions: tuple[AgentDefinition, ...] | list[AgentDefinition]
    ) -> None:
        for definition in definitions:
            self.register(definition)

    def register_factory(
        self, name: str, factory: AgentFactory, *, aliases: tuple[str, ...] = ()
    ) -> None:
        if name in self._factories:
            raise RegistryError(f"Agent '{name}' is already registered.")

        self._factories[name] = factory
        for alias in aliases:
            self.register_alias(alias, name)

    def register_alias(self, alias: str, name: str) -> None:
        if name not in self._factories:
            raise RegistryError(f"Cannot register alias '{alias}' for unknown agent '{name}'.")

        existing = self._aliases.get(alias)
        if existing is not None and existing != name:
            raise RegistryError(f"Alias '{alias}' is already registered for agent '{existing}'.")

        self._aliases[alias] = name

    def get_definition(self, name: str) -> AgentDefinition:
        resolved = self._aliases.get(name, name)
        definition = self._definitions.get(resolved)
        if definition is None:
            raise RegistryError(f"Agent '{name}' does not have a declarative definition.")
        return definition

    def build(self, name: str) -> Agent[Any, Any]:
        resolved = self._aliases.get(name, name)
        factory = self._factories.get(resolved)
        if factory is None:
            raise RegistryError(f"Agent '{name}' is not registered.")
        return factory(self.model_registry)

    def list_agents(self) -> tuple[str, ...]:
        return tuple(self._factories.keys())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (name in self._factories or name in self._aliases)
