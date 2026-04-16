# SOARcuit Shared

`soarcuit-shared` contains the small, durable core that multiple SOARcuit
services can share without inheriting Thalamus-specific policy.

Current scope:

- canonical domain models for inbound observations and memes
- stable enums and validation helpers
- Pub/Sub payload decoding, normalization, and message-kind detection
- shared database and configuration primitives
- centralized structured logging
- reusable model and agent registries for `pydantic_ai`

Out of scope:

- routing policy
- telemetry setup
- service-specific persistence implementations and schema-specific helpers

## Logging

`shared` now configures `structlog` centrally on import so every service gets
the same baseline logger behavior by default.

- `gcp` format is used automatically in Google Cloud environments
- `console` format is used on interactive terminals
- `json` format is used otherwise

Environment variables:

- `SOAR_LOG_LEVEL` or `LOG_LEVEL`
- `SOAR_LOG_FORMAT` or `LOG_FORMAT`
- `SOAR_AUTO_CONFIGURE_LOGGING=false` to opt out of import-time setup

Recommended app bootstrap:

```python
from shared import configure_logging, get_logger

configure_logging(service_name="cortex", force=True)
logger = get_logger(__name__)

logger.info("service_starting", component="api")
```

Context helpers are also exported:

```python
from shared import bind_log_context, clear_log_context

bind_log_context(request_id="abc123", tenant="demo")
clear_log_context()
```

## Model Registry

`shared.registry.ModelRegistry` exposes a provider-aware registry of available
LLMs based on configured API keys.

- providers currently supported: Google, OpenAI, Anthropic
- default model tiers: `high_end`, `normal`, `fast`
- provider aliases: `google:normal`, `openai:fast`, `anthropic:high_end`
- default-provider aliases: `default:normal`, `default:fast`, etc.

Example:

```python
from shared.config.config import LLMSettings
from shared.registry import LLMProvider, ModelRegistry

settings = LLMSettings(open_ai_api_key="test-key")
registry = ModelRegistry.from_settings(settings, default_provider=LLMProvider.OPENAI)

model = registry.build_model("default:normal")
```

## Agent Registry

`shared.registry.AgentRegistry` is a bench of reusable `pydantic_ai.Agent`
definitions built on top of the model registry.

Included starter agents:

- `assistant`
- `planner`
- `summarizer`
- `reviewer`
- `researcher`
- `duckduckgo_researcher`
- `wikipedia_researcher`
- `fact_checker`

Example:

```python
from shared.config.config import LLMSettings
from shared.registry import AgentRegistry, LLMProvider

settings = LLMSettings(open_ai_api_key="test-key")
agents = AgentRegistry.from_settings(settings, default_provider=LLMProvider.OPENAI)

planner = agents.build("planner")
researcher = agents.build("research")
```

You can also register additional agents declaratively:

```python
from shared.registry import AgentDefinition

agents.register(
    AgentDefinition(
        name="draft-writer",
        model="default:normal",
        instructions="Write concise first drafts from bullet points.",
    )
)
```

## Package Surface

Common entry points:

- `shared.get_logger`
- `shared.configure_logging`
- `shared.initialize_logger`
- `shared.initialize_tracing`
- `shared.registry.ModelRegistry`
- `shared.registry.AgentRegistry`

## Deployment

Publish to private artifactory at
<https://us-east4-python.pkg.dev/soarcuit/soarcuit-dev-python-repo/simple/>
