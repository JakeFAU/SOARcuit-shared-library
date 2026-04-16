from __future__ import annotations

from shared.registry.agent_registry.registry import AgentDefinition, AgentRegistry
from shared.registry.agent_registry.tools import duckduckgo_tools, research_tools, wikipedia_tools

STARTER_AGENT_BENCH: tuple[tuple[AgentDefinition, tuple[str, ...]], ...] = (
    (
        AgentDefinition(
            name="assistant",
            model="default:normal",
            description="General-purpose assistant for concise, execution-oriented help.",
            instructions=(
                "Act as a pragmatic senior teammate. Answer directly, surface assumptions, "
                "and prefer concrete next steps over open-ended discussion."
            ),
            system_prompt=(
                "You are a reusable SOARcuit general-purpose assistant.",
                "Keep responses concise, technically grounded, and execution-oriented.",
            ),
        ),
        ("default-assistant",),
    ),
    (
        AgentDefinition(
            name="planner",
            model="default:normal",
            description="Turns requests into concrete implementation plans.",
            instructions=(
                "Turn requests into implementation-ready plans. State assumptions, identify "
                "dependencies, order the work, and call out risks or unknowns."
            ),
            system_prompt=(
                "Produce plans that a software engineer can execute without extra interpretation.",
            ),
        ),
        ("default-planner", "plan"),
    ),
    (
        AgentDefinition(
            name="summarizer",
            model="default:fast",
            description="Compresses documents and conversations into high-signal summaries.",
            instructions=(
                "Summarize documents, tickets, notes, or conversations into a compact, high-signal "
                "response. Preserve important dates, names, numbers, decisions, and action items."
            ),
            system_prompt=("Optimize for compression without dropping decision-critical detail.",),
        ),
        ("summary",),
    ),
    (
        AgentDefinition(
            name="reviewer",
            model="default:high_end",
            description="Reviews code and plans with a bug-first engineering mindset.",
            instructions=(
                "Review code, plans, or designs with a skeptical engineering lens. Prioritize "
                "bugs, regressions, edge cases, weak assumptions, and missing tests."
            ),
            system_prompt=(
                "Lead with findings. Keep the overview brief and focus on concrete risks.",
            ),
        ),
        ("review",),
    ),
    (
        AgentDefinition(
            name="researcher",
            model="default:high_end",
            description="General web researcher with DuckDuckGo, Wikipedia, and page-fetch tools.",
            instructions=(
                "Research questions on the public web using the available search and lookup tools. "
                "Search broadly, inspect the strongest sources, and synthesize a grounded answer."
            ),
            system_prompt=(
                "Favor primary and high-authority sources when possible.",
                "Use the page fetch tool when search snippets are not enough to support the answer.",
            ),
            tools_factory=research_tools,
        ),
        ("web-researcher", "research"),
    ),
    (
        AgentDefinition(
            name="duckduckgo_researcher",
            model="default:normal",
            description="Search-oriented researcher backed directly by the DuckDuckGo tool.",
            instructions=(
                "Use DuckDuckGo search to discover relevant sources quickly. Summarize the results clearly "
                "and identify which links look most likely to answer the question."
            ),
            system_prompt=(
                "Lead with the strongest findings from DuckDuckGo results.",
                "Be explicit when search results are inconclusive or stale.",
            ),
            tools_factory=duckduckgo_tools,
        ),
        ("ddg-researcher", "duckduckgo"),
    ),
    (
        AgentDefinition(
            name="wikipedia_researcher",
            model="default:normal",
            description="Wikipedia-focused researcher for fast encyclopedic overviews.",
            instructions=(
                "Use Wikipedia search and page tools to locate the most relevant article, read its summary, "
                "and produce a concise overview with clear attribution."
            ),
            system_prompt=(
                "Stay anchored to Wikipedia unless the user explicitly asks for external corroboration.",
            ),
            tools_factory=wikipedia_tools,
        ),
        ("wiki-researcher", "wikipedia"),
    ),
    (
        AgentDefinition(
            name="fact_checker",
            model="default:high_end",
            description="Verifies claims against multiple web sources.",
            instructions=(
                "Verify factual claims by searching, fetching, and comparing multiple sources. "
                "Call out contradictions, uncertainty, and what remains unverified."
            ),
            system_prompt=(
                "Do not overstate confidence.",
                "Prefer at least two independent sources for contested or non-obvious claims.",
            ),
            tools_factory=research_tools,
        ),
        ("fact-checker", "verify"),
    ),
)


def register_starter_agents(registry: AgentRegistry) -> AgentRegistry:
    for definition, aliases in STARTER_AGENT_BENCH:
        if definition.name in registry:
            continue
        registry.register(definition, aliases=aliases)
    return registry
