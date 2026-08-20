"""Researcher agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        # Gather sources using SearchClient
        sources = self.search_client.search(query=query, max_results=max_sources)
        state.sources = sources

        # Format sources text for LLM
        sources_summary = "\n".join(
            [f"- Title: {s.title}\n  URL: {s.url}\n  Snippet: {s.snippet}" for s in sources]
        )

        system_prompt = (
            "You are a Senior Technical Researcher. Your goal is to gather key factual evidence "
            "and produce concise, well-structured research notes from provided reference documents."
        )
        user_prompt = (
            f"Topic: {query}\nAudience: {state.request.audience}\n\n"
            f"Sources Documented:\n{sources_summary}\n\n"
            "Produce structured research notes summarizing key facts, findings, "
            "and technical concepts."
        )

        llm_resp = self.llm_client.complete(system_prompt, user_prompt)
        state.research_notes = llm_resp.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=f"Gathered {len(sources)} sources and compiled research notes.",
                metadata={
                    "sources_count": len(sources),
                    "tokens": llm_resp.input_tokens + (llm_resp.output_tokens or 0)
                    if llm_resp.input_tokens
                    else 0,
                    "cost": llm_resp.cost_usd,
                },
            )
        )
        state.add_trace_event(
            "researcher_completed",
            {"sources_count": len(sources), "cost_usd": llm_resp.cost_usd},
        )
        logger.info(f"[Researcher] Gathered {len(sources)} sources.")
        return state
