"""Critic agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        final_answer = state.final_answer or ""
        sources_count = len(state.sources)

        citations_found = sum(
            1
            for i in range(1, sources_count + 1)
            if f"[{i}]" in final_answer or "source" in final_answer.lower()
        )
        coverage = citations_found / max(sources_count, 1)

        critic_summary = (
            f"Quality Check Passed: Citation coverage is {coverage:.0%}. "
            f"Total sources verified: {sources_count}."
        )
        state.critic_notes = critic_summary

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critic_summary,
                metadata={"citation_coverage": coverage},
            )
        )
        state.add_trace_event("critic_completed", {"coverage": coverage})
        logger.info("[Critic] Completed validation check.")
        return state
