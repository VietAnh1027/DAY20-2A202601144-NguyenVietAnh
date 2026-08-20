"""Analyst agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        research_notes = state.research_notes or "No research notes provided."

        system_prompt = (
            "You are a Lead Data & Systems Analyst. Your goal is to take raw research notes, "
            "extract key technical claims, compare perspectives, evaluate evidence strength, "
            "and synthesize structured analysis notes for executive/technical audiences."
        )
        user_prompt = (
            f"Research Notes to Analyze:\n{research_notes}\n\n"
            "Provide structured analysis notes including: Core Architectural Insights, "
            "Strengths & Trade-offs, and Potential Failure Modes."
        )

        llm_resp = self.llm_client.complete(system_prompt, user_prompt)
        state.analysis_notes = llm_resp.content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content="Extracted key technical claims and compiled analysis notes.",
                metadata={"cost": llm_resp.cost_usd},
            )
        )
        state.add_trace_event("analyst_completed", {"cost_usd": llm_resp.cost_usd})
        logger.info("[Analyst] Completed analysis.")
        return state
