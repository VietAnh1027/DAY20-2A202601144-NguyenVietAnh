"""Supervisor / router implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Inspect state and determine next route."""
        # Guardrail: Check max_iterations limit to prevent infinite loops
        if state.iteration >= self.settings.max_iterations:
            next_route = "done"
            state.is_completed = True
        elif not state.research_notes or not state.sources:
            next_route = "researcher"
        elif not state.analysis_notes:
            next_route = "analyst"
        elif not state.final_answer:
            next_route = "writer"
        elif not state.critic_notes:
            next_route = "critic"
        else:
            next_route = "done"
            state.is_completed = True

        state.record_route(next_route)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Routed next step to: {next_route} (Iteration: {state.iteration})",
                metadata={"next_route": next_route, "iteration": state.iteration},
            )
        )
        state.add_trace_event(
            "supervisor_route", {"next_route": next_route, "iteration": state.iteration}
        )
        logger.info(f"[Supervisor] Iteration {state.iteration} -> Route: {next_route}")
        return state
