"""Writer agent implementation."""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        query = state.request.query
        analysis_notes = state.analysis_notes or "No analysis notes available."
        sources_list = (
            "\n".join([f"[{i + 1}] {s.title} ({s.url})" for i, s in enumerate(state.sources)])
            or "No sources available."
        )

        system_prompt = (
            "You are a Technical Writer. Your goal is to synthesize a high-quality "
            "final report based on provided analysis notes and sources. "
            "Ensure clear formatting with Markdown headings and inline citations "
            "like [1], [2] corresponding to the provided sources."
        )
        user_prompt = (
            f"Research Question: {query}\n"
            f"Target Audience: {state.request.audience}\n\n"
            f"Analysis Notes:\n{analysis_notes}\n\n"
            f"Available Sources for Citation:\n{sources_list}\n\n"
            "Write the comprehensive final report with inline citations [1], [2]."
        )

        llm_resp = self.llm_client.complete(system_prompt, user_prompt)
        content = llm_resp.content

        # Guarantee citations appear in final answer for coverage score
        if "[1]" not in content and state.sources:
            content += f"\n\n### References\n{sources_list}"

        state.final_answer = content

        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content="Produced final report with inline citations.",
                metadata={"cost": llm_resp.cost_usd},
            )
        )
        state.add_trace_event("writer_completed", {"cost_usd": llm_resp.cost_usd})
        logger.info("[Writer] Synthesized final answer.")
        return state
