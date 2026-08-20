"""LangGraph / State-Machine workflow implementation."""

import logging
from typing import Any

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph/state workflow."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def build(self) -> dict[str, Any]:
        """Create a graph map representation of nodes and edges."""
        return {
            "nodes": {
                "supervisor": self.supervisor,
                "researcher": self.researcher,
                "analyst": self.analyst,
                "writer": self.writer,
                "critic": self.critic,
            },
            "stop_condition": "done",
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the multi-agent workflow loop."""
        with trace_span("multi_agent_workflow", {"query": state.request.query}) as span:
            max_iters = self.settings.max_iterations

            while not state.is_completed and state.iteration < max_iters:
                state = self.supervisor.run(state)
                if not state.route_history:
                    break
                next_node = state.route_history[-1]

                if next_node == "done":
                    state.is_completed = True
                    break
                elif next_node == "researcher":
                    state = self.researcher.run(state)
                elif next_node == "analyst":
                    state = self.analyst.run(state)
                elif next_node == "writer":
                    state = self.writer.run(state)
                elif next_node == "critic":
                    state = self.critic.run(state)
                else:
                    logger.warning(f"Unknown route '{next_node}'. Terminating workflow.")
                    state.is_completed = True
                    break

            span["iterations"] = state.iteration
            span["final_answer_length"] = len(state.final_answer or "")
            logger.info(f"Workflow completed in {state.iteration} iterations.")
            return state
