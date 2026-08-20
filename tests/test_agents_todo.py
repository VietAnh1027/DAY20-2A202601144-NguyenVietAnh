"""Unit tests for agents and workflow routing."""

from multi_agent_research_lab.agents import (
    SupervisorAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


def test_supervisor_routing_sequence() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    supervisor = SupervisorAgent()

    # Step 1: Initial state -> Route to researcher
    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    # Populate research notes -> Route to analyst
    state.research_notes = "Notes"
    state.sources = []
    state = supervisor.run(state)
    assert state.route_history[-1] == "researcher"  # needs sources too

    state.sources = [{"title": "T", "url": "U", "snippet": "S"}]
    state = supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    # Populate analysis notes -> Route to writer
    state.analysis_notes = "Analysis"
    state = supervisor.run(state)
    assert state.route_history[-1] == "writer"

    # Populate final answer -> Route to critic
    state.final_answer = "Final answer [1]"
    state = supervisor.run(state)
    assert state.route_history[-1] == "critic"

    # Populate critic notes -> Route to done
    state.critic_notes = "Passed"
    state = supervisor.run(state)
    assert state.route_history[-1] == "done"
    assert state.is_completed is True


def test_supervisor_max_iterations_guardrail() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.iteration = 10  # Exceeds max_iterations (default 6)
    supervisor = SupervisorAgent()

    state = supervisor.run(state)
    assert state.route_history[-1] == "done"
    assert state.is_completed is True


def test_workflow_end_to_end() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain GraphRAG state of the art"))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)

    assert result.is_completed is True
    assert len(result.sources) > 0
    assert result.research_notes is not None
    assert result.analysis_notes is not None
    assert result.final_answer is not None
    assert result.critic_notes is not None
