"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def run_baseline_runner(query_str: str) -> ResearchState:
    """Single-agent baseline runner."""
    request = _parse_query(query_str)
    state = ResearchState(request=request)
    llm = LLMClient()
    resp = llm.complete(
        system_prompt=(
            "You are a single-agent baseline assistant. Answer the user query in a single response."
        ),
        user_prompt=request.query,
    )
    state.final_answer = resp.content
    state.record_route("baseline")
    return state


def run_multi_agent_runner(query_str: str) -> ResearchState:
    """Multi-agent workflow runner."""
    request = _parse_query(query_str)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline implementation."""
    _init()
    state = run_baseline_runner(query)
    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline Response"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    result = run_multi_agent_runner(query)
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Research query for benchmark"),
    ] = "Research GraphRAG state-of-the-art and write a 500-word summary",
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output report file path"),
    ] = "reports/benchmark_report.md",
) -> None:
    """Run single vs multi-agent benchmark and save report."""
    _init()
    console.print(f"[bold green]Running Benchmark for query:[/bold green] {query}")

    _, baseline_metrics = run_benchmark("Single-Agent Baseline", query, run_baseline_runner)
    _, multi_metrics = run_benchmark("Multi-Agent Workflow", query, run_multi_agent_runner)

    report_content = render_markdown_report([baseline_metrics, multi_metrics])

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_content, encoding="utf-8")

    console.print(Panel.fit(report_content, title="Benchmark Report"))
    console.print(f"[bold green]Report saved to:[/bold green] {out_path.resolve()}")


if __name__ == "__main__":
    app()
