"""Benchmark evaluation for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def compute_citation_coverage(state: ResearchState) -> float:
    """Calculate percentage of sources referenced in the final answer."""
    if not state.sources or not state.final_answer:
        return 0.0
    text = state.final_answer.lower()
    cited = 0
    for idx, src in enumerate(state.sources, 1):
        if (
            f"[{idx}]" in state.final_answer
            or src.title.lower()[:20] in text
            or (src.url and src.url.lower() in text)
        ):
            cited += 1
    return cited / len(state.sources)


def compute_quality_score(state: ResearchState) -> float:
    """Calculate a 0-10 quality score based on report completeness and structure."""
    if not state.final_answer:
        return 0.0
    score = 5.0
    answer = state.final_answer
    # Length & structure bonus
    if len(answer) > 200:
        score += 1.5
    if "#" in answer:  # markdown headers
        score += 1.5
    if state.sources and len(state.sources) >= 2:
        score += 1.0
    if compute_citation_coverage(state) > 0.5:
        score += 1.0
    return min(score, 10.0)


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, quality score, citation coverage, and error rate."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Sum estimated costs from agent results
    cost = sum(
        res.metadata.get("cost", 0.0) or 0.0
        for res in state.agent_results
        if isinstance(res.metadata, dict)
    )

    quality = compute_quality_score(state)
    coverage = compute_citation_coverage(state)
    failure = 1.0 if state.errors or not state.final_answer else 0.0

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost if cost > 0 else 0.0005,
        quality_score=quality,
        citation_coverage=coverage,
        failure_rate=failure,
        notes=f"Iterations: {state.iteration}, Sources: {len(state.sources)}",
    )
    return state, metrics
