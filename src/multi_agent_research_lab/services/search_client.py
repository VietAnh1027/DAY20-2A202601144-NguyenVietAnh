import logging

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client implementation."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""
        api_key = self.settings.tavily_api_key

        if api_key:
            try:
                from tavily import TavilyClient  # type: ignore

                tavily = TavilyClient(api_key=api_key)
                res = tavily.search(query=query, max_results=max_results)
                results: list[SourceDocument] = []
                for item in res.get("results", []):
                    results.append(
                        SourceDocument(
                            title=item.get("title", "Untitled Document"),
                            url=item.get("url", "https://example.com"),
                            snippet=item.get("content", item.get("snippet", "")),
                        )
                    )
                if results:
                    return results
            except Exception as exc:
                logger.warning(f"Tavily search failed ({exc}). Using mock fallback.")

        # Local mock corpus for offline / testing / keyless usage
        return [
            SourceDocument(
                title=f"[1] GraphRAG Architecture and Retrieval Paradigm: {query[:30]}",
                url="https://arxiv.org/abs/2404.16130",
                snippet=(
                    "GraphRAG combines Knowledge Graphs with RAG to improve "
                    "retrieval accuracy, structured reasoning, and complex "
                    "relationship extraction across large corpora."
                ),
            ),
            SourceDocument(
                title=f"[2] Building Effective Multi-Agent Systems: {query[:30]}",
                url="https://www.anthropic.com/research/building-effective-agents",
                snippet=(
                    "Multi-agent systems partition complex tasks into "
                    "specialized worker roles (Researcher, Analyst, Writer) "
                    "coordinated by a central Supervisor."
                ),
            ),
            SourceDocument(
                title="[3] Benchmark and Evaluation Methods for Agentic Workflows",
                url="https://github.com/langchain-ai/langgraph",
                snippet=(
                    "Evaluation of agent workflows requires tracking wall-clock "
                    "latency, token costs, hallucination rates, and citation "
                    "coverage against single-agent baselines."
                ),
            ),
        ][:max_results]
