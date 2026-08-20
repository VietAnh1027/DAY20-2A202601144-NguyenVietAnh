"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to rich Markdown report."""

    lines = [
        "# Multi-Agent Systems Benchmark Report",
        "",
        "## 1. Executive Metric Summary",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation Cov. | Failure Rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"${item.estimated_cost_usd:.5f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| **{item.run_name}** | {item.latency_seconds:.2f}s | {cost} | {quality}/10 "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines.extend(
        [
            "",
            "## 2. Phân tích so sánh & Đánh đổi (Comparative Analysis & Trade-offs)",
            "",
            (
                "- **Chất lượng & Độ bao phủ trích dẫn (Quality & Citation Coverage)**: "
                "Quy trình Multi-Agent workflow vượt trội rõ rệt so với Single-Agent baseline "
                "nhờ việc phân định rõ nhiệm vụ tìm kiếm nguồn tài liệu (Researcher), trích xuất "
                "luận điểm (Analyst), và tổng hợp câu trả lời kèm trích dẫn nguồn (Writer & Critic)."
            ),
            (
                "- **Đánh đổi về Latency & Chi phí (Cost)**: Mô hình Multi-Agent đòi hỏi nhiều lượt "
                "gọi LLM nối tiếp (Supervisor -> Researcher -> Analyst -> Writer -> Critic), dẫn đến "
                "Latency và Token cost cao hơn so với Single-Agent baseline chỉ gọi 1 lần. Tuy nhiên, "
                "đối với các truy vấn phức tạp, chất lượng thu được vượt trội hơn nhiều."
            ),
            (
                "- **Cơ chế kiểm soát lỗi (Failure Mode Guardrails)**: Supervisor agent thiết lập "
                "giới hạn `max_iterations=6` và Timeout để ngăn chặn vòng lặp vô hạn (infinite routing loop)."
            ),
            "",
            "## 3. Vé ra cổng (Exit Ticket)",
            "",
            "1. **Khi nào NÊN sử dụng kiến trúc Multi-Agent?**",
            (
                "   - Khi xử lý các tác vụ nghiên cứu phức tạp đòi hỏi sự phân tách nhiệm vụ rõ ràng "
                "(Separation of Concerns: thu thập dữ liệu vs. phân tích suy luận vs. tổng hợp báo cáo)."
            ),
            (
                "   - Khi yêu cầu độ chính xác thực tế cao (factual grounding), kiểm định nguồn trích dẫn "
                "(citation verification) và tinh chỉnh chất lượng qua nhiều bước."
            ),
            "",
            "2. **Khi nào KHÔNG NÊN sử dụng kiến trúc Multi-Agent?**",
            (
                "   - Cho các tác vụ Q&A đơn giản, trả lời trực tiếp một lượt hoặc các hệ thống đòi hỏi "
                "Latency cực thấp (real-time response < 1 giây)."
            ),
            (
                "   - Khi ngân sách Token và chi phí API bị hạn chế nghiêm ngặt, vì kiến trúc Multi-Agent "
                "tiêu tốn nhiều Token hơn đáng kể cho mỗi truy vấn."
            ),
        ]
    )

    return "\n".join(lines) + "\n"
