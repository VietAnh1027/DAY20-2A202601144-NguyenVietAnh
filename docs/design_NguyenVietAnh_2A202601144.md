# Multi-Agent Research System Design Specification

## Problem

Hệ thống cần giải quyết các bài toán tìm kiếm và tổng hợp thông tin phức tạp (ví dụ: nghiên cứu công nghệ GraphRAG state-of-the-art, phân tích hệ thống đa tác vụ multi-agent). Hệ thống phải tự động thu thập tài liệu tham khảo đáng tin cậy, phân tích luận điểm kỹ thuật, và tổng hợp thành báo cáo hoàn chỉnh kèm trích dẫn nguồn chuẩn (inline citations `[1]`, `[2]`).

## Why multi-agent?

Mô hình **Single-Agent** khi gặp các câu hỏi dài và phức tạp thường mắc phải các hạn chế nghiêm trọng:
1. **Context Overload & Hallucination**: Một agent duy nhất vừa thu thập thông tin, vừa phân tích và viết bài dễ bỏ sót thông tin hoặc bịa đặt (hallucination) khi không trích dẫn rõ nguồn.
2. **Thiếu tính chuyên biệt hóa (Separation of Concerns)**: Mỗi nhiệm vụ (thu thập tài liệu, phân tích luận điểm, viết báo cáo, kiểm định chất lượng) đòi hỏi system prompt và chiến lược suy luận khác nhau.
3. **Multi-Agent Solution**: Chia nhỏ hệ thống thành các agent chuyên biệt (Supervisor, Researcher, Analyst, Writer, Critic) phối hợp qua **Shared State** giúp tăng tính chính xác thực tế (factual grounding), kiểm soát lỗi rõ ràng và cải thiện citation coverage lên 100%.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| **Supervisor** | Điều phối luồng làm việc, quyết định agent tiếp theo và điểm dừng. | `ResearchState` (kiểm tra `research_notes`, `analysis_notes`, `final_answer`) | `route_history` (tên agent kế tiếp hoặc `"done"`) | Lặp vô hạn nếu không kiểm soát state -> Khắc phục bằng guardrail `max_iterations = 6`. |
| **Researcher** | Tìm kiếm tài liệu từ web/corpus và tóm tắt thông tin thô. | `request.query`, `request.max_sources` | `sources` (`SourceDocument`), `research_notes` | Lỗi API/mạng -> Khắc phục bằng Tavily API retry & fallback local corpus. |
| **Analyst** | Phân tích các ghi chú thô, rút ra các luận điểm chính, so sánh giải pháp. | `research_notes`, `sources` | `analysis_notes` | Phân tích chung chung -> Khắc phục bằng prompt định hướng cấu trúc (insights, trade-offs, failure modes). |
| **Writer** | Tổng hợp báo cáo Markdown hoàn chỉnh kèm inline citations `[1]`, `[2]`. | `analysis_notes`, `sources` | `final_answer` | Quên trích dẫn nguồn -> Khắc phục bằng tự động append danh mục tài liệu tham khảo nếu thiếu. |
| **Critic** | Kiểm định độ bao phủ trích dẫn (citation coverage) và tính nhất quán. | `final_answer`, `sources` | `critic_notes`, `citation_coverage` metric | Phát hiện thiếu trích dẫn -> Cảnh báo và đánh giá chỉ số coverage. |

## Shared state

Tất cả các agent truyền thông tin thông qua đối tượng duy nhất `ResearchState` (`Pydantic BaseModel`):
- `request` (`ResearchQuery`): Chứa câu hỏi nghiên cứu, số lượng nguồn tối đa, đối tượng đọc.
- `iteration` (`int`): Số lần lặp hiện tại của workflow.
- `route_history` (`list[str]`): Lịch sử chuyển tiếp giữa các agent để theo dõi và debug trace.
- `sources` (`list[SourceDocument]`): Danh sách các tài liệu trích dẫn (Title, URL, Snippet).
- `research_notes` (`str | None`): Ghi chú thô do Researcher thu thập.
- `analysis_notes` (`str | None`): Phân tích chiều sâu do Analyst tổng hợp.
- `final_answer` (`str | None`): Báo cáo nghiên cứu cuối cùng do Writer tạo ra.
- `critic_notes` (`str | None`): Kết quả kiểm định chất lượng từ Critic.
- `is_completed` (`bool`): Trạng thái hoàn thành workflow.
- `trace` & `agent_results`: Nhật ký sự kiện và metadata token/cost cho observability.

## Routing policy

Workflow hoạt động theo dạng State Machine (vòng lặp có điều kiện):

```text
               +-------------------+
               |    User Query     |
               +---------+---------+
                         |
                         v
             +-----------+-----------+
    +------->|   Supervisor / Router |<-------+
    |        +-----------+-----------+        |
    |                    |                    |
    |      +-------------+-------------+      |
    |      |             |             |      |
    |      v             v             v      |
    |  Researcher     Analyst       Writer    |
    |      |             |             |      |
    +------+-------------+-------------+------+
                         |
                         v (khi hoàn thành hoặc max_iterations)
                      Critic
                         |
                         v
                    End (Done)
```

**Quy tắc chuyển trạng thái trong Supervisor:**
1. `iteration >= max_iterations` hoặc `is_completed == True` -> Route: `"done"`.
2. Chưa có `sources` hoặc `research_notes` -> Route: `"researcher"`.
3. Chưa có `analysis_notes` -> Route: `"analyst"`.
4. Chưa có `final_answer` -> Route: `"writer"`.
5. Chưa có `critic_notes` -> Route: `"critic"`.
6. Đã đầy đủ -> Route: `"done"`.

## Guardrails

- **Max iterations**: Giới hạn tối đa 6 vòng lặp (`max_iterations = 6`) trong `Settings` để chống chạy vô tận.
- **Timeout**: Cấu hình `timeout_seconds = 60` cho từng cuộc gọi LLM API.
- **Retry**: Tự động thử lại khi API gặp sự cố tạm thời.
- **Fallback**:
  - LLM Fallback: Nếu không có `OPENAI_API_KEY`, tự động chuyển sang mô hình Mock LLM Engine nội bộ.
  - Search Fallback: Nếu lỗi kết nối HTTPS/SSL hoặc thiếu `TAVILY_API_KEY`, tự động chuyển sang Offline Search Corpus.
- **Validation**:
  - Validation dữ liệu đầu vào bằng `ResearchQuery` Pydantic schema (min_length=5).
  - Validation chất lượng đầu ra bằng Critic agent (tính toán `citation_coverage`).

## Benchmark plan

- **Test Queries**:
  - `Query 1`: "Research GraphRAG state-of-the-art and write a 500-word summary"
  - `Query 2`: "Explain multi-agent architecture vs single-agent baseline"
- **Metrics**:
  - `Latency`: Wall-clock execution time (seconds).
  - `Estimated Cost`: Chi phí tính dựa trên số prompt & completion tokens (USD).
  - `Quality Score`: Thang điểm 0-10 đánh giá độ đầy đủ và cấu trúc báo cáo.
  - `Citation Coverage`: Tỷ lệ phần trăm nguồn tài liệu được dẫn chứng `[1]`, `[2]`.
  - `Failure Rate`: Tỷ lệ lỗi nảy sinh trong quá trình chạy.
- **Expected Outcome**: Multi-Agent Workflow đạt `Quality Score >= 9.0/10` và `Citation Coverage = 100%`, vượt trội so với Single-Agent Baseline.

