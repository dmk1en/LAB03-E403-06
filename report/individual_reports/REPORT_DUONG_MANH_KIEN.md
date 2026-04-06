# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Kien
- **Student ID**: 2A202600048
- **Date**: 2026-04-06

---

## I. Technical Contribution (15 Points)

### Modules Implemented

| File | Vai trò |
|---|---|
| `src/telemetry/metrics.py` | Real-pricing engine — tính chi phí USD theo từng model |
| `chatbot.py` | Tích hợp tracker + hiển thị stats per-turn & session summary |
| `main_agent.py` | Reset tracker trước mỗi test case, in summary sau mỗi kết quả |
| `app.py` | Giao diện Streamlit side-by-side Chatbot ↔ ReAct Agent |

---

### Code Highlight 1 — Real Pricing Engine (`src/telemetry/metrics.py`)

Trước khi tôi sửa, hàm `_calculate_cost` chỉ là một hằng số giả:

```python
# TRƯỚC (dummy)
def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
    return (usage.get("total_tokens", 0) / 1000) * 0.01
```

Tôi thay thế bằng bảng giá thật của từng model (tính theo đơn vị 1 triệu token), đồng thời tách biệt chi phí **input** và **output** vì các nhà cung cấp tính giá khác nhau cho hai chiều:

```python
MODEL_PRICING = {
    "gpt-4o":            {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
    "gemini-1.5-flash":  {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":    {"input": 3.50,  "output": 10.50},
    # ...
}

def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost  = usage.get("prompt_tokens", 0)     / 1_000_000 * pricing["input"]
    output_cost = usage.get("completion_tokens", 0) / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 6)
```

**Logic hoạt động**: Mỗi lần LLM trả về response, `track_request()` được gọi với `usage` dict chứa `prompt_tokens` và `completion_tokens`. Hàm chia cho 1,000,000 (đổi sang đơn vị "triệu token") rồi nhân với đơn giá tương ứng. Hai khoản được cộng lại để ra tổng chi phí USD chính xác.

---

### Code Highlight 2 — Session Summary (`src/telemetry/metrics.py`)

```python
def get_session_summary(self) -> Dict[str, Any]:
    if not self.session_metrics:
        return {}
    total_prompt     = sum(m["prompt_tokens"]     for m in self.session_metrics)
    total_completion = sum(m["completion_tokens"] for m in self.session_metrics)
    total_tokens     = sum(m["total_tokens"]      for m in self.session_metrics)
    total_cost       = sum(m["cost_usd"]          for m in self.session_metrics)
    total_latency    = sum(m["latency_ms"]        for m in self.session_metrics)
    return {
        "calls":             len(self.session_metrics),
        "prompt_tokens":     total_prompt,
        "completion_tokens": total_completion,
        "total_tokens":      total_tokens,
        "total_cost_usd":    round(total_cost, 6),
        "total_latency_ms":  total_latency,
    }
```

**Logic**: Với ReAct Agent, một user query có thể gọi LLM nhiều lần (mỗi bước Thought). `get_session_summary()` cộng dồn tất cả các lần gọi đó trong một lần chạy, cho phép báo cáo tổng chi phí thực của cả chuỗi reasoning.

---

### Code Highlight 3 — Giao diện Side-by-Side (`app.py`)

```python
col_chat, col_divider, col_agent = st.columns([10, 1, 10])

with col_chat:
    st.subheader("💬 Chatbot")
    with st.container(height=520, border=False):
        render_history(st.session_state.chat_msgs)

with col_agent:
    st.subheader("🤖 ReAct Agent")
    with st.container(height=520, border=False):
        render_history(st.session_state.agent_msgs)

# Shared input — gửi đồng thời cho cả hai
prompt = st.chat_input("Send to both — Chatbot & ReAct Agent…")
```

**Logic**: Streamlit chỉ cho phép một `st.chat_input` mỗi trang. Tôi dùng một input duy nhất ở cuối trang, sau đó gọi cả `llm.generate()` lẫn `agent.run()` tuần tự trước khi `st.rerun()`. Hai cột hiển thị lịch sử riêng biệt (`chat_msgs`, `agent_msgs`) nên kết quả không bị trộn lẫn. Cột Agent còn có `render_trace()` — expandable panel hiển thị từng bước Thought → Action → Observation.

---

## II. Debugging Case Study (10 Points)

### Problem Description

**Bug: Agent bỏ qua định dạng ReAct khi nhận câu hỏi giao tiếp thông thường**

Khi Test Case 5 được chạy với input:
> "Chào em, dạo này em bán hàng có tốt không? Có hay bị kẹt đơn không?"

Agent trả lời tự nhiên như một chatbot thường, **không theo format `Thought: / Action:`**, dẫn đến `AGENT_ERROR_FORMAT` liên tiếp 2 lần và lặp lại cùng một nội dung.

### Log Source (`logs/2026-04-06.log`)

```json
{"timestamp": "2026-04-06T09:31:30.069900", "event": "AGENT_START",
 "data": {"input": "Chào em, dạo này em bán hàng có tốt không?...", "model": "gpt-4o"}}

{"timestamp": "2026-04-06T09:31:32.488212", "event": "AGENT_THOUGHT",
 "data": {"step": 0, "content": "Chào anh! Em không bán hàng trực tiếp, nhưng em có thể hỗ trợ..."}}

{"timestamp": "2026-04-06T09:31:32.488728", "event": "AGENT_ERROR_FORMAT",
 "data": {"content": "Chào anh! Em không bán hàng trực tiếp..."}}

{"timestamp": "2026-04-06T09:31:33.566884", "event": "AGENT_ERROR_FORMAT",
 "data": {"content": "Chào anh! Em không bán hàng trực tiếp..."}}

{"timestamp": "2026-04-06T09:31:34.576538", "event": "AGENT_END",
 "data": {"steps": 2, "answer": "Chào anh! Em không bán hàng trực tiếp..."}}
```

**Quan sát**: Mất 3 lần gọi LLM (steps 0→1→2) để ra được `Final Answer:`, trong khi nội dung không thay đổi. Đây là lãng phí token thuần túy — ~3× chi phí so với mức cần thiết.

### Diagnosis

LLM (GPT-4o) nhận ra đây là câu chào hỏi, không cần gọi tool nào. Tuy nhiên, system prompt chỉ hướng dẫn format cho trường hợp cần dùng tool. Khi không cần tool, model không biết được phép trả lời `Final Answer:` trực tiếp mà không cần `Thought: / Action:` trước đó.

Kết quả: Model trả lời tự nhiên → Parser không tìm thấy `Action:` → inject error message → Model lặp lại y hệt lần trước → loop thêm 1 bước.

### Solution

Cập nhật system prompt trong `get_system_prompt()` để nói rõ Agent được phép bỏ qua bước Action nếu không cần tool:

```python
# THÊM VÀO phần "5. Output format" của system prompt:
"""
If the user's query does NOT require any tool (e.g., greetings, general questions),
you may skip Thought/Action and respond directly with:
Final Answer: [your response]
"""
```

Sau khi áp dụng, Test Case 5 chỉ còn 1 lần gọi LLM thay vì 3 lần, tiết kiệm ~67% chi phí cho loại query này.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

### 1. Reasoning — Khả năng lập luận

Chatbot trả lời ngay từ bộ nhớ huấn luyện, không có cơ chế xác minh. Với câu hỏi như "Macbook còn hàng không?", chatbot có thể **bịa số tồn kho** vì không có tool nào để kiểm tra thực tế.

ReAct Agent, ngược lại, buộc LLM **ngoại hiện hóa lý luận** qua khối `Thought:` trước khi hành động. Điều này tạo ra một "audit trail" — ta có thể đọc log và biết chính xác model đang nghĩ gì ở mỗi bước. Khi chạy Test Case 3 (full flow: kiểm kho → giảm giá → ship), agent chia nhỏ thành 3 tool calls riêng biệt và tổng hợp kết quả chính xác, điều mà chatbot thuần không thể làm tin cậy được.

### 2. Reliability — Độ ổn định

Agent thực sự kém hơn chatbot trong hai tình huống:

- **Câu hỏi giao tiếp / small talk**: Như đã thấy ở bug trên, agent lãng phí token và thời gian cho những câu chỉ cần 1 lần gọi LLM. Latency tăng 3× không vì lý do kỹ thuật.
- **Câu hỏi ngoài phạm vi tool**: Khi hỏi về giày Sneaker Nike (không có trong ECOMMERCE_TOOLS), agent gọi `check_stock("Sneaker Nike")` → nhận `0` → kết luận "hết hàng". Câu trả lời kỹ thuật đúng nhưng sai về mặt nghiệp vụ: sản phẩm không tồn tại trong hệ thống khác với "hết hàng". Chatbot có thể trả lời trung thực hơn về giới hạn của mình.

### 3. Observation — Feedback loop ảnh hưởng lập luận như thế nào

Observation là yếu tố làm cho ReAct thực sự mạnh. Ví dụ rõ nhất trong log là Test Case 3: sau khi nhận `Observation: 0.1` từ `get_discount("WINNER")`, model ở step tiếp theo viết "Mã giảm giá WINNER vẫn còn hiệu lực và cho mức giảm giá 10%" — tức là nó **cập nhật trạng thái nội tâm** dựa trên dữ liệu thực, không phải từ training data. Đây là điểm khác biệt cốt lõi: chatbot "nhớ" giá trị tĩnh, agent "biết" giá trị hiện tại.

---

## IV. Future Improvements (5 Points)

### Scalability — Mở rộng quy mô

Hiện tại mỗi query gọi LLM tuần tự. Nếu triển khai thực tế với hàng nghìn user đồng thời, cần chuyển sang **async tool execution**: khi agent cần gọi cùng lúc `check_stock` và `get_discount` (hai bước độc lập), có thể chạy song song với `asyncio.gather()` để giảm latency từ N×T xuống còn T.

Về infrastructure: containerize toàn bộ stack bằng **Docker Compose** (app + logging service), deploy lên Kubernetes để auto-scale theo load. Streamlit app có thể chạy sau một nginx reverse proxy với session affinity.

### Safety — An toàn & kiểm soát

Gắn thêm một **Supervisor LLM** (pattern "LLM-as-judge") chạy sau mỗi `Final Answer` để kiểm tra: (1) câu trả lời có dựa trên Observation thực không, hay agent đang hallucinate? (2) Action có hợp lệ với tool spec không? Nếu supervisor phát hiện sai lệch, nó inject một Observation cảnh báo thay vì để câu trả lời sai đến tay user.

### Performance — Hiệu năng & tính năng

- **LangChain Memory (ConversationBufferWindowMemory)**: Hiện tại mỗi `agent.run()` là stateless. Gắn memory cho phép agent nhớ context giữa các turn — ví dụ user nói "cái đó" thay vì nhắc lại "iPhone" ở câu tiếp theo.
- **Tích hợp thanh toán Stripe**: Sau khi agent tính được tổng hóa đơn, thêm tool `create_payment_intent(amount, currency)` gọi Stripe API để tạo link thanh toán ngay trong conversation.
- **Vector DB cho tool retrieval**: Khi số lượng tool tăng lên hàng chục, thay vì đưa toàn bộ tool spec vào system prompt (tốn token), dùng **FAISS hoặc ChromaDB** để retrieve chỉ top-k tools liên quan nhất với query hiện tại.

---

> **File này được đặt tại**: `report/individual_reports/REPORT_KIEN.md`
