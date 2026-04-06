# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Tạ Vĩnh Phúc
- **Student ID**: 2A202600424
- **Date**: 2026-04-06

---

## I. Technical Contribution (15 Points)

*Describe your specific contribution to the codebase (e.g., implemented a specific tool, fixed the parser, etc.).*

- **Modules Implementated**: `src/tools/ecommerce_tools.py`, `src/agent/agent.py`, system prompt.
- **Code Highlights**: 
  - Khai báo thêm tool tính tiền: `def calc_total_price(price: float, quantity: int) -> str:`
  - Đăng ký tool mới vào mảng `ECOMMERCE_TOOLS_SPEC` để Agent nhận diện.
  - Implement block parse và gọi tool ở hàm `_execute_tool` trong class `ReActAgent`.
- **Documentation**: Hàm `calc_total_price` kết hợp với Action block của mô hình để tính tổng số tiền thay vì để LLM tự tính toán và dễ dẫn đến hallucination. Tool này return về một string tổng tiền để prompt vào `Observation`.

---

## II. Debugging Case Study (10 Points)

*Analyze a specific failure event you encountered during the lab using the logging system.*

- **Problem Description**: Agent bị kẹt trong vòng lặp vô tận do liên tục đưa ra output bị thiếu prefix `Action:` khi mô hình nhầm lẫn giữa định dạng format trả lời trực tiếp thay vì bám đúng Thought-Action-Observation.
- **Log Source**: File `logs/agent_telemetry.log` (Ví dụ log: `{"event": "AGENT_ERROR_FORMAT", "data": {"content": "Thought: Now that I know the iPhone is in stock... Action: None required at this moment."}}`)
- **Diagnosis**: Lý do là System Prompt chưa đưa ra pattern strict hoặc bộ dữ liệu Few-shot chưa đủ nhấn mạnh bước dừng `Final Answer:`. Do đó LLM tự ý sinh ra `Action: None required...` nhưng Parser của agent quy định bắt buộc phải chộp được pattern cụ thể bằng RegEx.
- **Solution**: 
  - Khắc phục 1: Viết lại System Prompt nhấn mạnh phần 5 (output format), buộc LLM nếu muốn ngừng dùng tool thì phải xài `Final Answer:`.
  - Khắc phục 2: Thay vì chỉ vứt lỗi đi, append thẳng dòng nhắc nhở vào observation để đưa lại cho LLM từ đó nó tỉnh và feedback lại ở bước tới thành `Final Answer:`.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

*Reflect on the reasoning capability difference.*

1.  **Reasoning**: ReAct Framework nhờ có khối `Thought` mà Agent không phải nhắm mắt trả lời bừa như Chatbot thông thường. Cụ thể, khi chatbot được hỏi "Hàng Macbook còn không? Nếu còn tính tổng tiền...", nó sẽ tự bịa (ảo giác) ra giá, cân nặng và phí ship. Nhờ ReAct, Agent tự nhận thức được việc nó phải "lấy tồn kho trước", sau đó "kiểm tra mã giảm hía" và tiếp tới mới tự gọi các Tool `calc_shipping` hay `calc_total_price` một cách minh bạch, từng bước một.
2.  **Reliability**: Điểm yếu của ReAct so với Chatbot thông thường là độ ổn định trong cú pháp trả về (rất dễ bị format error nếu model yếu như Phi-3) và thời gian phản hồi (latency). Trong những câu hỏi quá đơn giản ("Hello, hơ are you?"), Agent thường cố tìm cách gọi Tool thay vì cứ chào lại bình thường, đôi khi dẫn đến quá tải các bước lặp không cần thiết.
3.  **Observation**: Observation (kết quả thực thi tool) đóng vai trò là "con mắt" của Agent cập nhật vào bộ nhớ ngay lập tức. Nếu Tool trả về lỗi (ví dụ thiếu params), chữ "ERROR" lập tức xuất hiện trong Observation, buộc LLM phải tự soi lại Thought bước kế tiếp và gọi lại Tool với argument đúng định dạng. Khác hẳn với việc chatbot cũ phải xin lỗi và kêu người dùng cung cấp lại từ đầu.

---

## IV. Future Improvements (5 Points)

*How would you scale this for a production-level AI agent system?*

- **Scalability**: Xây dựng kiến trúc Asynchronous Tool Execution bằng Celery hoặc Kafka để tránh việc Agent bị block khi một tool tra cứu DB nội bộ hoặc API bên ngoài bị nghẽn gây latency cao.
- **Safety**: Bổ sung một mô hình "Supervisor LLM" (hoặc một rule-based hệ thống chặn) có nhiệm vụ duyệt trước câu trả lời (Final Answer) hoặc tham số Tool (Action args) của Agent chính trước khi gửi ra ngoài. Giúp ngăn chặn các hành vi prompt injection làm hỏng tham số nội bộ.
- **Performance**: Khi số lượng tools phình to lên hàng trăm, Agent không thể nhét hết specs vào System Prompt vì tràn context window. Cải thiện bằng cách dùng Vector DB kết hợp RAG để truy xuất chỉ 3-5 công cụ thực sự cần thiết dựa trên query của user ở mỗi lượt hội thoại.

---

> [!NOTE]
> Submit this report by renaming it to `REPORT_[YOUR_NAME].md` and placing it in this folder.
