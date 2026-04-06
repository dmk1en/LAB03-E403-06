# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Nguyễn Văn Hiếu
- **Student ID**: [Điền mã sinh viên của bạn vào đây]
- **Date**: 2026-04-06

---

## I. Technical Contribution (15 Points)

_Trách nhiệm chính của tôi trong Nhóm là tham gia vào phân hệ Core Engine, cơ chế phân phối công cụ thông minh (Dispatcher) và xây dựng hệ thống giám sát hiệu năng (Telemetry Dashboard) cùng với Prompt Engineering._

- **Modules Implemented**:
  - `src/agent/agent.py` (Thiết kế và lập trình hàm `get_system_prompt` định hình nhân cách và cấu trúc ReAct).
  - `main_agent.py` (Xây dựng Telemetry Dashboard giám sát chi phí và độ trễ).
- **Code Highlights**:
  - Xây dựng thành công block code in ra **Telemetry Dashboard** giúp theo dõi P50/P99 latency, Average tokens và Total cost (USD) cho từng test case cụ thể.
  - Tối ưu hóa **System Prompt** theo chuẩn 5 phần (Identity, Capabilities, Instructions, Constraints, Output format). Việc ràng buộc cứng về `Output format` giúp Agent bám sát chuẩn ReAct và không bị Hallucination (ảo giác).
- **Documentation**: Mối tương quan của luồng code được thể hiện rõ: Hàm `get_system_prompt()` đóng vai trò tạo khung xương và "luật chơi" cho LLM. Nhờ prompt mạnh mẽ, Core Engine và Dispatcher có thể dễ dàng cắt gọt được các chuỗi lệnh `Thought`, `Action` và mapping đúng tới công cụ cần thiết. Trong khi đó, **Telemetry dashboard** đứng ngoài giám sát toàn bộ quá trình giao tiếp này, đánh giá hiệu năng vận hành theo thời gian thực (real-time).

---

## II. Debugging Case Study (10 Points)

- **Problem Description**: Trong quá trình phát triển Core Engine, Agent thường xuyên bị lỗi trả về kết quả tự do giống như Chatbot thông thường thay vì tuân thủ định dạng tuần tự của ReAct (Thought/Action/Observation). Hậu quả là RegEx parser để bóc lọc chuỗi lệnh không bắt được Action, gây ra exception liên tục và đốt sạch số lần Retry của hệ thống.
- **Diagnosis**: 
  - Nguyên nhân cốt lõi nằm ở prompt ban đầu khá lỏng lẻo. Mô hình ngôn ngữ sau khi suy diễn được vài bước thường bị "lạc đề" và bỏ qua cấu trúc yêu cầu.
- **Solution**:
  - Tôi đã trực tiếp tái thiết kế đồ sộ lại hàm `get_system_prompt()`, ép buộc sử dụng cấu trúc 5 thành phần cô đọng. Điểm nhấn là ở block **Constraints** ("Tối đa max_steps... không được bịa thông tin") và **Output format** ("Bạn PHẢI tuân thủ CHÍNH XÁC định dạng ReAct sau..."). 
  - Kể từ khi refactor sang prompt mới này, tình trạng lỗi chuỗi output từ bộ xử lý cốt lõi (Core Engine) giảm về 0, giúp hệ thống Dispatcher gọi hàm chính xác tuyệt đối.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Việc chia nhỏ quá trình tư duy (System prompt ép buộc phải có `Thought`) giúp mô hình hành động có cơ sở và logic. Khác với Chatbot truyền thống thường hay đoán mò kết quả (ảo giác) khi gặp câu hỏi phức tạp về số liệu thống kê, ReAct Agent bắt buộc tự đặt ra kế hoạch giải quyết trước khi thực thi gọi Tool.
2. **Reliability**: ReAct phụ thuộc rất nhiều vào kỹ năng Prompt Engineering. Nếu cung cấp đủ Constraints, kết quả sẽ rất đáng tin cậy. Tuy nhiên điểm yếu là đối với các đoạn "Small talk" (như người dùng chào hỏi, chỉ nói cảm ơn), việc phải ép nhả chuỗi `Action` làm lãng phí Token không đáng có. Đây là điều chức năng Chatbot tự do đang làm tốt hơn nhờ phản hồi tức thì và tự nhiên.
3. **Observation**: Khả năng nhận feedback từ thế giới thực (thông qua các Observation) chính là "đôi mắt" của Agent. Việc tôi bổ sung quy tắc "Luôn sử dụng tool để lấy dữ liệu thực" vào mục `Instructions` giúp Core Engine hoàn toàn dựa vào data được trả về thay vì dữ liệu được huấn luyện cũ kỹ trên LLM.

---

## IV. Future Improvements (5 Points)

- **Telemetry Logging**: Nâng cấp Telemetry Dashboard hiện tại. Thay vì chỉ `print` in thẳng ra Terminal Console, tính năng này có thể được phát triển thêm để ghi log data thẳng vào dạng tệp `.csv` hoặc `.json`, thuận tiện cho việc đưa vào BI Tools để vẽ biểu đồ và phân tích chi phí toàn đợt của bộ Test Cases.
- **Dynamic Prompting**: Trong System Prompt, thay vì nén toàn bộ mô tả Tool (Tool Descriptions) một cách tĩnh, trong tương lai có thể tối ưu bằng Vector Embedding để lập cơ chế Retriever (Lấy tự động) - chỉ nhồi vào prompt những Tool thực sự match với ý định của người dùng, giúp lượng `total_tokens` được giảm đáng kể và tiết kiệm kinh phí hệ thống trong hàm tính Cost.
- **Performance**: Việc kiểm soát độ trễ (latency) P50, P99 có thể được cải thiện thêm bằng một hệ thống Caching Memory. Nếu Dispatcher nhận ra Tool và Tham số tương tự vừa được gọi ở ngay Turn trước, nó có thể nhả kết quả ngay lập tức thay vì request lại Network, qua đó kéo giảm thông số `total_latency_ms`.
