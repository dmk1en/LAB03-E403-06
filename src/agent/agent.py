import os
import re
from typing import List, Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker

class ReActAgent:
    """
    SKELETON: A ReAct-style Agent that follows the Thought-Action-Observation loop.
    Students should implement the core loop logic and tool execution.
    """
    
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history = []
        self.trace: List[Dict[str, Any]] = []  # populated during run()

    def get_system_prompt(self) -> str:
        """
        System prompt tiếng Việt theo cấu trúc 5 phần (Identity, Capabilities, Instructions, Constraints, Output format)
        giúp Agent hoạt động ổn định và bám sát định dạng ReAct.
        """
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        
        return f"""
        1. Identity: "Bạn là một trợ lý thương mại điện tử (E-commerce Assistant) chính xác và hữu ích."

        2. Capabilities: "Các công cụ bạn có thể sử dụng:
        {tool_descriptions}"

        3. Instructions: "Hãy chia mục tiêu thành các bước nhỏ. Suy nghĩ từng bước một cách logic.
        Luôn sử dụng tool để lấy dữ liệu thực (tồn kho, giảm giá, vận chuyển).
        Dừng lại khi đã thu thập đủ thông tin cần thiết."

        4. Constraints: "Tối đa {self.max_steps} lần gọi tool.
        Không được tự bịa (hallucinate) thông tin về sản phẩm, giá hoặc tồn kho.
        Nếu không tìm thấy thông tin, hãy nói rõ ràng với người dùng."

        5. Output format: "Bạn PHẢI tuân thủ CHÍNH XÁC định dạng ReAct sau:

        Thought: suy nghĩ của bạn về bước tiếp theo.
        Action: tool_name(arguments)
        Observation: kết quả từ tool (chờ hệ thống cung cấp).
        ... (lặp lại Thought/Action/Observation nếu cần)
        Final Answer: câu trả lời cuối cùng đầy đủ cho người dùng."
        """

    def run(self, user_input: str) -> str:
        """
        TODO: Implement the ReAct loop logic.
        1. Generate Thought + Action.
        2. Parse Action and execute Tool.
        3. Append Observation to prompt and repeat until Final Answer.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})
        self.trace = []

        system_prompt = self.get_system_prompt()
        current_prompt = user_input
        steps = 0
        consecutive_errors = 0

        while steps < self.max_steps:
            # Generate LLM response
            result = self.llm.generate(current_prompt, system_prompt=system_prompt)
            content = result.get("content", "")

            # Tránh Ảo Giác: LLM tự sinh ra Observation thay vì đợi Tool chạy
            if "Observation:" in content:
                content = content.split("Observation:")[0].strip()

            # Track token usage and cost
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )
            
            logger.log_event("AGENT_THOUGHT", {"step": steps, "content": content})
            # Print for visual trace
            print(f"---\n🧠 Bước {steps+1}:\n{content.strip()}")

            step_entry: Dict[str, Any] = {"step": steps + 1, "thought": content.strip()}

            # 1. Parse Final Answer
            if "Final Answer:" in content:
                final_answer_match = re.search(r'Final Answer:\s*(.*)', content, re.DOTALL)
                if final_answer_match:
                    ans = final_answer_match.group(1).strip()
                    step_entry["final_answer"] = ans
                    self.trace.append(step_entry)
                    logger.log_event("AGENT_END", {"steps": steps, "answer": ans})
                    return ans

            # 2. Parse Action and Tool
            # We look for: Action: tool_name(arg1, arg2)
            action_match = re.search(r'Action:\s*([A-Za-z0-9_]+)\((.*?)\)', content)
            if action_match:
                tool_name = action_match.group(1).strip()
                args_str = action_match.group(2).strip()

                logger.log_event("AGENT_ACTION", {"tool": tool_name, "args": args_str})

                # Execute Tool
                obs = self._execute_tool(tool_name, args_str)
                logger.log_event("AGENT_OBSERVATION", {"observation": obs})
                print(f"🔍 Trải nghiệm (Observation): {obs}")

                step_entry["action"] = f"{tool_name}({args_str})"
                step_entry["observation"] = obs

                if obs.startswith("ERROR"):
                    consecutive_errors += 1
                    logger.log_event("AGENT_ERROR_RETRY", {"error": obs, "retry": consecutive_errors})
                    print(f"⚠️ [Tự động sửa lỗi]: Phát hiện lỗi từ Tool. Ép LLM sửa sai (Lần {consecutive_errors}/3)...")
                    
                    if consecutive_errors >= 3:
                        logger.log_event("AGENT_END", {"steps": steps, "status": "FAIL_RETRY"})
                        return f"Thất bại: LLM không tự sửa được lỗi sau 3 lần. Lỗi cuối: {obs}"
                    
                    current_prompt += f"\n\n{content}\nObservation: {obs}\nSYSTEM WARNING: Lỗi khi gọi hàm! Hãy đọc kỹ mô tả, số lượng và kiểu tham số của hàm `{tool_name}` trong System prompt rồi viết lời giải lại cho đúng."
                    self.trace.append(step_entry)
                    continue
                else:
                    consecutive_errors = 0 
                    current_prompt += f"\n\n{content}\nObservation: {obs}\n"
            else:
                consecutive_errors += 1
                logger.log_event("AGENT_ERROR_FORMAT", {"content": content, "retry": consecutive_errors})
                print(f"⚠️ [Tự động sửa lỗi]: Sai vỡ Format ReAct. Ép LLM nắn lại (Lần {consecutive_errors}/3)...")
                
                if consecutive_errors >= 3:
                    logger.log_event("AGENT_END", {"steps": steps, "status": "FAIL_FORMAT"})
                    return "Thất bại: LLM liên tục viết sai Format ReAct 3 lần. Nghi ngờ Ảo giác nặng."

                step_entry["error"] = "Format error – missing Action or Final Answer"
                current_prompt += f"\n\n{content}\nObservation: ERROR - Bạn đã quên ghi định dạng đúng (Action: tool_name(args)) hoặc quên gọi chữ Final Answer:. Xin hãy viết lại định dạng.\n"
                
                self.trace.append(step_entry)
                continue

            self.trace.append(step_entry)

            steps += 1
            
        logger.log_event("AGENT_END", {"steps": steps})

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Helper method to execute tools by name.
        """
        try:
            # Local import dể tránh rác context. Khởi tạo kết nối toolkit.
            from src.tools.ecommerce_tools import check_stock, get_discount, calc_shipping, calc_total_price, convert_currency
            
            # Xử lý args chung để lọc bỏ các tên biến (VD: weight=5.0 -> 5.0)
            raw_args = args.split(",")
            split_args = []
            for a in raw_args:
                a = a.strip()
                if "=" in a:
                    a = a.split("=", 1)[1]
                split_args.append(a.strip("'\" "))

            if tool_name == "check_stock":
                res = check_stock(split_args[0] if split_args else "")
                return str(res)
                
            elif tool_name == "get_discount":
                res = get_discount(split_args[0] if split_args else "")
                return str(res)
                
            elif tool_name == "calc_shipping":
                if len(split_args) < 2:
                    return "ERROR: calc_shipping cần đúng 2 tham số: weight, destination"
                weight = float(split_args[0])
                dest = split_args[1]
                res = calc_shipping(weight, dest)
                return str(res)
                
            elif tool_name == "calc_total_price":
                if len(split_args) < 2:
                    return "ERROR: calc_total_price cần 2 tham số: price, quantity"
                price = float(split_args[0])
                quantity = int(split_args[1])
                res = calc_total_price(price, quantity)
                return str(res)
                
            elif tool_name == "convert_currency":
                if len(split_args) < 3:
                    return "ERROR: convert_currency cần đúng 3 tham số: amount, from_currency, to_currency"
                try:
                    amount = float(split_args[0])
                except ValueError:
                    return "ERROR: Lỗi tham số. Biến `amount` phải là kiểu số Float."
                from_c = split_args[1]
                to_c = split_args[2]
                res = convert_currency(amount, from_c, to_c)
                return str(res)

            else:
                return f"ERROR: Tool '{tool_name}' không tồn tại. Hãy chắc chắn bạn gọi tool có trong list."
        except Exception as e:
            return f"ERROR executing tool '{tool_name}': {e}"
