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
        System prompt áp dụng cấu trúc 5 phần (Identity, Capabilities, Instructions, Constraints, Output format)
        giúp Agent hoạt động ổn định và bám sát định dạng ReAct.
        """
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        
        return f"""
        1. Identity: "You are a helpful and precise E-commerce Assistant."
        
        2. Capabilities: "Tools available to you:
        {tool_descriptions}"
        
        3. Instructions: "Break goals into sub-tasks. Think step-by-step. Always use tools to fetch real data (stock, discount, shipping). Stop khi đã thu thập đủ evidence."
        
        4. Constraints: "Max {self.max_steps} tool calls. Never invent or hallucinate product details, prices, or stock. Nếu không tìm thấy thông tin, hãy nói rõ."
        
        5. Output format: "You must strictly follow this exact ReAct format for your output:
        Thought: your line of reasoning about what to do next.
        Action: tool_name(arguments)
        Observation: result of the tool call (Wait for user/system to provide this).
        ... (repeat Thought/Action/Observation if needed)
        Final Answer: your final comprehensive response to the user."
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

        while steps < self.max_steps:
            # Generate LLM response
            result = self.llm.generate(current_prompt, system_prompt=system_prompt)
            content = result.get("content", "")

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

                # Pass back the loop
                current_prompt += f"\n\n{content}\nObservation: {obs}\n"
            else:
                # LLM bị ảo giác, không tuân thủ định dạng
                logger.log_event("AGENT_ERROR_FORMAT", {"content": content})
                step_entry["error"] = "Format error – missing Action or Final Answer"
                current_prompt += f"\n\n{content}\nObservation: ERROR - Bạn đã quên định dạng (Action: tool_name(args)) hoặc quên gọi Final Answer:. Vui lòng tuân thủ chặt chẽ định dạng.\n"

            self.trace.append(step_entry)

            steps += 1
            
        logger.log_event("AGENT_END", {"steps": steps})

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Helper method to execute tools by name.
        """
        try:
            # Local import dể tránh rác context. Khởi tạo kết nối toolkit.
            from src.tools.ecommerce_tools import check_stock, get_discount, calc_shipping
            
            if tool_name == "check_stock":
                # args thường có dạng 'iphone' hoặc "iphone"
                clean_arg = args.strip("'\" ")
                res = check_stock(clean_arg)
                return str(res)
                
            elif tool_name == "get_discount":
                clean_arg = args.strip("'\" ")
                res = get_discount(clean_arg)
                return str(res)
                
            elif tool_name == "calc_shipping":
                # args ví dụ: 2.5, 'Hanoi'
                split_args = [a.strip("'\" ") for a in args.split(",")]
                if len(split_args) < 2:
                    return "ERROR: calc_shipping cần đúng 2 tham số: weight, destination"
                weight = float(split_args[0])
                dest = split_args[1]
                res = calc_shipping(weight, dest)
                return str(res)
                
            else:
                return f"ERROR: Tool '{tool_name}' không tồn tại. Hãy chắc chắn bạn gọi tool có trong list."
        except Exception as e:
            return f"ERROR executing tool '{tool_name}': {e}"
