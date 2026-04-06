import time
from typing import Dict, Any, List
from src.telemetry.logger import logger

# Pricing per 1M tokens (input, output) in USD
# Source: official provider pricing pages
MODEL_PRICING = {
    # OpenAI
    "gpt-4o":                  {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":             {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":             {"input": 10.00, "output": 30.00},
    "gpt-4":                   {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo":           {"input": 0.50,  "output": 1.50},
    # Google Gemini
    "gemini-1.5-flash":        {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro":          {"input": 3.50,  "output": 10.50},
    "gemini-2.0-flash":        {"input": 0.10,  "output": 0.40},
    "gemini-2.0-flash-lite":   {"input": 0.075, "output": 0.30},
}

DEFAULT_PRICING = {"input": 0.50, "output": 1.50}  # fallback


class PerformanceTracker:
    """
    Tracking industry-standard metrics for LLMs.
    """
    def __init__(self):
        self.session_metrics = []

    def track_request(self, provider: str, model: str, usage: Dict[str, int], latency_ms: int):
        """
        Logs a single request metric to our telemetry.
        """
        cost = self._calculate_cost(model, usage)
        metric = {
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost_usd": cost,
        }
        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> float:
        """
        Calculates real cost in USD based on model pricing (per 1M tokens).
        """
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        input_cost  = usage.get("prompt_tokens", 0)     / 1_000_000 * pricing["input"]
        output_cost = usage.get("completion_tokens", 0) / 1_000_000 * pricing["output"]
        return round(input_cost + output_cost, 6)

    def get_session_summary(self) -> Dict[str, Any]:
        """Returns aggregated token and cost totals for the current session."""
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

    def reset_session(self):
        """Clears session metrics (call before each test case to get per-run stats)."""
        self.session_metrics = []


# Global tracker instance
tracker = PerformanceTracker()
