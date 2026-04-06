import argparse
import os
import sys
from typing import Optional

from dotenv import load_dotenv

# from src.core.gemini_provider import GeminiProvider
from src.core.openai_provider import OpenAIProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


BASELINE_SYSTEM_PROMPT = (
    "You are a helpful AI chatbot. "
    "Answer clearly and concisely. "
    "If you are uncertain, say so instead of making up facts."
)


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print(fallback)


def create_provider(provider_name: str, model_name: Optional[str] = None):
    provider = provider_name.strip().lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env")
        model = model_name or os.getenv("DEFAULT_MODEL", "gpt-4o")
        return OpenAIProvider(model_name=model, api_key=api_key)

    if provider in {"google", "gemini"}:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")
        model = model_name or os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
        return GeminiProvider(model_name=model, api_key=api_key)

    if provider == "local":
        model_path = os.getenv("LOCAL_MODEL_PATH")
        if not model_path:
            raise ValueError("Missing LOCAL_MODEL_PATH in .env")
        # Lazy import to avoid hard-failing when llama-cpp is not installed.
        from src.core.local_provider import LocalProvider

        return LocalProvider(model_path=model_path)

    raise ValueError("Unsupported provider. Use one of: openai | google | local")


def _print_session_summary():
    summary = tracker.get_session_summary()
    if not summary:
        return
    print("\n" + "="*60)
    print("📊 SESSION SUMMARY")
    print(f"   LLM calls      : {summary['calls']}")
    print(f"   Prompt tokens  : {summary['prompt_tokens']:,}")
    print(f"   Output tokens  : {summary['completion_tokens']:,}")
    print(f"   Total tokens   : {summary['total_tokens']:,}")
    print(f"   Estimated cost : ${summary['total_cost_usd']:.6f} USD")
    print(f"   Total latency  : {summary['total_latency_ms']:,} ms")
    print("="*60 + "\n")


def run_once(user_input: str, provider_name: str, model_name: Optional[str] = None) -> str:
    llm = create_provider(provider_name=provider_name, model_name=model_name)
    result = llm.generate(user_input, system_prompt=BASELINE_SYSTEM_PROMPT)

    tracker.track_request(
        provider=result.get("provider", provider_name),
        model=llm.model_name,
        usage=result.get("usage", {}),
        latency_ms=result.get("latency_ms", 0),
    )
    logger.log_event(
        "CHATBOT_TURN",
        {
            "provider": result.get("provider", provider_name),
            "model": llm.model_name,
            "input": user_input,
            "output": result.get("content", ""),
        },
    )

    usage = result.get("usage", {})
    cost = tracker._calculate_cost(llm.model_name, usage)
    print(
        f"[tokens: prompt={usage.get('prompt_tokens', 0):,} "
        f"output={usage.get('completion_tokens', 0):,} "
        f"total={usage.get('total_tokens', 0):,} | "
        f"cost=${cost:.6f} USD | "
        f"latency={result.get('latency_ms', 0):,}ms]"
    )
    return result.get("content", "")


def interactive_chat(provider_name: str, model_name: Optional[str] = None):
    llm = create_provider(provider_name=provider_name, model_name=model_name)
    print(f"Baseline Chatbot is running with {provider_name} ({llm.model_name}).")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            _print_session_summary()
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            _print_session_summary()
            print("Goodbye.")
            break

        try:
            result = llm.generate(user_input, system_prompt=BASELINE_SYSTEM_PROMPT)
            tracker.track_request(
                provider=result.get("provider", provider_name),
                model=llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )
            logger.log_event(
                "CHATBOT_TURN",
                {
                    "provider": result.get("provider", provider_name),
                    "model": llm.model_name,
                    "input": user_input,
                    "output": result.get("content", ""),
                },
            )
            safe_print(f"Assistant: {result.get('content', '').strip()}\n")
            usage = result.get("usage", {})
            cost = tracker._calculate_cost(llm.model_name, usage)
            print(
                f"[tokens: prompt={usage.get('prompt_tokens', 0):,} "
                f"output={usage.get('completion_tokens', 0):,} "
                f"total={usage.get('total_tokens', 0):,} | "
                f"cost=${cost:.6f} USD | "
                f"latency={result.get('latency_ms', 0):,}ms]\n"
            )
        except Exception as exc:
            logger.log_event("CHATBOT_ERROR", {"error": str(exc)})
            print(f"Error: {exc}\n")


def main():
    # Prevent Unicode crashes on some Windows terminals using legacy encoding.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    load_dotenv()
    parser = argparse.ArgumentParser(description="Baseline chatbot for Lab 3")
    parser.add_argument("--provider", default=os.getenv("DEFAULT_PROVIDER", "openai"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--once", default=None, help="Run one prompt and exit")
    args = parser.parse_args()

    logger.log_event(
        "CHATBOT_START",
        {"provider": args.provider, "model_override": args.model is not None},
    )

    try:
        if args.once is not None:
            output = run_once(args.once, provider_name=args.provider, model_name=args.model)
            safe_print(output)
        else:
            interactive_chat(provider_name=args.provider, model_name=args.model)
    finally:
        logger.log_event("CHATBOT_END", {"provider": args.provider})


if __name__ == "__main__":
    main()
