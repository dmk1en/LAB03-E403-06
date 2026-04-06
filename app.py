"""
Streamlit frontend – Chatbot (left) vs ReAct Agent (right), side-by-side.
Single shared input sends the same message to both simultaneously.

Run:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Lab 3 – Compare",
    page_icon="🤖",
    layout="wide",
)

SYSTEM_PROMPT = (
    "You are a helpful AI chatbot. "
    "Answer clearly and concisely. "
    "If you are uncertain, say so instead of making up facts."
)

# ── helpers ───────────────────────────────────────────────────────────────────

def get_provider(provider_name: str, model_name: str):
    if provider_name == "OpenAI":
        from src.core.openai_provider import OpenAIProvider
        return OpenAIProvider(model_name=model_name, api_key=os.getenv("OPENAI_API_KEY", ""))
    from src.core.gemini_provider import GeminiProvider
    return GeminiProvider(model_name=model_name, api_key=os.getenv("GEMINI_API_KEY", ""))


def format_cost(c: float) -> str:
    return f"${c:.6f}"


def stats_line(usage: dict, cost: float, latency_ms: int) -> str:
    return (
        f"🔢 {usage.get('prompt_tokens',0):,} → {usage.get('completion_tokens',0):,} "
        f"(total {usage.get('total_tokens',0):,}) · 💰 {format_cost(cost)} · ⏱ {latency_ms:,} ms"
    )


def render_trace(trace: list, expanded: bool = False):
    if not trace:
        return
    with st.expander("🔍 ReAct Trace", expanded=expanded):
        for step in trace:
            st.markdown(f"**Step {step['step']}**")
            st.code(step["thought"], language="text")
            if "action" in step:
                st.markdown(f"**Action:** `{step['action']}`")
                st.markdown(f"**Observation:** {step['observation']}")
            if "final_answer" in step:
                st.success(f"Final Answer: {step['final_answer']}")
            if "error" in step:
                st.warning(step["error"])
            st.divider()


def accumulate(key: str, usage: dict, cost: float):
    s = st.session_state.totals[key]
    s["prompt"]     += usage.get("prompt_tokens", 0)
    s["completion"] += usage.get("completion_tokens", 0)
    s["total"]      += usage.get("total_tokens", 0)
    s["cost"]       += cost


# ── session state init ────────────────────────────────────────────────────────
for _k in ("chat_msgs", "agent_msgs"):
    if _k not in st.session_state:
        st.session_state[_k] = []

if "totals" not in st.session_state:
    st.session_state.totals = {
        "chat":  {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0},
        "agent": {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0},
    }

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    provider_name = st.selectbox("Provider", ["OpenAI", "Gemini"])
    model_options = {
        "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    }
    model_name = st.selectbox("Model", model_options[provider_name])

    st.divider()

    if st.button("🗑️ Clear all"):
        st.session_state.chat_msgs  = []
        st.session_state.agent_msgs = []
        st.session_state.totals = {
            "chat":  {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0},
            "agent": {"prompt": 0, "completion": 0, "total": 0, "cost": 0.0},
        }
        st.rerun()

    st.divider()
    for label, key in [("💬 Chatbot", "chat"), ("🤖 ReAct Agent", "agent")]:
        st.markdown(f"**{label}**")
        t = st.session_state.totals[key]
        c1, c2 = st.columns(2)
        c1.metric("Tokens", f"{t['total']:,}")
        c2.metric("Cost",   format_cost(t["cost"]))
        st.caption(f"Prompt {t['prompt']:,} · Output {t['completion']:,}")
        st.divider()

# ── page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align:center;margin-bottom:0'>🤖 AI Lab 3 — Side-by-Side Comparison</h2>"
    f"<p style='text-align:center;color:grey'>{provider_name} · {model_name}</p>",
    unsafe_allow_html=True,
)

# ── two-column layout ─────────────────────────────────────────────────────────
col_chat, col_divider, col_agent = st.columns([10, 1, 10])

# helper: render a column's message history
def render_history(msgs: list):
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "stats" in msg:
                st.caption(msg["stats"])
            if msg.get("trace"):
                render_trace(msg["trace"], expanded=False)

with col_chat:
    st.subheader("💬 Chatbot")
    chat_container = st.container(height=520, border=False)
    with chat_container:
        render_history(st.session_state.chat_msgs)

with col_divider:
    st.html(
        "<div style='border-left:2px solid #e0e0e0;height:600px;"
        "margin:auto;width:0;margin-top:40px'></div>"
    )

with col_agent:
    st.subheader("🤖 ReAct Agent")
    agent_container = st.container(height=520, border=False)
    with agent_container:
        render_history(st.session_state.agent_msgs)

# ── shared input ──────────────────────────────────────────────────────────────
prompt = st.chat_input("Send to both — Chatbot & ReAct Agent…")

if prompt:
    from src.telemetry.metrics import tracker
    from src.tools.ecommerce_tools import ECOMMERCE_TOOLS_SPEC
    from src.agent.agent import ReActAgent

    # append user bubble immediately
    st.session_state.chat_msgs.append({"role": "user", "content": prompt})
    st.session_state.agent_msgs.append({"role": "user", "content": prompt})

    # ── run chatbot ────────────────────────────────────────────────────────────
    chat_answer = chat_stats = ""
    try:
        tracker.reset_session()
        llm = get_provider(provider_name, model_name)
        result = llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        tracker.track_request(
            provider=result.get("provider", provider_name),
            model=llm.model_name,
            usage=result.get("usage", {}),
            latency_ms=result.get("latency_ms", 0),
        )
        chat_answer = result.get("content", "")
        s = tracker.get_session_summary()
        usage = {k: s.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        cost  = s.get("total_cost_usd", 0.0)
        chat_stats = stats_line(usage, cost, s.get("total_latency_ms", 0))
        accumulate("chat", usage, cost)
    except Exception as e:
        chat_answer = f"❌ Error: {e}"

    # ── run agent ──────────────────────────────────────────────────────────────
    agent_answer = agent_stats = ""
    agent_trace: list = []
    try:
        tracker.reset_session()
        llm = get_provider(provider_name, model_name)
        agent = ReActAgent(llm=llm, tools=ECOMMERCE_TOOLS_SPEC, max_steps=7)
        agent_answer = agent.run(prompt)
        agent_trace  = agent.trace
        s = tracker.get_session_summary()
        usage = {k: s.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        cost  = s.get("total_cost_usd", 0.0)
        agent_stats = stats_line(usage, cost, s.get("total_latency_ms", 0))
        accumulate("agent", usage, cost)
    except Exception as e:
        agent_answer = f"❌ Error: {e}"

    # ── persist to history ────────────────────────────────────────────────────
    st.session_state.chat_msgs.append({
        "role": "assistant", "content": chat_answer, "stats": chat_stats, "trace": [],
    })
    st.session_state.agent_msgs.append({
        "role": "assistant", "content": agent_answer, "stats": agent_stats, "trace": agent_trace,
    })

    st.rerun()
