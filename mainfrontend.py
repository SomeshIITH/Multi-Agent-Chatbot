# multi_mode_app.py
from __future__ import annotations

import json
import re
import uuid
import hashlib
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# =========================
# Import your 3 backends
# =========================

# Memory chatbot
from simpleChatbotBackend import chatbot as memory_bot, retrieve_all_threads as memory_threads
from langchain_core.messages import AIMessage, HumanMessage

# RAG chatbot
from ragChatbotBackend import (
    chatbot as rag_bot,
    ingest_pdf,
    retrieve_all_threads as rag_threads,
    thread_document_metadata,
)
from langchain_core.messages import ToolMessage

# Blog bot (graph app)
from blogChatbotBackend import app as blog_app

# =========================
# App Mode Selector
# =========================

st.set_page_config(page_title="Multi AI Workspace", layout="wide")

mode = st.sidebar.radio(
    "Choose AI Mode",
    [
        "🧠 Memory Chatbot",
        "📚 RAG + Tools Chatbot",
        "✍️ Research Blog Agent",
    ],
)

# -----------------------------
# helper: streaming wrapper for LangGraph apps
# -----------------------------
def try_stream(graph_app, inputs: Dict[str, Any]):
    """
    Stream graph progress if available; else invoke.
    Yields ("updates"/"values"/"final", payload).
    """
    try:
        # prefer "updates" streaming if available
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        # fallback to "values" streaming
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    # final fallback to synchronous invoke
    out = graph_app.invoke(inputs)
    yield ("final", out)

def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    """
    Merge a streaming payload update into current_state.
    Handles payload shapes like {"node_name": {...}} or plain dict updates.
    """
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state

# ============================================================
# ================= MEMORY CHATBOT ==========================
# ============================================================

def run_memory_chat():

    st.title(" Memory Chatbot")

    ss = st.session_state

    if "memory_thread" not in ss:
        ss.memory_thread = str(uuid.uuid4())

    if "memory_history" not in ss:
        ss.memory_history = []

    if "memory_threads" not in ss:
        ss.memory_threads = memory_threads()

    # Sidebar
    st.sidebar.subheader("Memory Threads")

    if st.sidebar.button("New Memory Chat"):
        ss.memory_thread = str(uuid.uuid4())
        ss.memory_history = []
        st.rerun()

    for tid in ss.memory_threads[::-1]:
        if st.sidebar.button(tid, key=f"mem-{tid}"):
            ss.memory_thread = tid
            state = memory_bot.get_state(
                config={"configurable": {"thread_id": tid, "user_id": "user"}}
            )
            msgs = state.values.get("messages", [])
            ss.memory_history = [
                {
                    "role": "assistant" if isinstance(m, AIMessage) else "user",
                    "content": m.content,
                }
                for m in msgs
            ]
            st.rerun()

    # Display chat
    for msg in ss.memory_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Talk to memory bot…")

    if user_input:

        ss.memory_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):

            def stream():
                for chunk, _ in memory_bot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={
                        "configurable": {
                            "thread_id": ss.memory_thread,
                            "user_id": "user",
                        }
                    },
                    stream_mode="messages",
                ):
                    if isinstance(chunk, AIMessage):
                        yield chunk.content

            ai_text = st.write_stream(stream())

        ss.memory_history.append({"role": "assistant", "content": ai_text})

# ============================================================
# ================== RAG CHATBOT ============================
# ============================================================

def run_rag_chat():

    st.title(" RAG + Tools Chatbot")

    ss = st.session_state

    if "rag_thread" not in ss:
        ss.rag_thread = str(uuid.uuid4())

    if "rag_history" not in ss:
        ss.rag_history = []

    if "rag_threads" not in ss:
        ss.rag_threads = rag_threads()

    if "rag_docs" not in ss:
        ss.rag_docs = {}

    thread_docs = ss.rag_docs.setdefault(ss.rag_thread, {})

    # Sidebar
    st.sidebar.subheader("RAG Threads")

    if st.sidebar.button("New RAG Chat"):
        ss.rag_thread = str(uuid.uuid4())
        ss.rag_history = []
        st.rerun()

    uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

    if uploaded:
        data = uploaded.getvalue()
        h = hashlib.md5(data).hexdigest()

        if h not in thread_docs:
            with st.sidebar.status("Indexing PDF…"):
                try:
                    summary = ingest_pdf(data, ss.rag_thread, uploaded.name)
                    thread_docs[h] = summary
                except Exception as e:
                    st.sidebar.error(f"PDF ingestion failed: {e}")

    # Threads
    for tid in ss.rag_threads[::-1]:
        if st.sidebar.button(tid, key=f"rag-{tid}"):
            ss.rag_thread = tid
            ss.rag_history = []
            st.rerun()

    # Display chat
    for msg in ss.rag_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask RAG bot…")

    if user_input:

        ss.rag_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):

            status_box = None

            def stream():
                nonlocal status_box
                for chunk, _ in rag_bot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": ss.rag_thread}},
                    stream_mode="messages",
                ):

                    if isinstance(chunk, ToolMessage):
                        if status_box is None:
                            status_box = st.status("Using tool…")
                        else:
                            # update to show running
                            status_box.update(label=f"🔧 Tool running…", state="running")

                    if isinstance(chunk, AIMessage):
                        yield chunk.content

            ai_text = st.write_stream(stream())

            if status_box:
                status_box.update(label="Done", state="complete")

        ss.rag_history.append({"role": "assistant", "content": ai_text})


# ============================================================
# ================= BLOG AGENT ==============================
# ============================================================
def run_blog_agent():
    """
    Streaming-capable blog runner that integrates with blogChatbotBackend.app.
    Only this function was changed — memory and rag handlers are preserved as you provided.
    """

    st.title(" Research Blog Agent")

    # Sidebar controls for blog run
    with st.sidebar:
        st.header("Generate New Blog")
        topic = st.text_area("Topic", height=140)
        as_of = st.date_input("As-of date", value=date.today())
        run_btn = st.button("🚀 Generate Blog", type="primary")

        st.divider()
        st.subheader("Past blogs (local .md files)")
        # show simple list of .md in cwd
        past_files = sorted(Path(".").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if past_files:
            labels = [f"{p.stem}  ·  {p.name}" for p in past_files[:50]]
            selected_label = st.selectbox("Select saved blog", ["(none)"] + labels, index=0)
            selected_md_file = None
            if selected_label and selected_label != "(none)":
                idx = labels.index(selected_label)
                # guard index - ensure mapping to past_files subset
                if idx < len(past_files[:50]):
                    selected_md_file = past_files[idx]
            if st.button("📂 Load selected blog"):
                if selected_md_file:
                    md_text = selected_md_file.read_text(encoding="utf-8", errors="replace")
                    st.session_state["blog_last_out"] = {"plan": None, "evidence": [], "final_blog": md_text}
                    st.session_state["blog_topic_prefill"] = selected_md_file.stem

    # prefill hint if available
    topic_prefill = st.session_state.get("blog_topic_prefill", "")
    if topic_prefill and not st.session_state.get("blog_topic_filled"):
        # show a little hint above the editor if we have a prefill
        st.caption(f"Loaded topic hint: {topic_prefill}")

    # storage for last run
    if "blog_last_out" not in st.session_state:
        st.session_state["blog_last_out"] = None

    # Tabs for plan/evidence/preview/logs
    tab_plan, tab_evidence, tab_preview, tab_logs = st.tabs(
        ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🧾 Logs"]
    )

    logs: List[str] = []

    def log(msg: str):
        logs.append(msg)

    if run_btn:
        if not topic or not topic.strip():
            st.warning("Please enter a topic.")
            st.stop()

        inputs: Dict[str, Any] = {
            "topic": topic.strip(),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of.isoformat(),
            "recency_days": 7,
            "sections": [],
            "final_blog": "",
        }

        status = st.status("Running blog agent…", expanded=True)
        progress_area = st.empty()

        current_state: Dict[str, Any] = {}
        last_node = None

        try:
            for kind, payload in try_stream(blog_app, inputs):
                if kind in ("updates", "values"):
                    node_name = None
                    if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                        node_name = next(iter(payload.keys()))
                    if node_name and node_name != last_node:
                        status.write(f"➡️ Node: `{node_name}`")
                        last_node = node_name

                    current_state = extract_latest_state(current_state, payload)

                    summary = {
                        "mode": current_state.get("mode"),
                        "needs_research": current_state.get("needs_research"),
                        "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                        "evidence_count": len(current_state.get("evidence", []) or []),
                        # <-- fixed isinstance usage here (was a syntax error before)
                        "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                        "sections_done": len(current_state.get("sections", []) or []),
                    }
                    progress_area.json(summary)

                    log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

                elif kind == "final":
                    out = payload
                    # store in session_state for rendering/downloading
                    st.session_state["blog_last_out"] = out
                    status.update(label="✅ Done", state="complete", expanded=False)
                    log("[final] received final state")
        except Exception as e:
            try:
                status.update(label="❌ Error", state="error", expanded=False)
            except Exception:
                pass
            st.error(f"Blog run failed: {e}")
            log(f"error: {e}")

    # Render last result (if any)
    out = st.session_state.get("blog_last_out")
    if out:
        # --- Plan tab ---
        with tab_plan:
            st.subheader("Plan")
            plan_obj = out.get("plan")
            if not plan_obj:
                st.info("No plan found in output.")
            else:
                if hasattr(plan_obj, "model_dump"):
                    plan_dict = plan_obj.model_dump()
                elif isinstance(plan_obj, dict):
                    plan_dict = plan_obj
                else:
                    plan_dict = json.loads(json.dumps(plan_obj, default=str))

                st.write("**Title:**", plan_dict.get("blog_title"))
                cols = st.columns(3)
                cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
                cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
                cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

                tasks = plan_dict.get("tasks", [])
                if tasks:
                    df = pd.DataFrame(
                        [
                            {
                                "id": t.get("id"),
                                "title": t.get("title"),
                                "target_words": t.get("target_words"),
                                "requires_research": t.get("requires_research"),
                                "requires_citations": t.get("requires_citations"),
                                "requires_code": t.get("requires_code"),
                                "tags": ", ".join(t.get("tags") or []),
                            }
                            for t in tasks
                        ]
                    ).sort_values("id")
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    with st.expander("Task details"):
                        st.json(tasks)

        # --- Evidence tab ---
        with tab_evidence:
            st.subheader("Evidence")
            evidence = out.get("evidence") or []
            if not evidence:
                st.info("No evidence returned (maybe closed_book mode or no Tavily key/results).")
            else:
                rows = []
                for e in evidence:
                    if hasattr(e, "model_dump"):
                        e = e.model_dump()
                    rows.append(
                        {
                            "title": e.get("title"),
                            "published_at": e.get("published_at"),
                            "source": e.get("source"),
                            "url": e.get("url"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- Preview tab ---
        with tab_preview:
            st.subheader("Markdown Preview")

            # backend may name final markdown as "final_blog" or "final"
            final_md = out.get("final_blog") or out.get("final") or ""
            if not final_md:
                st.warning("No final markdown found.")
            else:
                st.markdown(final_md, unsafe_allow_html=False)

                # get blog title for filename
                plan_obj = out.get("plan")
                if hasattr(plan_obj, "blog_title"):
                    blog_title = plan_obj.blog_title
                elif isinstance(plan_obj, dict):
                    blog_title = plan_obj.get("blog_title", "blog")
                else:
                    def extract_title(md: str, fallback: str) -> str:
                        for line in md.splitlines():
                            if line.startswith("# "):
                                return line[2:].strip() or fallback
                        return fallback
                    blog_title = extract_title(final_md, "blog")

                md_filename = f"{re.sub(r'[^a-z0-9_-]+','_', blog_title.lower())}.md"
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                )

        # --- Logs tab ---
        with tab_logs:
            st.subheader("Logs")
            if "blog_logs" not in st.session_state:
                st.session_state["blog_logs"] = []
            if logs:
                st.session_state["blog_logs"].extend(logs)

            st.text_area("Event log", value="\n\n".join(st.session_state["blog_logs"][-120:]), height=520)
    else:
        st.info("Enter a topic and click **Generate Blog**.")


# ============================================================
# ================= Router ============================
# ============================================================

if mode.startswith("🧠"):
    run_memory_chat()

elif mode.startswith("📚"):
    run_rag_chat()

elif mode.startswith("✍️"):
    run_blog_agent()