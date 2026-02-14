import uuid
import streamlit as st

from simpleChatbotBackend import chatbot, retrieve_all_threads
from langchain_core.messages import AIMessage, HumanMessage

# ========================= Thread Utilities =========================

def generate_thread_id():
    return f"chat-{uuid.uuid4().hex[:8]}"

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def load_conversation(thread_id, user_id):
    try:
        state = chatbot.get_state(
            config={"configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
            }}
        )
        if not state or not state.values:
            return []

        msgs = state.values.get("messages", [])

        return [
            {
                "role": "assistant" if isinstance(m, AIMessage) else "user",
                "content": m.content,
            }
            for m in msgs
        ]

    except Exception:
        return []

# ======================= Session Initialization ===================

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "somesh"

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

thread_key = st.session_state["thread_id"]
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================

st.sidebar.title(" Memory Chatbot")
st.sidebar.markdown(f"**Current Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for t in threads:
        if st.sidebar.button(t, key=f"thread-{t}"):
            selected_thread = t

# ============================ Main Layout ========================

st.title("I'm Memory bot , how are  you ?")

# Display history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type message…")

if user_input:

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": thread_key,
            "user_id": st.session_state["user_id"],
        },
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):

        def stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage):
                    yield chunk.content

        ai_text = st.write_stream(stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_text}
    )

st.divider()

# ================= Thread Switching =================

if selected_thread:

    st.session_state["thread_id"] = selected_thread

    st.session_state["message_history"] = load_conversation(
        selected_thread,
        st.session_state["user_id"]
    )

    st.toast("Switched conversation")
    st.rerun()
