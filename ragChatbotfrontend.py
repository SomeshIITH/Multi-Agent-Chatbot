
# import uuid
# import hashlib
# import streamlit as st
# from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# from Chatbot import (
#     chatbot,
#     ingest_pdf,
#     retrieve_all_threads,
#     thread_document_metadata,
# )

# # =========================== Utilities ===========================

# def generate_thread_id():
#     # Always use string IDs
#     return str(uuid.uuid4())


# def reset_chat():
#     thread_id = generate_thread_id()
#     st.session_state["thread_id"] = thread_id
#     add_thread(thread_id)
#     st.session_state["message_history"] = []


# def add_thread(thread_id):
#     if thread_id not in st.session_state["chat_threads"]:
#         st.session_state["chat_threads"].append(thread_id)


# def load_conversation(thread_id):
#     try:
#         state = chatbot.get_state(
#             config={"configurable": {"thread_id": thread_id}}
#         )
#         if not state or not state.values:
#             return []
#         return state.values.get("messages", [])
#     except Exception:
#         return []


# # ======================= Session Initialization ===================

# if "message_history" not in st.session_state:
#     st.session_state["message_history"] = []

# if "thread_id" not in st.session_state:
#     st.session_state["thread_id"] = generate_thread_id()

# if "chat_threads" not in st.session_state:
#     st.session_state["chat_threads"] = retrieve_all_threads()

# if "ingested_docs" not in st.session_state:
#     st.session_state["ingested_docs"] = {}

# add_thread(st.session_state["thread_id"])

# thread_key = st.session_state["thread_id"]
# thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
# threads = st.session_state["chat_threads"][::-1]
# selected_thread = None

# # ============================ Sidebar ============================

# st.sidebar.title("LangGraph PDF Chatbot")
# st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

# if st.sidebar.button("New Chat", use_container_width=True):
#     reset_chat()
#     st.rerun()

# # Document status
# if thread_docs:
#     latest_doc = list(thread_docs.values())[-1]
#     st.sidebar.success(
#         f"Using `{latest_doc.get('filename')}` "
#         f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
#     )
# else:
#     st.sidebar.info("No PDF indexed yet.")

# # PDF upload
# uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# if uploaded_pdf:
#     pdf_bytes = uploaded_pdf.getvalue()
#     pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

#     if pdf_hash in thread_docs:
#         st.sidebar.info("PDF already indexed for this chat.")
#     else:
#         with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
#             summary = ingest_pdf(
#                 pdf_bytes,
#                 thread_id=thread_key,
#                 filename=uploaded_pdf.name,
#             )
#             thread_docs[pdf_hash] = summary
#             status_box.update(label="✅ PDF indexed", state="complete")

# # Past conversations
# st.sidebar.subheader("Past conversations")

# if not threads:
#     st.sidebar.write("No past conversations yet.")
# else:
#     for t in threads:
#         if st.sidebar.button(t, key=f"thread-{t}"):
#             selected_thread = t

# # ============================ Main Layout ========================

# st.title("Multi tasking Chatbot")

# # Display history
# for msg in st.session_state["message_history"]:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# user_input = st.chat_input("Ask about your document or use tools")

# if user_input:

#     # Show user message
#     st.session_state["message_history"].append(
#         {"role": "user", "content": user_input}
#     )

#     with st.chat_message("user"):
#         st.write(user_input)

#     CONFIG = {
#         "configurable": {"thread_id": thread_key},
#         "metadata": {"thread_id": thread_key},
#         "run_name": "chat_turn",
#     }

#     with st.chat_message("assistant"):

#         status_holder = {"box": None}

#         def ai_stream():
#             for chunk, _ in chatbot.stream(
#                 {"messages": [HumanMessage(content=user_input)]},
#                 config=CONFIG,
#                 stream_mode="messages",
#             ):

#                 # Tool activity indicator
#                 if isinstance(chunk, ToolMessage):
#                     tool_name = getattr(chunk, "name", "tool")

#                     if status_holder["box"] is None:
#                         status_holder["box"] = st.status(
#                             f"🔧 Using `{tool_name}`…", expanded=True
#                         )
#                     else:
#                         status_holder["box"].update(
#                             label=f"🔧 Using `{tool_name}`…",
#                             state="running",
#                         )

#                 # AI message streaming
#                 if isinstance(chunk, AIMessage):
#                     if isinstance(chunk.content, str):
#                         yield chunk.content

#         ai_response = st.write_stream(ai_stream())

#         if status_holder["box"] is not None:
#             status_holder["box"].update(
#                 label="✅ Tool finished",
#                 state="complete",
#                 expanded=False,
#             )

#     # Save assistant response
#     st.session_state["message_history"].append(
#         {"role": "assistant", "content": ai_response}
#     )

#     # Show document metadata
#     meta = thread_document_metadata(thread_key)
#     if meta:
#         st.caption(
#             f"Document indexed: {meta.get('filename')} "
#             f"(chunks: {meta.get('chunks')}, pages: {meta.get('documents')})"
#         )

# st.divider()

# # ================= Thread switching =================

# if selected_thread:

#     st.session_state["thread_id"] = selected_thread

#     messages = load_conversation(selected_thread)

#     temp = []
#     for m in messages:
#         role = "user" if isinstance(m, HumanMessage) else "assistant"
#         temp.append({"role": role, "content": m.content})

#     st.session_state["message_history"] = temp
#     st.session_state["ingested_docs"].setdefault(selected_thread, {})

#     st.toast("Switched conversation")
#     st.rerun()


import uuid
import hashlib
import string
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ragChatbotBackend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ========================= Base62 Thread ID =========================

BASE62 = string.digits + string.ascii_letters


def base62_encode(num: int) -> str:
    if num == 0:
        return BASE62[0]

    arr = []
    base = len(BASE62)

    while num:
        num, rem = divmod(num, base)
        arr.append(BASE62[rem])

    arr.reverse()
    return "".join(arr)


def generate_thread_id():
    raw = uuid.uuid4().bytes
    digest = hashlib.sha256(raw).digest()
    num = int.from_bytes(digest, "big")

    short_id = base62_encode(num)[:8]
    return f"chat-{short_id}"


# =========================== Utilities ===========================

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        if not state or not state.values:
            return []
        return state.values.get("messages", [])
    except Exception:
        return []


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = st.session_state["thread_id"]
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Document status
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

# PDF upload
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

    if pdf_hash in thread_docs:
        st.sidebar.info("PDF already indexed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                pdf_bytes,
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[pdf_hash] = summary
            status_box.update(label="✅ PDF indexed", state="complete")

# Past conversations
st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for t in threads:
        if st.sidebar.button(t, key=f"thread-{t}"):
            selected_thread = t

# ============================ Main Layout ========================

st.title("Multi Tasking Chatbot")

# Display history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):

        status_holder = {"box": None}

        def ai_stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):

                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                        )

                if isinstance(chunk, AIMessage):
                    if isinstance(chunk.content, str):
                        yield chunk.content

        ai_response = st.write_stream(ai_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_response}
    )

    meta = thread_document_metadata(thread_key)
    if meta:
        st.caption(
            f"Document indexed: {meta.get('filename')} "
            f"(chunks: {meta.get('chunks')}, pages: {meta.get('documents')})"
        )

st.divider()

# ================= Thread switching =================

if selected_thread:

    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(selected_thread)

    temp = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        temp.append({"role": role, "content": m.content})

    st.session_state["message_history"] = temp
    st.session_state["ingested_docs"].setdefault(selected_thread, {})

    st.toast("Switched conversation")
    st.rerun()
