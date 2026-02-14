"""
LangGraph Long-Term Memory Chatbot Backend with  SQLite conversation checkpoints ,  Long-term memory store
, Automatic memory extraction , Personalized replies , Streaming compatible
"""

from __future__ import annotations

import os
import uuid
import sqlite3
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangGraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.checkpoint.sqlite import SqliteSaver

# LangChain
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq

load_dotenv()

# ============================================================
# MODELS
# ============================================================

chat_llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

memory_llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ============================================================
# MEMORY STORE
# ============================================================

store = InMemoryStore()

# ============================================================
# MEMORY SCHEMA
# ============================================================

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory")
    is_new: bool

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

memory_extractor = memory_llm.with_structured_output(MemoryDecision)

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are a personalized assistant with long-term memory.

User memory:
{user_memory}

Rules:
- Personalize responses when possible
- Use known name/preferences
- Avoid guessing
- Suggest 3 follow-up questions
"""

MEMORY_PROMPT = """
Extract stable personal facts:

Existing memory:
{existing}

Store only:
- name
- goals
- projects
- preferences

Mark duplicates as is_new=false.
"""

# MEMORY NODE

def remember_node(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "memory")

    existing = store.search(ns)
    existing_text = "\n".join(
        item.value["data"] for item in existing
    ) or "(empty)"

    last_user_msg = state["messages"][-1].content

    decision: MemoryDecision = memory_extractor.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(existing=existing_text)),
        {"role": "user", "content": last_user_msg},
    ])

    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(ns, str(uuid.uuid4()), {"data": mem.text})

    return {}

# ============================================================
# CHAT NODE
# ============================================================

def chat_node(
    state: MessagesState,
    config: RunnableConfig,
    store: BaseStore,
):
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "memory")

    items = store.search(ns)
    memory_text = "\n".join(
        item.value["data"] for item in items
    ) or "(empty)"

    system_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(user_memory=memory_text)
    )

    response = chat_llm.invoke(
        [system_msg] + state["messages"]
    )

    return {"messages": [response]}

# GRAPH BUILD

builder = StateGraph(MessagesState)

builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_edge("chat", END)

# SQLite checkpoint
conn = sqlite3.connect("simpleChatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

chatbot = builder.compile(
    store=store,
    checkpointer=checkpointer,
)

# UTILITIES

def retrieve_all_threads():
    threads = set()
    for cp in checkpointer.list(None):
        threads.add(cp.config["configurable"]["thread_id"])
    return list(threads)
