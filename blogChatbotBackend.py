# from __future__ import annotations
# import operator

# from typing import TypedDict,List,Annotated,Literal,Optional
# from pydantic import BaseModel, Field

# from langgraph.graph import START,END,StateGraph
# from langgraph.types import Send

# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage, SystemMessage

# from langchain_community.tools.tavily_search import TavilySearchResults

# import os
# from dotenv import load_dotenv
# from pathlib import Path
# from datetime import date, timedelta
# load_dotenv()

# LLM1 = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=os.getenv('GROQ_API_KEY'))

# class Task(BaseModel):
#     id : int
#     title : str
#     goal : str= Field(...,description="One sentence describing what the reader should learn after reading this section")
#     bullets : List[str] = Field(...,min_length=3,max_length=5,description="2-3 non overlapping subpoints to cover in this secton")
#     target_words : int  = Field(...,description="Number of words to cover in this section (50-100)")
#     tags : List[str] = Field(default_factory=list)
    
#     requires_citations : bool = False
#     requires_code : bool = False
#     requires_research : bool = False
    
    
# class Plan(BaseModel):
#     blog_title : str
#     audience : str = Field(...,description="Who is this blog for")
#     tone : str = Field(...,description="What is the writing tone of the blog(casual,informal,formal)")
#     blog_kind : Literal["explainer","biotechnology","tutorial"] = "explainer"
#     constraints : List[str] = Field(default_factory=list)
#     tasks : List[Task]
    
# class SearchedItem(BaseModel):
#     title : str
#     url : str
#     published_at : Optional[str]= None
#     snippet : Optional[str] = None
#     source : Optional[str] = None
    
# class SearchedPacks(BaseModel):
#     evidence : List[SearchedItem] = Field(default_factory=list)
    
    
# class RouterDecision(BaseModel):
#     needs_research : bool
#     mode : Literal["open-book","close-book","hybrid"]
#     queries : List[str] = Field(default_factory=list)
    
# class State(TypedDict):
#     topic : str     #user will provide topic
#     mode : str      #routing decision
#     needs_research : bool 
#     queries : List[str]
#     evidence : Optional[SearchedItem]
    
#     plan : Optional[Plan]     #orchestrator will provide plan
#     sections : Annotated[List[tuple[int,str]],operator.add]    #each worker will provide section string and we add here (taskid,section)
#     final_blog : str    #we store final blog here
    

# router_system_content =""" You are a routing module for a technical blog planner.

# Decide whether web research is needed BEFORE planning.

# Modes:
# - close-book (needs_research=false):
#   Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
# - hybrid (needs_research=true):
#   Mostly evergreen but needs up-to-date examples/tools/models to be useful.
# - open-book (needs_research=true):
#   Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

# If needs_research=true:
# - Output 2-3 high-signal queries.
# - Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM")."""


# research_system_content = """You are a research synthesizer for technical writing.

# Given raw web search results, produce a deduplicated list of EvidenceItem objects.

# Rules:
# - Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
# - If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
#   If missing or unclear, set published_at=null. Do NOT guess.
# - Keep snippets short.
# - Deduplicate by URL.
# """

# orchestrator_system_content = """You are a senior technical writer and developer advocate.
# Your job is to produce a highly actionable outline for a technical blog post.

# Hard requirements:
# - Create 2-3 sections (tasks) suitable for the topic and audience.
# - Each task must include:
#   1) goal (1 sentence)
#   2) 2 -3 bullets that are concrete, specific, and non-overlapping
#   3) target word count (50 - 100)

# Quality bar:
# - Assume the reader is a developer; use correct terminology.
# - Ensure the overall plan includes at least 2 of these somewhere:
#   * minimal code sketch / MWE (set requires_code=True for that section)
#   * edge cases / failure modes
#   * performance/cost considerations
#   * security/privacy considerations (if relevant)
#   * debugging/observability tips

# Grounding rules:
# - Mode closed_book: keep it evergreen; do not depend on evidence.
# - Mode hybrid:
#   - Use evidence for up-to-date examples (models/tools/releases) in bullets.
#   - Mark sections using fresh info as requires_research=True and requires_citations=True.
# - Mode open_book:
#   - Set blog_kind = "news_roundup".
#   - Every section is about summarizing events + implications.
#   - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
#   - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
#     and includes only what can be supported.

# Output must strictly match the Plan schema.
# """

# worker_system_content = """You are a senior technical writer and developer advocate.
# Write ONE section of a technical blog post in Markdown.

# Hard constraints:
# - Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
# - Stay close to Target words (±15%).
# - Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
# - Start with a '## <Section Title>' heading.

# Scope guard:
# - If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
#   Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
#   Focus on summarizing events and implications.

# Grounding policy:
# - If mode == open_book:
#   - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
#   - For each event claim, attach a source as a Markdown link: ([Source](URL)).
#   - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
# - If requires_citations == true:
#   - For outside-world claims, cite Evidence URLs the same way.
# - Evergreen reasoning is OK without citations unless requires_citations is true.

# Code:
# - If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

# Style:
# - Short paragraphs, bullets where helpful, code fences for code.
# """

# def router(state : State)->dict:
#     topic = state["topic"]
#     decision = LLM1.with_structured_output(RouterDecision,method="function_calling").invoke(
#         [
#             SystemMessage(content=f"{router_system_content}"),
#             HumanMessage(content = f"Topic : {topic}")
#         ]
#     )
#     return {"mode" : decision.mode, "needs_research" : decision.needs_research, "queries" : decision.queries}

# def routernext(state : State):
#     if state["needs_research"] == True:
#         return "research_node"
#     else:
#         return "orchestrator"
    

# def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    
#     tool = TavilySearchResults(max_results=max_results)
#     results = tool.invoke({"query": query})

#     normalized: List[dict] = []
#     for r in results or []:
#         normalized.append(
#             {
#                 "title": r.get("title") or "",
#                 "url": r.get("url") or "",
#                 "snippet": r.get("content") or r.get("snippet") or "",
#                 "published_at": r.get("published_date") or r.get("published_at"),
#                 "source": r.get("source"),
#             }
#         )
#     return normalized
  

# def research_node(state: State) -> dict:

#     # take the first 10 queries from state
#     queries = (state.get("queries", []) or [])
#     max_results = 2

#     raw_results: List[dict] = []

#     for q in queries:
#         raw_results.extend(_tavily_search(q, max_results=max_results))

#     if not raw_results:
#         return {"evidence": []}

#     extractor = LLM1.with_structured_output(SearchedPacks)
#     pack = extractor.invoke(
#         [
#             SystemMessage(content=research_system_content),
#             HumanMessage(content=f"Raw results:\n{raw_results}"),
#         ]
#     )

#     # Deduplicate by URL
#     dedup = {}
#     for e in pack.evidence:
#         if e.url:
#             dedup[e.url] = e

#     return {"evidence": list(dedup.values())}


# def orchestrator(state : State)->dict:
#     """ orchestrator function take state-topic as input and return Plan Object as output"""
#     evidence = state.get("evidence",[])
#     mode = state.get("mode","close-book")
    
#     plan = LLM1.with_structured_output(Plan).invoke(
#         [
#             SystemMessage(content=f"{orchestrator_system_content}"),
#             HumanMessage(content = f"Topic : {state['topic']}\n" f"Mode : {mode}\n" f"Evidence (ONLY use for fresh claims; may be empty):\n" f"{[e.model_dump() for e in evidence][:16]}")
#         ]
#     )
#     return {"plan" : plan}

# def fanout(state : State):
#     """ create n workers for n tasks in plan"""
#     return [Send(
#             "worker",
#                 {
#                     "task": task.model_dump(),
#                     "topic": state["topic"],
#                     "mode": state["mode"],
#                     "plan": state["plan"].model_dump(),
#                     "evidence": [e.model_dump() for e in state.get("evidence", [])],
#                 },
#             )
#             for task in state["plan"].tasks
#     ]
    
# def worker(payload : dict)->dict:
#     task = Task(**payload["task"])
#     plan = Plan(**payload["plan"])
#     evidence = [SearchedItem(**e) for e in payload.get("evidence", [])]
#     topic = payload["topic"]
#     mode = payload.get("mode", "closed_book")

#     bullets_text = "\n- " + "\n- ".join(task.bullets)

#     evidence_text = ""
#     if evidence:
#         evidence_text = "\n".join(
#             f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
#             for e in evidence[:20]
#         )

#     section_md = LLM1.invoke(
#         [
#             SystemMessage(content=worker_system_content),
#             HumanMessage(
#                 content=(
#                     f"Blog title: {plan.blog_title}\n"
#                     f"Audience: {plan.audience}\n"
#                     f"Tone: {plan.tone}\n"
#                     f"Blog kind: {plan.blog_kind}\n"
#                     f"Constraints: {plan.constraints}\n"
#                     f"Topic: {topic}\n"
#                     f"Mode: {mode}\n\n"
#                     f"Section title: {task.title}\n"
#                     f"Goal: {task.goal}\n"
#                     f"Target words: {task.target_words}\n"
#                     f"Tags: {task.tags}\n"
#                     f"requires_research: {task.requires_research}\n"
#                     f"requires_citations: {task.requires_citations}\n"
#                     f"requires_code: {task.requires_code}\n"
#                     f"Bullets:{bullets_text}\n\n"
#                     f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
#                 )
#             ),
#         ]
#     ).content.strip()

#     return {"sections": [(task.id, section_md)]}

# def reducer(state:State)->dict:
    
#     plan = state["plan"]

#     ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
#     body = "\n\n".join(ordered_sections).strip()
#     final_md = f"# {plan.blog_title}\n\n{body}\n"

#     filename = f"{plan.blog_title}.md"
#     Path(filename).write_text(final_md, encoding="utf-8")

#     return {"final_blog": final_md}




# graph = StateGraph(State)
# graph.add_node("router",router)
# graph.add_node("research_node",research_node)
# graph.add_node("orchestrator",orchestrator)
# graph.add_node("worker",worker)
# graph.add_node("reducer",reducer)

# graph.add_edge(START,"router")
# graph.add_conditional_edges("router",routernext,{"research_node": "research_node", "orchestrator": "orchestrator"})
# # orchestrator node, call the fanout function to decide what happens next, and route execution to one or more worker nodes based on its result.”
# graph.add_edge("research_node","orchestrator")
# graph.add_conditional_edges("orchestrator",fanout,["worker"])
# graph.add_edge("worker","reducer")
# graph.add_edge("reducer",END)

# app = graph.compile()



#### Here i used 2 LLM to distribute task to workers.


from __future__ import annotations
import operator

from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# TWO-LLM ARCHITECTURE (ONLY CHANGE)

PLANNER_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

WORKER_LLM = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# MODELS

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(...)
    bullets: List[str] = Field(..., min_length=3, max_length=5)
    target_words: int = Field(...)
    tags: List[str] = Field(default_factory=list)

    requires_citations: bool = False
    requires_code: bool = False
    requires_research: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "biotechnology", "tutorial"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class SearchedItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class SearchedPacks(BaseModel):
    evidence: List[SearchedItem] = Field(default_factory=list)


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["open-book", "close-book", "hybrid"]
    queries: List[str] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: Optional[List[SearchedItem]]

    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    final_blog: str


# PROMPTS 


router_system_content = """ You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.
Modes:
- close-book
- hybrid
- open-book
If needs_research=true:
- Output 2-3 high-signal queries.
"""

research_system_content = """You are a research synthesizer for technical writing."""

orchestrator_system_content = """You are a senior technical writer and developer advocate."""

worker_system_content = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown."""

# ROUTER


def router(state: State) -> dict:
    topic = state["topic"]

    decision = PLANNER_LLM.with_structured_output(
        RouterDecision, method="function_calling"
    ).invoke(
        [
            SystemMessage(content=router_system_content),
            HumanMessage(content=f"Topic : {topic}"),
        ]
    )

    return {
        "mode": decision.mode,
        "needs_research": decision.needs_research,
        "queries": decision.queries,
    }


def routernext(state: State):
    return "research_node" if state["needs_research"] else "orchestrator"



# RESEARCH


def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


def research_node(state: State) -> dict:
    queries = state.get("queries", [])
    raw_results = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=2))

    if not raw_results:
        return {"evidence": []}

    extractor = PLANNER_LLM.with_structured_output(SearchedPacks)

    pack = extractor.invoke(
        [
            SystemMessage(content=research_system_content),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}



def orchestrator(state: State) -> dict:
    evidence = state.get("evidence", [])
    mode = state.get("mode", "close-book")

    plan = PLANNER_LLM.with_structured_output(Plan).invoke(
        [
            SystemMessage(content=orchestrator_system_content),
            HumanMessage(
                content=f"Topic: {state['topic']}\nMode: {mode}\nEvidence:\n{[e.model_dump() for e in evidence][:16]}"
            ),
        ]
    )

    return {"plan": plan}



def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]



def worker(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [SearchedItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = "\n".join(
        f"- {e.title} | {e.url}" for e in evidence[:5]
    )

    section_md = WORKER_LLM.invoke(
        [
            SystemMessage(content=worker_system_content),
            HumanMessage(
                content=f"""
Blog title: {plan.blog_title}
Topic: {payload["topic"]}

Section: {task.title}
Goal: {task.goal}
Words: {task.target_words}

Bullets:{bullets_text}

Evidence:
{evidence_text}
"""
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}



def reducer(state: State) -> dict:
    plan = state["plan"]

    ordered_sections = [
        md for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]

    final_md = f"# {plan.blog_title}\n\n" + "\n\n".join(ordered_sections)

    Path(f"{plan.blog_title}.md").write_text(final_md, encoding="utf-8")

    return {"final_blog": final_md}



graph = StateGraph(State)

graph.add_node("router", router)
graph.add_node("research_node", research_node)
graph.add_node("orchestrator", orchestrator)
graph.add_node("worker", worker)
graph.add_node("reducer", reducer)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    routernext,
    {"research_node": "research_node", "orchestrator": "orchestrator"},
)

graph.add_edge("research_node", "orchestrator")
graph.add_conditional_edges("orchestrator", fanout, ["worker"])
graph.add_edge("worker", "reducer")
graph.add_edge("reducer", END)

app = graph.compile()




    

