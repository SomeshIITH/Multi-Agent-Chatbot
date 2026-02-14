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
#     bullets : List[str] = Field(...,min_length=3,max_length=5,description="3-5 non overlapping subpoints to cover in this secton")
#     target_words : int  = Field(...,description="Number of words to cover in this section (100-150)")
#     tags : List[str] = Field(default_factory=list)
    
#     requires_citations : bool = False
#     requires_code : bool = False
#     requires_research : bool = False
      
# class Plan(BaseModel):
#     blog_title : str
#     audience : str = Field(...,description="Who is this blog for")
#     tone : str = Field(...,description="What is the writing tone of the blog(casual,informal,crsip,formal)")
#     blog_kind : Literal["explainer","system-design","biotechnology","tutorial","news-roundup","comparison"] = "explainer"
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
    
#     #extra added thing for image
#     as_of : str
#     recency_days : int
#     merged_md : str
#     md_with_placeholders : str
#     image_specs : List[dict]
      
# class ImageSpec(BaseModel):
#     placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
#     filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
#     alt: str
#     caption: str
#     prompt: str = Field(..., description="Prompt to send to the image model.")
#     size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
#     quality: Literal["low", "medium", "high"] = "medium"

# class GlobalImagePlan(BaseModel):
#     md_with_placeholders: str
#     images: List[ImageSpec] = Field(default_factory=list)
    
    
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
# - Output 3-5 high-signal queries.
# - Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
# - If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES."""

# research_system_content = """You are a research synthesizer for technical writing.

# Given raw web search results, produce a deduplicated list of EvidenceItem objects.

# Rules:
# - Only include items with a non-empty url.
# - Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
# - If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
#   If missing or unclear, set published_at=null. Do NOT guess.
# - Keep snippets short.
# - Deduplicate by URL.
# """

# orchestrator_system_content = """You are a senior technical writer and developer advocate.
# Your job is to produce a highly actionable outline for a technical blog post.

# Hard requirements:
# - Create 4-6 sections (tasks) suitable for the topic and audience.
# - Each task must include:
#   1) goal (1 sentence)
#   2) 3 - 4 bullets that are concrete, specific, and non-overlapping
#   3) target word count (150 - 200)

# Quality bar:
# - Assume the reader is a developer; use correct terminology.
# - Bullets must be actionable: build/compare/measure/verify/debug.
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
# - Avoid fluff/marketing. Be precise and implementation-oriented.
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
    
# def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    
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
#     max_results = 5

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


# # ============================================================
# # 8) ReducerWithImages (subgraph)
# #    merge_content -> decide_images -> generate_and_place_images
# # ============================================================
# def merge_content(state: State) -> dict:

#     plan = state["plan"]

#     ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
#     body = "\n\n".join(ordered_sections).strip()
#     merged_md = f"# {plan.blog_title}\n\n{body}\n"
#     return {"merged_md": merged_md}


# DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
# Decide if images/diagrams are needed for THIS blog.

# Rules:
# - Max 3 images total.
# - Each image must materially improve understanding (diagram/flow/table-like visual).
# - Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
# - If no images needed: md_with_placeholders must equal input and images=[].
# - Avoid decorative images; prefer technical diagrams with short labels.
# Return strictly GlobalImagePlan.
# """

# def decide_images(state: State) -> dict:
    
#     planner = LLM1.with_structured_output(GlobalImagePlan)
#     merged_md = state["merged_md"]
#     plan = state["plan"]
#     assert plan is not None

#     image_plan = planner.invoke(
#         [
#             SystemMessage(content=DECIDE_IMAGES_SYSTEM),
#             HumanMessage(
#                 content=(
#                     f"Blog kind: {plan.blog_kind}\n"
#                     f"Topic: {state['topic']}\n\n"
#                     "Insert placeholders + propose image prompts.\n\n"
#                     f"{merged_md}"
#                 )
#             ),
#         ]
#     )

#     return {
#         "md_with_placeholders": image_plan.md_with_placeholders,
#         "image_specs": [img.model_dump() for img in image_plan.images],
#     }


# def _gemini_generate_image_bytes(prompt: str) -> bytes:
#     """
#     Returns raw image bytes generated by Gemini.
#     Requires: pip install google-genai
#     Env var: GOOGLE_API_KEY
#     """
#     from google import genai
#     from google.genai import types

#     api_key = os.environ.get("GOOGLE_API_KEY")
#     if not api_key:
#         raise RuntimeError("GOOGLE_API_KEY is not set.")

#     client = genai.Client(api_key=api_key)

#     resp = client.models.generate_content(
#         model="gemini-2.5-flash-image",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_modalities=["IMAGE"],
#             safety_settings=[
#                 types.SafetySetting(
#                     category="HARM_CATEGORY_DANGEROUS_CONTENT",
#                     threshold="BLOCK_ONLY_HIGH",
#                 )
#             ],
#         ),
#     )

#     # Depending on SDK version, parts may hang off resp.candidates[0].content.parts
#     parts = getattr(resp, "parts", None)
#     if not parts and getattr(resp, "candidates", None):
#         try:
#             parts = resp.candidates[0].content.parts
#         except Exception:
#             parts = None

#     if not parts:
#         raise RuntimeError("No image content returned (safety/quota/SDK change).")

#     for part in parts:
#         inline = getattr(part, "inline_data", None)
#         if inline and getattr(inline, "data", None):
#             return inline.data

#     raise RuntimeError("No inline image bytes found in response.")


# def generate_and_place_images(state: State) -> dict:

#     plan = state["plan"]
#     assert plan is not None

#     md = state.get("md_with_placeholders") or state["merged_md"]
#     image_specs = state.get("image_specs", []) or []

#     # If no images requested, just write merged markdown
#     if not image_specs:
#         filename = f"{plan.blog_title}.md"
#         Path(filename).write_text(md, encoding="utf-8")
#         return {"final": md}

#     images_dir = Path("images")
#     images_dir.mkdir(exist_ok=True)

#     for spec in image_specs:
#         placeholder = spec["placeholder"]
#         filename = spec["filename"]
#         out_path = images_dir / filename

#         # generate only if needed
#         if not out_path.exists():
#             try:
#                 img_bytes = _gemini_generate_image_bytes(spec["prompt"])
#                 out_path.write_bytes(img_bytes)
#             except Exception as e:
#                 # graceful fallback: keep doc usable
#                 prompt_block = (
#                     f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
#                     f"> **Alt:** {spec.get('alt','')}\n>\n"
#                     f"> **Prompt:** {spec.get('prompt','')}\n>\n"
#                     f"> **Error:** {e}\n"
#                 )
#                 md = md.replace(placeholder, prompt_block)
#                 continue

#         img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
#         md = md.replace(placeholder, img_md)

#     filename = f"{plan.blog_title}.md"
#     Path(filename).write_text(md, encoding="utf-8")
#     return {"final": md}

# # build reducer subgraph
# reducer_graph = StateGraph(State)
# reducer_graph.add_node("merge_content", merge_content)
# reducer_graph.add_node("decide_images", decide_images)
# reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
# reducer_graph.add_edge(START, "merge_content")
# reducer_graph.add_edge("merge_content", "decide_images")
# reducer_graph.add_edge("decide_images", "generate_and_place_images")
# reducer_graph.add_edge("generate_and_place_images", END)
# reducer_subgraph = reducer_graph.compile()

# reducer_subgraph



# graph = StateGraph(State)
# graph.add_node("router",router)
# graph.add_node("research_node",research_node)
# graph.add_node("orchestrator",orchestrator)
# graph.add_node("worker",worker)
# graph.add_node("reducer",reducer_subgraph)

# graph.add_edge(START,"router")
# graph.add_conditional_edges("router",routernext,{"research_node": "research_node", "orchestrator": "orchestrator"})
# # orchestrator node, call the fanout function to decide what happens next, and route execution to one or more worker nodes based on its result.”
# graph.add_edge("research_node","orchestrator")
# graph.add_conditional_edges("orchestrator",fanout,["worker"])
# graph.add_edge("worker","reducer")
# graph.add_edge("reducer",END)

# app = graph.compile()


from __future__ import annotations
import operator

from typing import TypedDict,List,Annotated,Literal,Optional
from pydantic import BaseModel, Field

from langgraph.graph import START,END,StateGraph
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.tools.tavily_search import TavilySearchResults

import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import date, timedelta
load_dotenv()

LLM1 = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=os.getenv('GROQ_API_KEY'))

class Task(BaseModel):
    id : int
    title : str
    goal : str= Field(...,description="One sentence describing what the reader should learn after reading this section")
    bullets : List[str] = Field(...,min_length=3,max_length=5,description="2-3 non overlapping subpoints to cover in this secton")
    target_words : int  = Field(...,description="Number of words to cover in this section (50-100)")
    tags : List[str] = Field(default_factory=list)
    
    requires_citations : bool = False
    requires_code : bool = False
    requires_research : bool = False
    
    
class Plan(BaseModel):
    blog_title : str
    audience : str = Field(...,description="Who is this blog for")
    tone : str = Field(...,description="What is the writing tone of the blog(casual,informal,formal)")
    blog_kind : Literal["explainer","biotechnology","tutorial"] = "explainer"
    constraints : List[str] = Field(default_factory=list)
    tasks : List[Task]
    
class SearchedItem(BaseModel):
    title : str
    url : str
    published_at : Optional[str]= None
    snippet : Optional[str] = None
    source : Optional[str] = None
    
class SearchedPacks(BaseModel):
    evidence : List[SearchedItem] = Field(default_factory=list)
    
    
class RouterDecision(BaseModel):
    needs_research : bool
    mode : Literal["open-book","close-book","hybrid"]
    queries : List[str] = Field(default_factory=list)
    
class State(TypedDict):
    topic : str     #user will provide topic
    mode : str      #routing decision
    needs_research : bool 
    queries : List[str]
    evidence : Optional[SearchedItem]
    
    plan : Optional[Plan]     #orchestrator will provide plan
    sections : Annotated[List[tuple[int,str]],operator.add]    #each worker will provide section string and we add here (taskid,section)
    final_blog : str    #we store final blog here
    

router_system_content =""" You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- close-book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open-book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 2-3 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM")."""


research_system_content = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

orchestrator_system_content = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 2-3 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 2 -3 bullets that are concrete, specific, and non-overlapping
  3) target word count (50 - 100)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

worker_system_content = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
"""

def router(state : State)->dict:
    topic = state["topic"]
    decision = LLM1.with_structured_output(RouterDecision,method="function_calling").invoke(
        [
            SystemMessage(content=f"{router_system_content}"),
            HumanMessage(content = f"Topic : {topic}")
        ]
    )
    return {"mode" : decision.mode, "needs_research" : decision.needs_research, "queries" : decision.queries}

def routernext(state : State):
    if state["needs_research"] == True:
        return "research_node"
    else:
        return "orchestrator"
    

def _tavily_search(query: str, max_results: int = 3) -> List[dict]:
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
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

    # take the first 10 queries from state
    queries = (state.get("queries", []) or [])
    max_results = 2

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = LLM1.with_structured_output(SearchedPacks)
    pack = extractor.invoke(
        [
            SystemMessage(content=research_system_content),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


def orchestrator(state : State)->dict:
    """ orchestrator function take state-topic as input and return Plan Object as output"""
    evidence = state.get("evidence",[])
    mode = state.get("mode","close-book")
    
    plan = LLM1.with_structured_output(Plan).invoke(
        [
            SystemMessage(content=f"{orchestrator_system_content}"),
            HumanMessage(content = f"Topic : {state['topic']}\n" f"Mode : {mode}\n" f"Evidence (ONLY use for fresh claims; may be empty):\n" f"{[e.model_dump() for e in evidence][:16]}")
        ]
    )
    return {"plan" : plan}

def fanout(state : State):
    """ create n workers for n tasks in plan"""
    return [Send(
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
    
def worker(payload : dict)->dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [SearchedItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = LLM1.invoke(
        [
            SystemMessage(content=worker_system_content),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

def reducer(state:State)->dict:
    
    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final_blog": final_md}




graph = StateGraph(State)
graph.add_node("router",router)
graph.add_node("research_node",research_node)
graph.add_node("orchestrator",orchestrator)
graph.add_node("worker",worker)
graph.add_node("reducer",reducer)

graph.add_edge(START,"router")
graph.add_conditional_edges("router",routernext,{"research_node": "research_node", "orchestrator": "orchestrator"})
# orchestrator node, call the fanout function to decide what happens next, and route execution to one or more worker nodes based on its result.”
graph.add_edge("research_node","orchestrator")
graph.add_conditional_edges("orchestrator",fanout,["worker"])
graph.add_edge("worker","reducer")
graph.add_edge("reducer",END)

app = graph.compile()




    

