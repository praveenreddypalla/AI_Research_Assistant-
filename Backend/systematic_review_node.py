"""
Systematic Review Node
======================
PRISMA-style 3-step pipeline for the 'Systematic review' workflow:

1. IDENTIFICATION: broad concurrent search (web + arXiv) gathering candidate
   documents and evidence.
2. SCREENING: a screening LLM call evaluates every identified document and
   keeps only the highly relevant ones (with an include/exclude reason each).
3. EXTRACTION & SYNTHESIS: builds the extraction matrix and writes a
   PRISMA-informed Markdown review using only the surviving documents.
"""

import asyncio
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core_engine.llm_router import LLMRouter
from core_engine.nodes.actions.web_searcher import execute_search_action
from core_engine.nodes.synthesizer import _build_extraction_matrix
from core_engine.utilities.arxiv_search import arxiv_researcher
from core_engine.utilities.progress import emit_progress

# Safety cap for the screening prompt context.
MAX_SCREENING_CHARS = 60000


# --- 1. PYDANTIC SCHEMAS ---
class ScreeningDecision(BaseModel):
    """One PRISMA screening verdict for a single identified document."""
    source: str = Field(description="Title or URL identifying the document exactly as it appears in the findings")
    include: bool = Field(description="True to include the document in the review, False to exclude it")
    reason: str = Field(description="One-sentence reason for the decision")


class ScreeningResult(BaseModel):
    """Structured output for the screening stage."""
    decisions: List[ScreeningDecision] = Field(
        default_factory=list,
        description="One decision per distinct document identified in the findings",
    )


# --- 2. THE SYSTEM PROMPTS ---
# NOTE: Keep these prompts free of stray literal curly braces (LangChain template rules).
SCREENING_PROMPT = """
You are the screening stage of a PRISMA-style systematic review. Today's date is {date}.

RESEARCH QUESTION: {query}

CANDIDATE FINDINGS (identified in the broad search):
{findings}

For every distinct document in the findings, decide whether to INCLUDE it
(highly relevant, substantive evidence for the research question) or EXCLUDE it
(off-topic, low quality, duplicate, or tangential). Provide a one-sentence
reason for each decision. Identify each document by its title or URL exactly
as it appears in the findings.
"""

REVIEW_PROMPT = """
You are a Senior Systematic Review Author. Today's date is {date}.
Write a PRISMA-informed systematic review in professional academic Markdown. Do NOT use markdown code fences.

RESEARCH QUESTION: {query}

SCREENING SUMMARY:
- Documents identified: {identified}
- Documents included after screening: {included}
- Documents excluded: {excluded}

INCLUDED SOURCES (use ONLY these):
{included_sources}

EVIDENCE FROM THE BROAD SEARCH:
{findings}

Structure the review with these sections: Background, Methods (describe the
search and screening process using the numbers above), Included Studies,
Synthesis of Findings, Limitations, and References.
Cite claims with numbered brackets [1], [2] mapped to the References section.
Use ONLY the included sources; never cite excluded or invented sources.
"""


def _filter_matrix(matrix: list, included_sources: List[str]) -> list:
    """Keep extraction records matching an included source (fuzzy substring match)."""
    if not matrix or not included_sources:
        return matrix
    lowered = [s.strip().lower() for s in included_sources if s and s.strip()]
    surviving = []
    for record in matrix:
        title = str(record.get("title", "")).strip().lower()
        source = str(record.get("url_or_source", "")).strip().lower()
        for candidate in lowered:
            if (title and (candidate in title or title in candidate)) or (
                source and (candidate in source or source in candidate)
            ):
                surviving.append(record)
                break
    # Never let over-aggressive matching wipe out the grid entirely.
    return surviving if surviving else matrix


# --- 3. THE LANGGRAPH NODE ---
async def systematic_review_node(state: dict):
    """Execute the 3-step PRISMA pipeline and finish the graph."""
    query = state.get("current_section") or state.get("query", "")
    print(f"[Systematic Review] Starting PRISMA-style pipeline for: '{query}'")
    router = LLMRouter()
    date = datetime.now().strftime("%Y-%m-%d")

    # --- STEP 1: IDENTIFICATION (broad concurrent search) ---
    emit_progress(state, "[Systematic Review] Step 1: Running the broad identification search.")

    async def safe_web_search() -> str:
        try:
            return await execute_search_action(gap=query, query=query)
        except Exception as e:
            print(f"[Systematic Review] Web search failed ({e}).")
            return ""

    async def safe_arxiv_search() -> str:
        try:
            return await arxiv_researcher.ainvoke({"query": query})
        except Exception as e:
            print(f"[Systematic Review] arXiv search failed ({e}).")
            return ""

    web_findings, arxiv_findings = await asyncio.gather(safe_web_search(), safe_arxiv_search())

    findings = ""
    if web_findings:
        findings += f"--- WEB SEARCH RESULTS ---\n{web_findings}\n\n"
    if arxiv_findings:
        findings += f"--- ARXIV RESULTS ---\n{arxiv_findings}\n\n"

    if not findings.strip():
        fallback = (
            "The broad identification search returned no usable documents, so a "
            "systematic review could not be completed. Please retry, or refine the research question."
        )
        return {
            "completed_sections": [fallback],
            "extraction_matrix": [],
            "research_complete": True,
            "research_history": "",
        }

    # --- STEP 2: SCREENING ---
    emit_progress(state, "[Systematic Review] Step 2: Screening documents for relevance.")
    included_sources: List[str] = []
    identified_count = 0
    try:
        structured_llm = router.fast_model.with_structured_output(ScreeningResult)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SCREENING_PROMPT),
            ("human", "Screen the candidate documents now."),
        ])
        screening: ScreeningResult = await (prompt | structured_llm).ainvoke({
            "date": date,
            "query": query,
            "findings": findings[:MAX_SCREENING_CHARS],
        })
        decisions = screening.decisions if screening else []
        identified_count = len(decisions)
        included_sources = [d.source for d in decisions if d.include]
        for decision in decisions:
            verdict = "INCLUDE" if decision.include else "EXCLUDE"
            print(f"[Systematic Review] {verdict}: {decision.source} ({decision.reason})")
    except Exception as e:
        # Fail open: keep all identified documents rather than aborting the review.
        print(f"[Systematic Review] Screening failed ({e}). Keeping all identified documents.")

    if included_sources:
        included_count = len(included_sources)
        excluded_count = max(identified_count - included_count, 0)
        included_sources_text = "\n".join(f"- {source}" for source in included_sources)
    else:
        included_count = identified_count
        excluded_count = 0
        included_sources_text = "All identified documents (the screening stage was unavailable or excluded nothing usable)."

    emit_progress(
        state,
        f"[Systematic Review] Screening complete: {included_count or 'all'} of {identified_count or 'the identified'} document(s) included.",
    )

    # --- STEP 3: EXTRACTION & SYNTHESIS on the surviving documents ---
    emit_progress(state, "[Systematic Review] Step 3: Extracting data and synthesizing the review.")
    matrix = _build_extraction_matrix(findings, state)
    matrix = _filter_matrix(matrix, included_sources)

    try:
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", REVIEW_PROMPT),
                ("human", "Write the systematic review now."),
            ])
            | router.main_model
            | StrOutputParser()
        )
        review = await chain.ainvoke({
            "date": date,
            "query": query,
            "identified": identified_count or "unknown",
            "included": included_count or "unknown",
            "excluded": excluded_count,
            "included_sources": included_sources_text,
            "findings": findings,
        })
    except Exception as e:
        print(f"[Systematic Review] Synthesis failed ({e}). Returning screening summary.")
        review = (
            "## Systematic Review (partial)\n\n"
            "Synthesis was unavailable. Screening summary of included sources:\n\n"
            f"{included_sources_text}"
        )

    emit_progress(state, "[Systematic Review] Review complete.")
    return {
        "completed_sections": [review],
        "extraction_matrix": matrix,
        "research_complete": True,
        "research_history": "",
    }
