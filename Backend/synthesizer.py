"""
Synthesizer Node
================
Handles the 'Reduce' phase of the architecture.
Once the QA node (Gap Analyzer) declares research complete, or the safety loop limit is reached,
this node consumes the raw, aggregated tool findings and drafts a final, academic-grade Markdown section.
"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core_engine.llm_router import LLMRouter
from core_engine.session_memory import format_messages_for_prompt
from core_engine.utilities.progress import emit_progress

# Safety cap for the matrix extraction prompt context.
MAX_MATRIX_FINDINGS_CHARS = 60000


# --- 0. EXTRACTION MATRIX SCHEMAS (Elicit-style grid) ---
class ExtractionRecord(BaseModel):
    """One structured row of the extraction matrix (one per source document)."""
    title: str = Field(description="Title of the source document or page")
    url_or_source: str = Field(description="URL, arXiv link, or document/file name of the source")
    abstract_summary: str = Field(description="Summary of the document in at most 2 sentences")
    methodology: str = Field(description="Methodology or approach described, or 'N/A' if not applicable")
    sample_size: str = Field(description="Sample size if reported, otherwise 'N/A'")
    key_findings: str = Field(description="Most important findings from this document")


class ExtractionMatrix(BaseModel):
    """Structured output schema for the matrix extraction call."""
    records: List[ExtractionRecord] = Field(
        default_factory=list,
        description="One record per distinct source document found in the findings",
    )


# NOTE: Keep this prompt free of stray literal curly braces (LangChain template rules).
MATRIX_PROMPT = """
You are a Research Data Extraction Specialist. Today's date is {date}.
From the raw research findings below, identify every distinct source document
(web page, arXiv paper, or uploaded PDF) and produce one structured record per document.

RESEARCH FINDINGS:
{findings}

For each distinct document extract:
- title: the document or page title.
- url_or_source: the URL, arXiv link, or document/file name.
- abstract_summary: a summary of the document in at most 2 sentences.
- methodology: the methodology or approach described. Use 'N/A' if not applicable.
- sample_size: the sample size if reported. Use 'N/A' if not applicable.
- key_findings: the most important findings from that document.

RULES:
1. Only include documents that actually appear in the findings. Never invent sources.
2. Deduplicate documents that appear multiple times.
3. Every field must be a plain string.
"""


def _build_extraction_matrix(findings: str, state: dict) -> list:
    """Run the structured matrix extraction. Fail-safe: returns [] on any error."""
    if not findings or not findings.strip() or findings.strip().lower().startswith("no findings"):
        return []

    emit_progress(state, "[Synthesizer] Building the structured extraction matrix.")
    try:
        router = LLMRouter()
        structured_llm = router.fast_model.with_structured_output(ExtractionMatrix)
        prompt = ChatPromptTemplate.from_messages([
            ("system", MATRIX_PROMPT),
            ("human", "Extract the structured records now."),
        ])
        chain = prompt | structured_llm
        matrix: ExtractionMatrix = chain.invoke({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "findings": findings[:MAX_MATRIX_FINDINGS_CHARS],
        })
        records = [record.model_dump() for record in (matrix.records if matrix else [])]
        print(f"[Synthesizer] Extraction matrix built with {len(records)} record(s).")
        return records
    except Exception as e:
        print(f"[Synthesizer] Matrix extraction failed ({e}). Returning empty matrix.")
        return []

# --- 1. THE SYSTEM PROMPT ---
SYNTHESIZER_PROMPT = """
You are a Senior Academic Researcher and Technical Writer. Today's date is {date}.
Your objective is to write a highly detailed, comprehensive section of a final report based on the provided research findings.

ORIGINAL OVERALL QUERY: {query}
SPECIFIC SECTION TITLE: {section_title}
SPECIFIC SECTION QUESTION: {section_question}

PRIOR CONVERSATION CONTEXT (earlier questions and reports from this session; may be empty):
{conversation}

RESEARCH FINDINGS TO SYNTHESIZE:
{findings}

GUIDELINES:
1. Write in a professional, academic markdown format. Do NOT use markdown code blocks (like ```markdown), just return the raw text.
2. Answer the specific section question directly using ONLY the provided research findings.
3. Do NOT hallucinate or add outside information that is not supported by the findings.
4. Include references to the source URLs or Document Names for all data. Use numbered brackets [1], [2], etc., immediately after the claim.
5. Provide a \"References\" section at the very bottom of your output listing the sources you used.
6. If the section question is a conversational follow-up, you may also draw on the prior conversation context above; prefer the new research findings whenever they conflict with older answers.

CRITICAL CITATION RULES:
1. NEVER use raw artifact names or search queries in your citations (e.g., do NOT output [Web Search Summary: '...' - 1]).
2. You must re-number all citations sequentially using standard academic brackets (e.g., [1], [2], [3]).
3. Ensure the 'References' section at the bottom perfectly maps these clean numbers to their respective URLs.
"""

# --- 2. THE LANGGRAPH NODE ---
def synthesizer_node(state: dict):
    """Consumes the contextual history and generates formatted Markdown."""
    user_query = state.get("query", "Unknown query")
    section_title = state.get("current_section_title", "General Research")
    section_question = state.get("current_section", "Summarize findings")
    findings = state.get("research_history", "No findings provided.")
    conversation = format_messages_for_prompt(state.get("chat_history", []) or [])
    emit_progress(state, f"[Synthesizer] Drafting the final Markdown report for {section_title}.")
    
    print(f"\u270d\ufe0f [Synthesizer] Writing report section: '{section_title}'...")
    
    # We use the Main Model here because processing thousands of characters of 
    # aggregated web/RAG data requires a highly robust context window and strong linguistic synthesis.
    router = LLMRouter()
    llm = router.main_model
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIZER_PROMPT),
        ("human", "Synthesize the findings and write the final section.")
    ])
    
    # Note: We pivot away from structured JSON here. We use StrOutputParser() 
    # because we want continuous, unstructured raw Markdown text as the final artifact.
    chain = prompt | llm | StrOutputParser()
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    final_draft = chain.invoke({
        "date": current_date, 
        "query": user_query,
        "section_title": section_title,
        "section_question": section_question,
        "conversation": conversation,
        "findings": findings
    })
    
    print(f"\u2705 [Synthesizer] Section '{section_title}' successfully written!")
    
    completed_sections = state.get("completed_sections", [])
    completed_sections.append(final_draft)
    
    # DUAL OUTPUT: alongside the Markdown report, extract the Elicit-style
    # structured matrix (one record per source document) before the ephemeral
    # research history is purged below.
    extraction_matrix = _build_extraction_matrix(findings, state)
    
    emit_progress(state, f"[Synthesizer] Finished writing {section_title}.")
    
    return {
        "completed_sections": completed_sections,
        "extraction_matrix": extraction_matrix,
        # Purge the ephemeral research history to prevent memory leaks across loops
        "research_history": "" 
    }


# --- 3. THE HEAVY SYNTHESIS NODE ('Report' workflow) ---
HEAVY_SYNTHESIZER_PROMPT = """
You are a Senior Academic Researcher writing a long-form literature review. Today's date is {date}.

ORIGINAL QUERY: {query}

PRIOR CONVERSATION CONTEXT (may be empty):
{conversation}

RESEARCH FINDINGS TO SYNTHESIZE:
{findings}

Write a comprehensive, long-form Markdown literature review. Do NOT use markdown code fences. Requirements:
1. A title and a short abstract.
2. Clearly organized thematic sections with level-2 headings.
3. Deep analysis: compare and contrast sources, discuss methodologies, highlight open problems and future directions.
4. Numbered citations [1], [2] immediately after claims, using ONLY the provided findings.
5. A final References section mapping the numbers to source URLs or document names.
Never invent sources or facts that are not in the findings.
"""


def heavy_synthesizer_node(state: dict):
    """'Report' workflow: long-form Markdown literature review, no extraction matrix."""
    user_query = state.get("query", "Unknown query")
    findings = state.get("research_history", "No findings provided.")
    conversation = format_messages_for_prompt(state.get("chat_history", []) or [])
    emit_progress(state, "[Heavy Synthesizer] Drafting the long-form literature review.")
    print("[Heavy Synthesizer] Writing long-form literature review...")

    router = LLMRouter()
    llm = router.main_model

    prompt = ChatPromptTemplate.from_messages([
        ("system", HEAVY_SYNTHESIZER_PROMPT),
        ("human", "Write the long-form literature review now."),
    ])
    chain = prompt | llm | StrOutputParser()

    final_draft = chain.invoke({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "query": user_query,
        "conversation": conversation,
        "findings": findings,
    })

    completed_sections = state.get("completed_sections", [])
    completed_sections.append(final_draft)
    emit_progress(state, "[Heavy Synthesizer] Finished the literature review.")

    return {
        "completed_sections": completed_sections,
        # The 'report' workflow intentionally omits the extraction matrix.
        "extraction_matrix": [],
        "research_history": "",
    }
