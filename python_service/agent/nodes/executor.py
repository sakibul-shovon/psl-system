"""
python_service/agent/nodes/executor.py

The Executor node — drafts ONE section of the document.

This node runs in PARALLEL — one instance per section in the plan.
Each instance handles a completely independent section and writes
its result back to the shared `section_drafts` list in state.

Pipeline for one section:
  1. Read its assigned SectionPlan from state["current_section"]
  2. Retrieve focused evidence using that section's retrieval_query
  3. Build a section-specific Gemini prompt
  4. Generate the section content
  5. Run NLI grounding verification on that section only
  6. Return a SectionDraft appended to section_drafts

WHY per-section retrieval?
  In the old pipeline, all sections shared one pool of top-5 chunks.
  If the query is "compensation and termination", the termination sections
  might retrieve evidence about salary and miss the termination clauses.
  Here each section retrieves its own focused evidence — tighter, more relevant.
"""

import json
import logging

from groq import Groq

from python_service.agent.state import DraftingState, SectionDraft
from python_service.config import settings
from python_service.generation.grounding import verify_draft
from python_service.observability.langfuse_client import observe
from python_service.retrieval.evidence import package_evidence
from python_service.retrieval.hybrid import retrieve

logger = logging.getLogger(__name__)

# ── Section-focused prompt ─────────────────────────────────────────────────────
# Simpler than the full draft prompt — we ask for ONE section only.
# The planner already decided the title; we pass it here so Gemini stays on topic.

_SECTION_PROMPT = """\
You are a legal document analyst for Pearson Specter Litt.

Write EXACTLY ONE section of a legal draft based ONLY on the evidence provided.

SECTION TITLE: {title}
SECTION BRIEF: {brief}
DOCUMENT: {document_title}
TARGET LENGTH: approximately {target_words} words

RULES:
1. Every factual claim MUST end with an inline citation [E1], [E2], etc.
2. Only use evidence provided below — do not invent facts.
3. If evidence is insufficient, write [INSUFFICIENT EVIDENCE: reason].
4. Dates, dollar amounts, party names must be quoted exactly from the evidence.
5. Do NOT add section headings inside the content string.

{pattern_block}

EVIDENCE:
{evidence_block}

Return ONLY valid JSON, no markdown:
{{
  "section_id": "{section_id}",
  "title": "{title}",
  "content": "the drafted paragraph(s) with [E1] inline citations",
  "citedEvidence": ["E1", "E2"],
  "confidence": "HIGH | MEDIUM | LOW"
}}
"""

_PATTERN_BLOCK_HEADER = "LEARNED STYLE PATTERNS (apply these to your writing):\n"


def _build_pattern_block(patterns: list[dict]) -> str:
    """Format patterns as numbered instructions if any exist."""
    if not patterns:
        return ""
    lines = [_PATTERN_BLOCK_HEADER]
    for i, p in enumerate(patterns[:5], start=1):   # cap at 5 most relevant
        lines.append(f"  {i}. [{p.get('rule_type','style').upper()}] {p.get('description','')}")
    return "\n".join(lines)


def _build_evidence_block(evidence_items) -> tuple[str, dict]:
    """
    Format evidence items as [E1]...[En] blocks for the prompt.
    Also returns a dict {evidence_id: content} needed by the grounding verifier.
    """
    blocks = []
    evidence_map = {}
    for item in evidence_items:
        label = item.evidence_id
        evidence_map[label] = item.content
        ocr_note = f" | OCR: {item.confidence_tier}" if item.has_low_conf_regions else ""
        blocks.append(
            f"[{label}] {item.breadcrumb}{ocr_note}\n{item.content}"
        )
    return "\n\n".join(blocks), evidence_map


@observe(name="groq-executor-section")
def _call_gemini_for_section(prompt: str) -> dict:
    """Call Groq Llama 3.3 70B in JSON mode for one section."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")
    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a legal drafting assistant. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=2048,
    )
    return json.loads(response.choices[0].message.content)


def executor_node(state: DraftingState) -> dict:
    """
    LangGraph node — drafts one section and returns it as a SectionDraft.

    Returns {"section_drafts": [one_section_draft]}.
    Because section_drafts uses `operator.add`, LangGraph appends this
    to the shared list rather than replacing it — so all parallel executors
    contribute their section without overwriting each other.
    """
    section    = state["current_section"]
    doc_id     = state["document_id"]
    doc_title  = state.get("document_title", "")
    patterns   = state.get("patterns", [])

    section_id    = section["section_id"]
    title         = section["title"]
    brief         = section["brief"]
    query         = section["retrieval_query"]   # focused query for THIS section
    target_words  = section["target_length_words"]

    logger.info("Executor [%s]: retrieving evidence for %r", section_id, title)

    # ── Step 1: Focused evidence retrieval ────────────────────────────────────
    # Each section runs its OWN retrieval with its own query.
    # This is what separates Phase B from the old pipeline.
    retrieval_result = retrieve(query, doc_id)

    if not retrieval_result.sufficient:
        # Not enough evidence for this section — return a low-confidence draft
        logger.warning("Executor [%s]: insufficient evidence", section_id)
        return {
            "section_drafts": [
                SectionDraft(
                    section_id=section_id,
                    title=title,
                    content=f"[INSUFFICIENT EVIDENCE: no relevant chunks found for '{title}']",
                    cited_evidence=[],
                    grounding_score=0.0,
                    confidence="LOW",
                    evidence_items=[],
                )
            ]
        }

    evidence_items = package_evidence(retrieval_result.evidence, document_title=doc_title)

    # ── Step 2: Build and send the section prompt ─────────────────────────────
    evidence_block, evidence_map = _build_evidence_block(evidence_items)
    pattern_block = _build_pattern_block(patterns)

    prompt = _SECTION_PROMPT.format(
        section_id=section_id,
        title=title,
        brief=brief,
        document_title=doc_title,
        target_words=target_words,
        evidence_block=evidence_block,
        pattern_block=pattern_block,
    )

    try:
        raw = _call_gemini_for_section(prompt)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Executor [%s]: Gemini failed: %s", section_id, exc)
        raw = {
            "section_id":    section_id,
            "title":         title,
            "content":       f"[GENERATION ERROR: {exc}]",
            "citedEvidence": [],
            "confidence":    "LOW",
        }

    content         = raw.get("content", "")
    cited_evidence  = raw.get("citedEvidence", [])

    # ── Step 3: NLI grounding check for this section only ────────────────────
    # We pass a single-element list because verify_draft expects a list of sections.
    section_as_dict = {"section_id": section_id, "title": title, "content": content}
    grounding = verify_draft([section_as_dict], evidence_map)

    # Map grounding score to a confidence label
    if grounding.grounding_score >= 0.75:
        confidence = "HIGH"
    elif grounding.grounding_score >= 0.50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    logger.info(
        "Executor [%s] done: grounding=%.3f (%s), citations=%s",
        section_id, grounding.grounding_score, confidence, cited_evidence,
    )

    return {
        "section_drafts": [
            SectionDraft(
                section_id=section_id,
                title=title,
                content=content,
                cited_evidence=cited_evidence,
                grounding_score=grounding.grounding_score,
                confidence=confidence,
                evidence_items=[e.to_dict() for e in evidence_items],
            )
        ]
    }
