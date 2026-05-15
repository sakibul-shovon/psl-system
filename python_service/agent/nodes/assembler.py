"""
python_service/agent/nodes/assembler.py

The Assembler node — the last node before END.

Runs after the Critic approves all sections (or max iterations are exhausted).
It collects all SectionDrafts, computes the overall grounding score, runs the
independent judge, saves a Draft row to SQLite, and writes the final fields
into state so the API route can build its response from them.

Think of it as the "publishing" step: the agent is done deliberating, and
the assembler packages the result for the outside world.
"""

import json
import logging
import uuid

from sqlmodel import Session

from python_service.agent.state import DraftingState
from python_service.db.models import Draft
from python_service.db.session import engine
from python_service.evaluation.adherence_checker import check_adherence
from python_service.evaluation.draft_judge import judge_draft

logger = logging.getLogger(__name__)


def assembler_node(state: DraftingState) -> dict:
    """
    LangGraph node — assembles the final draft and saves it to SQLite.

    Reads:
      state["section_drafts"]  — all completed SectionDraft objects
      state["patterns"]        — to check adherence
      state["document_id"]     — to save the Draft row

    Writes back:
      final_draft_id, final_title, final_sections,
      final_grounding_score, final_judge_scores, final_adherence
    """
    drafts      = state.get("section_drafts", [])
    document_id = state["document_id"]
    document_title = state.get("document_title", "")
    draft_type  = state.get("draft_type", "case_fact_summary")
    patterns    = state.get("patterns", [])
    query       = state.get("query", "")

    # ── Step 1: Sort sections into plan order ─────────────────────────────────
    # The custom reducer sorts alphabetically by section_id ("sec_1" < "sec_2"),
    # but verify the order explicitly in case of any edge cases.
    sorted_drafts = sorted(drafts, key=lambda d: d["section_id"])

    logger.info(
        "Assembler: packaging %d section(s) for document %r",
        len(sorted_drafts), document_id,
    )

    # ── Step 2: Convert SectionDraft → API section format ────────────────────
    # The API response uses camelCase keys matching the old pipeline schema.
    # We keep the same format so the API route doesn't need to change.
    final_sections = [
        {
            "section_id":    d["section_id"],
            "section_title": d["title"],
            "content":       d["content"],
            "evidence_ids":  d["cited_evidence"],
            "confidence":    d["confidence"],
            "grounding_score": d["grounding_score"],
            # Full evidence items kept so the UI can expand [E1] inline citations.
            # Keyed by evidence_id so the UI can look up "E1" → breadcrumb + content.
            "evidence_map": {
                ev.get("evidence_id", ""): {
                    "breadcrumb":       ev.get("breadcrumb", ""),
                    "content":          ev.get("content", "")[:600],
                    "confidence_tier":  ev.get("confidence_tier", "HIGH"),
                }
                for ev in d.get("evidence_items", [])
            },
        }
        for d in sorted_drafts
    ]

    # ── Step 3: Overall grounding score = average of per-section scores ───────
    # Each Executor ran its own NLI check; we average here rather than re-running
    # NLI on the entire assembled document (which would be slow and redundant).
    if sorted_drafts:
        overall_grounding = round(
            sum(d["grounding_score"] for d in sorted_drafts) / len(sorted_drafts), 3
        )
    else:
        overall_grounding = 0.0

    # ── Step 4: Adherence check — did sections follow learned patterns? ───────
    sections_as_dicts = [
        {"section_id": d["section_id"], "title": d["title"], "content": d["content"]}
        for d in sorted_drafts
    ]
    adherence = check_adherence(sections_as_dicts, patterns)

    # ── Step 5: Independent judge score ──────────────────────────────────────
    # Collect all evidence dicts from all sections for the judge
    all_evidence: list[dict] = []
    seen_ids: set[str] = set()
    for d in sorted_drafts:
        for ev in d.get("evidence_items", []):
            ev_id = ev.get("evidence_id", "")
            if ev_id not in seen_ids:
                all_evidence.append(ev)
                seen_ids.add(ev_id)

    judge_scores = judge_draft(sections_as_dicts, all_evidence)

    # ── Step 6: Save Draft row to SQLite ──────────────────────────────────────
    draft_id = str(uuid.uuid4())
    applied_pattern_ids = [p["pattern_id"] for p in patterns]

    draft_title = f"{document_title} — {query[:60]}" if document_title else query[:80]

    with Session(engine) as session:
        draft_row = Draft(
            draft_id=draft_id,
            document_id=document_id,
            draft_type=draft_type,
            title=draft_title,
            sections_json=json.dumps(final_sections),
            grounding_score=overall_grounding,
            warnings_json=json.dumps([]),
            judge_scores_json=json.dumps(judge_scores),
            applied_pattern_ids_json=json.dumps(applied_pattern_ids),
            processing_meta_json=json.dumps({
                "generationModel":  "groq/llama-3.3-70b-versatile",
                "judgeModel":       "groq/llama-3.3-70b-versatile",
                "retrievalMethod":  "dense+bm25+rerank (per-section)",
                "patternsApplied":  len(patterns),
                "adherenceScore":   adherence["adherence_score"],
                "documentType":     state.get("document_type", "unknown"),
                "draftType":        draft_type,
                "agentIterations":  state.get("iteration", 0),
                "query":            query,  # stored so /feedback can build episodic memory
            }),
        )
        session.add(draft_row)
        session.commit()

    logger.info(
        "Assembler: draft %r saved — grounding=%.3f, judge_overall=%s, patterns=%d",
        draft_id, overall_grounding,
        judge_scores.get("overall", "?"), len(patterns),
    )

    return {
        "final_draft_id":       draft_id,
        "final_title":          draft_title,
        "final_sections":       final_sections,
        "final_grounding_score": overall_grounding,
        "final_judge_scores":   judge_scores,
        "final_adherence":      adherence,
    }
