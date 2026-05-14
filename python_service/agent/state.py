"""
python_service/agent/state.py

Defines the shared "memory" of the drafting agent — DraftingState.

Every node in the LangGraph reads from this state and writes back to it.
Think of it like a whiteboard that the whole team can see and update.

The three key TypedDicts below represent the three stages of information
that flow through the agent:
  SectionPlan   — what the planner decides to write
  SectionDraft  — what the executor actually wrote
  CritiqueItem  — what the critic found wrong
"""

from typing import Annotated, Optional

from typing_extensions import TypedDict


def _merge_section_drafts(
    existing: list["SectionDraft"],
    incoming: list["SectionDraft"],
) -> list["SectionDraft"]:
    """
    Custom reducer for section_drafts.

    When a Refiner re-runs an Executor for a weak section, the new draft
    must REPLACE the old one, not append alongside it. This reducer keeps
    the latest version of each section_id.

    Why not operator.add?
      operator.add just concatenates. If sec_2 is rewritten, we'd end up
      with two sec_2 entries — one bad, one good. The assembler would have
      to guess which one to use. This reducer resolves it here instead.

    Order: existing entries first, incoming overwrites by section_id.
    """
    merged: dict[str, "SectionDraft"] = {d["section_id"]: d for d in existing}
    for draft in incoming:
        merged[draft["section_id"]] = draft   # newer version wins
    # Return in section_id order so the final output is deterministic
    return sorted(merged.values(), key=lambda d: d["section_id"])


# ── SectionPlan ───────────────────────────────────────────────────────────────
# Produced by the Planner node.
# One SectionPlan per section of the draft (usually 4–7).
#
# The key insight: each section gets its OWN retrieval_query.
# This means "Compensation" retrieves salary/bonus evidence,
# "Termination" retrieves termination clause evidence — focused, not mixed.

class SectionPlan(TypedDict):
    section_id: str               # "sec_1", "sec_2", etc.
    title: str                    # "Compensation Structure"
    brief: str                    # one sentence: what this section will cover
    retrieval_query: str          # the query sent to the evidence retriever
    target_length_words: int      # how long the section should be (100–300)


# ── SectionDraft ──────────────────────────────────────────────────────────────
# Produced by the Executor node (one per section).
# Contains the actual written text plus quality signals.

class SectionDraft(TypedDict):
    section_id: str
    title: str
    content: str                  # the drafted text with inline [E1] citations
    cited_evidence: list[str]     # ["E1", "E3"] — which evidence IDs were used
    grounding_score: float        # 0.0–1.0, from the NLI verifier
    confidence: str               # "HIGH" | "MEDIUM" | "LOW" (mirrors grounding)
    evidence_items: list[dict]    # the raw evidence dicts, kept for the critic


# ── CritiqueItem ──────────────────────────────────────────────────────────────
# Produced by the Critic node.
# One CritiqueItem per *weak* section — sections that are fine produce nothing.

class CritiqueItem(TypedDict):
    section_id: str
    weakness_type: str            # "UNGROUNDED" | "INCOMPLETE" | "STYLE_VIOLATION"
    description: str              # plain-English explanation of the problem
    suggested_fix: str            # concrete instruction for the Refiner


# ── DraftingState ─────────────────────────────────────────────────────────────
# The main whiteboard — passed through every node in the graph.
#
# `total=False` means every field is optional. This matters because early
# nodes (planner) haven't filled in the later fields (final_sections) yet.
#
# The special `Annotated[list[SectionDraft], operator.add]` on section_drafts
# is a LangGraph convention: when multiple parallel executor nodes each write
# one SectionDraft, LangGraph *merges* the lists automatically instead of
# one node overwriting the other's result.

class DraftingState(TypedDict, total=False):
    # ── Inputs (set once at the start, never change) ──────────────────────────
    document_id: str
    document_title: str
    document_type: str
    query: str
    draft_type: str

    # ── Planner output ────────────────────────────────────────────────────────
    plan: list[SectionPlan]

    # ── Executor outputs ──────────────────────────────────────────────────────
    # Custom reducer: parallel executors append their section; refined sections
    # overwrite the original by section_id (latest version wins).
    # See _merge_section_drafts above for the full explanation.
    section_drafts: Annotated[list[SectionDraft], _merge_section_drafts]

    # ── Critic output ─────────────────────────────────────────────────────────
    critique: list[CritiqueItem]
    iteration: int                # how many refinement loops have run (max 3)

    # ── Patterns (retrieved once, shared across all executors) ────────────────
    patterns: list[dict]

    # ── Executor private slot ─────────────────────────────────────────────────
    # Set by the dispatcher via Send() — each executor sees its own section here.
    # Never written to directly; always comes from the fan-out dispatch.
    current_section: Optional[SectionPlan]

    # ── Final output (assembled after critic approves) ────────────────────────
    final_draft_id: str
    final_title: str
    final_sections: list[dict]    # the section format the API response expects
    final_grounding_score: float
    final_judge_scores: dict
    final_adherence: dict
    trace_id: Optional[str]
