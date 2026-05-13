"""
Edit capture — stores operator edits into SQLite.

Receives a list of before/after section pairs from the /feedback endpoint,
computes lightweight diff stats, and persists each changed section as an
Edit row.  The classifier and pattern extractor run afterwards (in a
background task) against these stored rows.

Why store first, process later?
  The /feedback endpoint must respond immediately.  Groq LLM calls take
  1–3 seconds each; with 5 sections that would be 5–15 seconds of blocking.
  Storing the raw edit is fast (one SQL INSERT), so we return edit_ids right
  away and let process_edit_async run in the background.
"""

import json
import logging
import re
from datetime import datetime

from sqlmodel import Session

from python_service.db.models import Draft, Edit
from python_service.db.session import engine

logger = logging.getLogger(__name__)


def _diff_stats(original: str, edited: str) -> dict:
    """Compute lightweight character-level diff statistics."""
    added = max(0, len(edited) - len(original))
    removed = max(0, len(original) - len(edited))
    delta_ratio = round(abs(len(edited) - len(original)) / max(len(original), 1), 3)
    return {
        "addedChars": added,
        "removedChars": removed,
        "deltaRatio": delta_ratio,
        "originalLen": len(original),
        "editedLen": len(edited),
    }


def _extract_citations(text: str) -> list[str]:
    """Pull [E1], [E2] … citation labels from text."""
    return list({f"E{n}" for n in re.findall(r"\[E(\d+)\]", text)})


def store_edits(
    draft_id: str,
    edits: list[dict],
    operator_id: str = "op_harvey",
) -> list[str]:
    """
    Persist each operator edit as an Edit row in SQLite.

    Args:
        draft_id: the draft being edited.
        edits:    list of dicts — each must have:
                    section_id, section_title, original_text, edited_text
        operator_id: which operator submitted the edits.

    Returns:
        List of edit_ids (one per stored edit), in input order.
        Sections where original_text == edited_text are silently skipped.
    """
    with Session(engine) as session:
        draft = session.get(Draft, draft_id)
        if draft is None:
            raise ValueError(f"Draft {draft_id!r} not found")

        document_id = draft.document_id
        draft_type = draft.draft_type

        # Pull document_type from the draft's processing meta
        try:
            meta = json.loads(draft.processing_meta_json or "{}")
            document_type = meta.get("documentType", "unknown")
        except (json.JSONDecodeError, AttributeError):
            document_type = "unknown"

    edit_ids: list[str] = []
    stored = 0

    with Session(engine) as session:
        for item in edits:
            original = item.get("original_text", "").strip()
            edited = item.get("edited_text", "").strip()

            # Skip no-op edits
            if original == edited:
                continue

            citations = _extract_citations(edited) or _extract_citations(original)

            edit = Edit(
                draft_id=draft_id,
                document_id=document_id,
                document_type=document_type,
                draft_type=draft_type,
                section_id=item.get("section_id", ""),
                section_title=item.get("section_title", ""),
                original_text=original,
                edited_text=edited,
                diff_summary_json=json.dumps(_diff_stats(original, edited)),
                cited_evidence_json=json.dumps(citations),
                surrounding_context_json=json.dumps(
                    item.get("surrounding_context", {})
                ),
                operator_id=operator_id,
                edited_at=datetime.utcnow(),
                pattern_extraction_status="pending",
            )
            session.add(edit)
            session.flush()   # assigns edit_id before commit
            edit_ids.append(edit.edit_id)
            stored += 1

        session.commit()

    logger.info("Stored %d edit(s) for draft %s", stored, draft_id)
    return edit_ids
