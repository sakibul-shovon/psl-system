"""
Background edit processor — runs classify → extract → gate → store for one edit.

Called by the FastAPI background task after /feedback stores the raw edits.
Each edit is processed independently; one failure doesn't block the others.

Flow:
  1. Load Edit row from SQLite
  2. Classify the edit type (Groq 70B)
  3. Extract a generalized pattern rule (Groq 70B)
  4. Quality gate (reject low-confidence / too-short rules)
  5a. Pass → save Pattern to SQLite + embed description → upsert to Qdrant
  5b. Fail → mark edit as 'noise', nothing stored
  6. Update Edit row with classification result + pattern_id (if any)
"""

import json
import logging
import uuid

from sqlmodel import Session

from python_service.db.models import Edit, Pattern
from python_service.db.session import engine
from python_service.edit_loop.classifier import classify_edit
from python_service.edit_loop.pattern_extractor import extract_pattern
from python_service.evaluation.pattern_quality_gate import passes_quality_gate
from python_service.embedder import embed_one
from python_service.vector.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)


def process_edit(edit_id: str) -> None:
    """
    Run the full classify → extract → store pipeline for one edit.

    This function is safe to call in a FastAPI BackgroundTask — all
    exceptions are caught and logged; the edit row is always updated
    with the final status so the caller can inspect results.
    """
    with Session(engine) as session:
        edit = session.get(Edit, edit_id)
        if edit is None:
            logger.error("Edit %s not found — skipping", edit_id)
            return
        original = edit.original_text
        edited = edit.edited_text
        document_type = edit.document_type
        draft_type = edit.draft_type

    try:
        # ── Step 1: Classify ─────────────────────────────────────────────────
        classification = classify_edit(original, edited)
        edit_type = classification["edit_type"]

        # ── Step 2: Extract pattern (None for NOISE) ─────────────────────────
        pattern_dict = extract_pattern(
            original_text=original,
            edited_text=edited,
            edit_type=edit_type,
            document_type=document_type,
            draft_type=draft_type,
        )

        # ── Step 3: Quality gate ─────────────────────────────────────────────
        if pattern_dict is None:
            _mark_edit(edit_id, "noise", classification, pattern_id=None)
            logger.info("Edit %s → NOISE (no pattern extracted)", edit_id)
            return

        passes, reason = passes_quality_gate(pattern_dict)
        if not passes:
            _mark_edit(edit_id, "noise", classification, pattern_id=None)
            logger.info("Edit %s failed quality gate: %s", edit_id, reason)
            return

        # ── Step 4: Save pattern to SQLite ───────────────────────────────────
        # Capture plain values from pattern_dict now — after session closes,
        # SQLAlchemy detaches the ORM object and attribute access raises DetachedInstanceError.
        rule_type = pattern_dict.get("rule_type", edit_type.lower())
        description = pattern_dict["description"]
        few_shot_before = pattern_dict.get("few_shot_before", "")
        few_shot_after = pattern_dict.get("few_shot_after", "")
        confidence = float(pattern_dict.get("confidence", 0.5))
        applicable_doc_types = pattern_dict.get("applicable_document_types", [document_type])
        applicable_draft_types_list = pattern_dict.get("applicable_draft_types", [draft_type])
        applicable_section_types = pattern_dict.get("applicable_section_types", [])

        pattern_id = str(uuid.uuid4())
        with Session(engine) as session:
            stored_edit = session.get(Edit, edit_id)
            operator_id = stored_edit.operator_id if stored_edit else "op_harvey"
            pattern_row = Pattern(
                pattern_id=pattern_id,
                source_edit_ids_json=json.dumps([edit_id]),
                rule_type=rule_type,
                description=description,
                few_shot_before=few_shot_before,
                few_shot_after=few_shot_after,
                applicable_document_types_json=json.dumps(applicable_doc_types),
                applicable_draft_types_json=json.dumps(applicable_draft_types_list),
                applicable_section_types_json=json.dumps(applicable_section_types),
                frequency=1,
                operator_consensus=1.0,
                confidence=confidence,
                is_active=True,
                operator_ids_json=json.dumps([operator_id]),
            )
            session.add(pattern_row)
            session.commit()

        # ── Step 5: Embed description → upsert to Qdrant ─────────────────────
        try:
            vector = embed_one(description)
            qdrant_point_id = qdrant_store.upsert_pattern(
                pattern_id=pattern_id,
                principle_vector=vector,
                payload={
                    "pattern_id": pattern_id,
                    "rule_type": rule_type,
                    "description": description,
                    "few_shot_before": few_shot_before,
                    "few_shot_after": few_shot_after,
                    "confidence": confidence,
                    "is_active": True,
                    "document_types": applicable_doc_types,
                    "draft_types": applicable_draft_types_list,
                },
            )
            with Session(engine) as session:
                p = session.get(Pattern, pattern_id)
                if p:
                    p.qdrant_point_id = qdrant_point_id
                    session.add(p)
                    session.commit()
        except Exception as exc:
            logger.warning("Qdrant upsert failed for pattern %s: %s", pattern_id, exc)

        # ── Step 6: Mark edit as extracted ───────────────────────────────────
        _mark_edit(edit_id, "extracted", classification, pattern_id=pattern_id)
        logger.info(
            "Edit %s → pattern %s stored [%s]: %s",
            edit_id, pattern_id, rule_type, description[:60],
        )

    except Exception as exc:
        logger.error("process_edit(%s) failed: %s", edit_id, exc, exc_info=True)
        _mark_edit(edit_id, "failed", {}, pattern_id=None)


def _mark_edit(
    edit_id: str,
    status: str,
    classification: dict,
    pattern_id: str | None,
) -> None:
    """Update an Edit row with its final processing outcome."""
    with Session(engine) as session:
        edit = session.get(Edit, edit_id)
        if edit is None:
            return
        edit.pattern_extraction_status = status
        edit.edit_classification_json = json.dumps(classification)
        edit.extracted_pattern_id = pattern_id
        session.add(edit)
        session.commit()
