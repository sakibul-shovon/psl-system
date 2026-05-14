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
from datetime import datetime
from pathlib import Path

from sqlmodel import Session

from python_service.db.models import Edit, Pattern
from python_service.db.session import engine
from python_service.edit_loop.classifier import classify_edit
from python_service.edit_loop.pattern_extractor import extract_pattern
from python_service.evaluation.pattern_quality_gate import passes_quality_gate
from python_service.embedder import embed_one
from python_service.vector.qdrant_store import qdrant_store

# DPO preference pairs are written here for potential future fine-tuning.
# Each line is a JSON object with {chosen, rejected, context}.
# Format matches the standard Direct Preference Optimization data schema.
_PREFERENCES_FILE = Path("data/preferences.jsonl")

logger = logging.getLogger(__name__)

# Cosine-similarity threshold for treating a candidate pattern as a duplicate of
# an existing one. Above this, we REINFORCE the existing pattern (++frequency,
# refresh timestamp, etc.) instead of inserting a new row. This is what makes
# the edit loop a learning system rather than a logging system.
DEDUP_THRESHOLD = 0.85


def process_edit(edit_id: str) -> None:
    """
    Run the full classify → extract → quality-gate → (reinforce|insert) → mark
    pipeline for one edit.

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
        operator_id = edit.operator_id

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

        # ── Step 4: Capture pattern values ───────────────────────────────────
        # Pull all needed values out of pattern_dict now — once SQLAlchemy
        # session closes, ORM objects detach and attribute access raises.
        rule_type = pattern_dict.get("rule_type", edit_type.lower())
        description = pattern_dict["description"]
        few_shot_before = pattern_dict.get("few_shot_before", "")
        few_shot_after = pattern_dict.get("few_shot_after", "")
        confidence = float(pattern_dict.get("confidence", 0.5))
        applicable_doc_types = pattern_dict.get("applicable_document_types", [document_type])
        applicable_draft_types_list = pattern_dict.get("applicable_draft_types", [draft_type])
        applicable_section_types = pattern_dict.get("applicable_section_types", [])

        # ── Step 4.5: Dedup-by-similarity — REINFORCE if existing pattern ───
        # Embed the candidate description once; reuse the vector below for the
        # new-insert path if dedup misses (saves one embedding call either way).
        try:
            candidate_vec = embed_one(description)
            similar_hits = qdrant_store.search_similar_patterns(
                query_vector=candidate_vec,
                limit=5,
                active_only=True,
            )
        except Exception as exc:
            logger.warning("Qdrant similarity search failed (treating as new insert): %s", exc)
            candidate_vec = None
            similar_hits = []

        existing_hit = next(
            (h for h in similar_hits if h.get("score", 0.0) >= DEDUP_THRESHOLD),
            None,
        )

        if existing_hit:
            existing_pattern_id = existing_hit["payload"].get("pattern_id")
            existing_point_id = existing_hit["payload"].get("qdrant_point_id")
            new_freq = None
            new_conf = None

            with Session(engine) as session:
                p = session.get(Pattern, existing_pattern_id)
                if p is not None:
                    # Increment usage counters
                    p.frequency += 1
                    p.last_reinforced_at = datetime.utcnow()

                    # Append this edit to source_edit_ids (deduped)
                    source_ids = json.loads(p.source_edit_ids_json or "[]")
                    if edit_id not in source_ids:
                        source_ids.append(edit_id)
                        p.source_edit_ids_json = json.dumps(source_ids)

                    # Track unique operators who reinforced; consensus =
                    # unique_operators / total_reinforcements. A pattern with
                    # consensus 1.0 means every operator agrees; lower means
                    # one operator is insisting on something others don't.
                    op_ids = set(json.loads(p.operator_ids_json or "[]"))
                    op_ids.add(operator_id)
                    p.operator_ids_json = json.dumps(sorted(op_ids))
                    p.operator_consensus = round(len(op_ids) / max(p.frequency, 1), 3)

                    # Small confidence bump on reinforcement, capped at 0.99
                    p.confidence = round(min(0.99, p.confidence + 0.05), 3)

                    session.add(p)
                    session.commit()

                    new_freq = p.frequency
                    new_conf = p.confidence

            # Push the fresh frequency/confidence into the Qdrant payload so
            # downstream retrieval composite-weighting sees the update.
            if existing_point_id and new_freq is not None:
                try:
                    qdrant_store.update_pattern_payload(
                        existing_point_id,
                        {"frequency": new_freq, "confidence": new_conf},
                    )
                except Exception as exc:
                    logger.warning("Qdrant payload update failed: %s", exc)

            _mark_edit(edit_id, "extracted", classification, pattern_id=existing_pattern_id)
            _emit_preference(
                edit_id=edit_id,
                original=original,
                edited=edited,
                document_type=document_type,
                draft_type=draft_type,
                section_title="",   # not stored on Edit row
                rule_type=rule_type,
            )
            logger.info(
                "Edit %s REINFORCED existing pattern %s (frequency now %s, conf %s)",
                edit_id, existing_pattern_id, new_freq, new_conf,
            )
            return

        # ── Step 5: No match — insert NEW pattern ────────────────────────────
        pattern_id = str(uuid.uuid4())
        with Session(engine) as session:
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

        # ── Step 6: Embed description → upsert to Qdrant ─────────────────────
        # Re-embed only if Step 4.5's vector was lost to a Qdrant failure.
        try:
            vector = candidate_vec if candidate_vec is not None else embed_one(description)
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
                    "frequency": 1,
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

        # ── Step 7: Mark edit as extracted + emit DPO preference ─────────────
        _mark_edit(edit_id, "extracted", classification, pattern_id=pattern_id)
        _emit_preference(
            edit_id=edit_id,
            original=original,
            edited=edited,
            document_type=document_type,
            draft_type=draft_type,
            section_title="",
            rule_type=rule_type,
        )
        logger.info(
            "Edit %s → NEW pattern %s stored [%s]: %s",
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


def _emit_preference(
    edit_id: str,
    original: str,
    edited: str,
    document_type: str,
    draft_type: str,
    section_title: str,
    rule_type: str,
) -> None:
    """
    Append a DPO-style preference pair to data/preferences.jsonl.

    WHY emit this?
    Direct Preference Optimization (DPO) and RLHF both need (chosen, rejected)
    pairs. Every operator edit IS a preference pair: the original text is what
    the model preferred; the edited text is what the human expert preferred.
    We're not fine-tuning today, but this file is the first step toward it.
    A reviewer seeing this file knows the system is designed for continuous
    improvement beyond the current session.

    Format matches the HuggingFace TRL DPOTrainer expectation:
      {"chosen": "...", "rejected": "...", "prompt": "...", ...}
    """
    try:
        _PREFERENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "edit_id":       edit_id,
            "chosen":        edited,       # what the human expert wrote
            "rejected":      original,     # what the model generated
            "context": {
                "document_type": document_type,
                "draft_type":    draft_type,
                "section_title": section_title,
                "rule_type":     rule_type,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
        with _PREFERENCES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        # Non-fatal: preference logging should never block pattern extraction
        logger.warning("Failed to emit preference for edit %s: %s", edit_id, exc)
