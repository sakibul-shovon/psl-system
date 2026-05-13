"""
SQLite table definitions using SQLModel.

SQLModel = Pydantic v2 + SQLAlchemy in one class. The `table=True`
flag tells SQLAlchemy to create a real DB table; without it, the class
is a plain Pydantic model used just for validation / serialization.

All JSON list/dict fields are stored as TEXT (JSON strings). This is
intentional: SQLite has no native array type, and we never query
inside those JSON blobs — we always pull the whole row and parse in Python.
"""

import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ─── Document ──────────────────────────────────────────────────────────────────
# One row per uploaded file. Tracks the file path, type classification,
# and the structured fields Groq extracts (parties, dates, amounts).

class Document(SQLModel, table=True):
    document_id: str = Field(default_factory=_uuid, primary_key=True)
    title: str
    file_path: str                          # path inside data/uploads/
    document_type: str = "unknown"          # lease_agreement, service_agreement, etc.
    file_type: str = "pdf"                  # pdf | image
    uploaded_at: datetime = Field(default_factory=_now)
    processed_at: Optional[datetime] = None
    structured_fields_json: Optional[str] = None   # JSON: {parties, dates, amounts, ...}
    operator_id: str = "op_harvey"
    page_count: int = 0


# ─── Chunk ─────────────────────────────────────────────────────────────────────
# One row per legal section chunk. The content lives here; the embedding
# lives in Qdrant (no float arrays in SQLite). The two are linked by chunk_id.

class Chunk(SQLModel, table=True):
    chunk_id: str = Field(default_factory=_uuid, primary_key=True)
    document_id: str = Field(index=True)
    title: str                              # section heading, e.g. "Section 4.2"
    content: str                            # text, with [LOW_CONF:0.54] annotations
    breadcrumb: str                         # "Article IV > Section 4.2"
    structural_level: int = 1              # 1=Article, 2=Section, 3=subsection, 4=subsubsec
    page_range_json: str = "[]"            # JSON: [start_page, end_page]
    token_estimate: int = 0
    ocr_confidence_avg: float = 1.0
    ocr_confidence_min: float = 1.0
    has_low_conf_regions: bool = False
    extracted_fields_json: Optional[str] = None  # JSON: {amounts, parties}
    qdrant_point_id: Optional[str] = None  # UUID used in Qdrant


# ─── Draft ─────────────────────────────────────────────────────────────────────
# One row per generated draft. sections_json holds the full DraftSection list
# (title, content, citedEvidence, confidence) as a JSON string.

class Draft(SQLModel, table=True):
    draft_id: str = Field(default_factory=_uuid, primary_key=True)
    document_id: str = Field(index=True)
    draft_type: str = "case_fact_summary"
    title: str = ""
    sections_json: str = "[]"              # JSON: DraftSection[]
    grounding_score: float = 0.0
    ungrounded_claims_json: str = "[]"
    warnings_json: str = "[]"
    judge_scores_json: Optional[str] = None       # JSON: {groundedness, completeness, ...}
    applied_pattern_ids_json: str = "[]"   # JSON: [patternId, ...]
    processing_meta_json: str = "{}"       # model, traceId, timestamp, etc.
    created_at: datetime = Field(default_factory=_now)


# ─── Edit ──────────────────────────────────────────────────────────────────────
# One row per operator paragraph edit. Captures before/after text, which
# evidence was cited, and the async-filled classification + pattern result.

class Edit(SQLModel, table=True):
    edit_id: str = Field(default_factory=_uuid, primary_key=True)
    draft_id: str = Field(index=True)
    document_id: str = Field(index=True)
    document_type: str = "unknown"
    draft_type: str = "case_fact_summary"
    section_id: str = ""
    section_title: str = ""
    original_text: str
    edited_text: str
    diff_summary_json: str = "{}"          # {addedChars, removedChars, deltaRatio}
    cited_evidence_json: str = "[]"        # [E1, E3, ...]
    surrounding_context_json: str = "{}"   # {precedingSection, followingSection}
    operator_id: str = "op_harvey"
    edited_at: datetime = Field(default_factory=_now)
    # Filled in async after edit is stored:
    edit_classification_json: Optional[str] = None
    pattern_extraction_status: str = "pending"   # pending|extracted|noise|failed
    extracted_pattern_id: Optional[str] = None
    meta_json: str = "{}"                  # originalGroundingScore, judgeScore, ocrConf


# ─── Pattern ───────────────────────────────────────────────────────────────────
# One row per learned style/terminology/structure pattern. No embedding column —
# the vector lives in Qdrant `learned_patterns` collection, linked by qdrant_point_id.

class Pattern(SQLModel, table=True):
    pattern_id: str = Field(default_factory=_uuid, primary_key=True)
    source_edit_ids_json: str = "[]"       # [editId, ...]
    rule_type: str = "style"               # terminology|style|structure|precision|citation|omission_correction
    description: str                       # imperative one-sentence rule
    few_shot_before: str = ""
    few_shot_after: str = ""
    applicable_document_types_json: str = "[]"
    applicable_draft_types_json: str = "[]"
    applicable_section_types_json: str = "[]"
    frequency: int = 1                     # how many edits reinforced this pattern
    operator_consensus: float = 1.0        # fraction of unique operators who reinforced
    confidence: float = 0.0               # extraction confidence
    is_active: bool = True
    operator_ids_json: str = "[]"          # unique operators who contributed
    created_at: datetime = Field(default_factory=_now)
    last_reinforced_at: datetime = Field(default_factory=_now)
    qdrant_point_id: Optional[str] = None


# ─── Trace ─────────────────────────────────────────────────────────────────────
# Audit trail — one row per pipeline run (upload, generate, edit). stages_json
# is a list of {stage, startedAt, completedAt, durationMs, model, meta} objects.

class Trace(SQLModel, table=True):
    trace_id: str = Field(default_factory=_uuid, primary_key=True)
    request_type: str                      # upload|generate_draft|submit_edit
    document_id: Optional[str] = None
    draft_id: Optional[str] = None
    stages_json: str = "[]"                # JSON: StageRecord[]
    created_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[int] = None
