"""
Audit trace builder.

Usage inside a pipeline route:

    trace = TraceBuilder("generate_draft", document_id=doc_id)

    with trace.stage("retrieval"):
        results = retrieve(query, document_id)

    with trace.stage("generation", model="gemini-2.5-flash"):
        draft = generate_draft(prompt)

    trace.save(draft_id=draft_id)

Each `with trace.stage(...)` block records:
  - stage name
  - wall-clock start / end time
  - duration in milliseconds
  - model name (optional)
  - any extra keyword-arg metadata

`trace.save()` writes one Trace row to SQLite with all stages as JSON.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlmodel import Session

from python_service.db.models import Trace
from python_service.db.session import engine


class TraceBuilder:
    def __init__(
        self,
        request_type: str,
        document_id: str | None = None,
        draft_id: str | None = None,
    ) -> None:
        self.trace_id    = str(uuid4())
        self.request_type = request_type
        self.document_id  = document_id
        self.draft_id     = draft_id
        self._stages: list[dict[str, Any]] = []
        self._created_at  = datetime.now(timezone.utc)

    @contextmanager
    def stage(self, name: str, model: str = "", **meta: Any):
        """
        Context manager that times a pipeline stage.

        Example:
            with trace.stage("grounding", model="nli-deberta-v3-small"):
                result = verify_draft(sections, evidence_map)
        """
        record: dict[str, Any] = {
            "stage":      name,
            "model":      model,
            "startedAt":  datetime.now(timezone.utc).isoformat(),
            "meta":       meta,
        }
        t0 = time.perf_counter()
        try:
            yield record
        finally:
            record["durationMs"]   = int((time.perf_counter() - t0) * 1000)
            record["completedAt"]  = datetime.now(timezone.utc).isoformat()
            self._stages.append(record)

    def save(self, draft_id: str | None = None) -> "TraceBuilder":
        """Persist the completed trace to SQLite. Returns self for chaining."""
        if draft_id:
            self.draft_id = draft_id

        total_ms = sum(s.get("durationMs", 0) for s in self._stages)

        with Session(engine) as session:
            row = Trace(
                trace_id=self.trace_id,
                request_type=self.request_type,
                document_id=self.document_id,
                draft_id=self.draft_id,
                stages_json=json.dumps(self._stages),
                created_at=self._created_at,
                completed_at=datetime.now(timezone.utc),
                total_duration_ms=total_ms,
            )
            session.add(row)
            session.commit()

        return self
