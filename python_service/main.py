"""
FastAPI entrypoint.

`uvicorn python_service.main:app --reload` starts the server.
Routes are added here as each phase is built; all pipeline logic
lives in submodules (ingestion/, retrieval/, generation/, etc.).
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from python_service.config import settings
from python_service.db.models import Chunk, Document
from python_service.db.session import create_db_and_tables, engine
from python_service.vector.qdrant_store import qdrant_store
from python_service import jobs as job_store
from python_service.ingestion.pipeline import run_ingestion_pipeline
from python_service.retrieval.hybrid import retrieve
from python_service.retrieval.evidence import package_evidence
from python_service.db.models import Draft
from python_service.edit_loop.capture import store_edits
from python_service.edit_loop.processor import process_edit
from python_service.evaluation.improvement_validator import compute_improvement_report
from python_service.tracing import TraceBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting PSL Document Intelligence service...")

    # Create local data directories
    for d in [Path("./data"), settings.bm25_dir, settings.uploads_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create SQLite tables (idempotent — safe on every restart)
    create_db_and_tables()

    # Bootstrap Qdrant collections (idempotent — skipped if already exist)
    try:
        qdrant_store.ensure_collections()
    except Exception as exc:
        # Non-fatal at boot: if Qdrant isn't up yet, the first upload will fail
        # with a clear error rather than preventing the app from starting.
        logger.warning("Qdrant not reachable at startup: %s", exc)

    logger.info("PSL service ready. OCR: tesseract | DB: SQLite | Vectors: Qdrant")
    yield
    logger.info("PSL service shutting down.")


app = FastAPI(
    title="Pearson Specter Litt — Document Intelligence",
    version="0.1.0",
    description=(
        "Ingest messy legal documents, retrieve grounded evidence, generate "
        "drafts with inline citations, and learn from operator edits."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness + config sanity check. Hit this first when debugging."""
    return {
        "status": "ok",
        "service": "psl-document-intelligence",
        "version": "0.1.0",
        "qdrant_url": settings.qdrant_url,
        "tesseract_cmd": settings.tesseract_cmd,
        "has_gemini_key": bool(settings.gemini_api_key),
        "has_groq_key": bool(settings.groq_api_key),
        "default_operator_id": settings.default_operator_id,
    }


@app.get("/")
async def root():
    return {
        "service": "Pearson Specter Litt — Document Intelligence",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a PDF or image file, save it, and kick off the ingestion pipeline.

    Returns immediately with a jobId. Poll GET /job/{jobId} to track progress.
    The pipeline runs in the background: route → normalize → extract → chunk →
    embed → store (Qdrant + SQLite + BM25).
    """
    # Validate file type before doing any work
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {', '.join(allowed)}",
        )

    # Generate IDs up front so we can link the job → document immediately
    document_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    # Save the uploaded file to disk
    safe_name = f"{document_id}{suffix}"
    save_path = settings.uploads_dir / safe_name
    contents = await file.read()
    save_path.write_bytes(contents)

    # Create Document row in SQLite (minimal — pipeline fills in the rest)
    with Session(engine) as session:
        doc = Document(
            document_id=document_id,
            title=file.filename,
            file_path=str(save_path),
            file_type=suffix.lstrip("."),
            operator_id=settings.default_operator_id,
        )
        session.add(doc)
        session.commit()

    # Register the job so GET /job/{id} works immediately
    job_store.create_job(job_id, document_id=document_id)

    # Start the pipeline in the background — this returns right away
    background_tasks.add_task(run_ingestion_pipeline, save_path, document_id, job_id)

    return {
        "job_id": job_id,
        "document_id": document_id,
        "filename": file.filename,
        "message": "Upload received. Poll GET /job/{job_id} for progress.",
    }


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """Poll the status of a background pipeline job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return {
        "job_id": job.job_id,
        "document_id": job.document_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
    }


@app.get("/documents")
async def list_documents(limit: int = 100):
    """
    List all ingested documents, newest first.

    Used by the UI to populate document-selection dropdowns so users don't have
    to copy-paste UUIDs. Stores survive across browser refreshes because they
    live in SQLite, not session_state.
    """
    from sqlmodel import select
    with Session(engine) as session:
        docs = session.exec(
            select(Document)
            .order_by(Document.uploaded_at.desc())
            .limit(limit)
        ).all()

    return {
        "count": len(docs),
        "documents": [
            {
                "document_id": d.document_id,
                "title": d.title,
                "document_type": d.document_type,
                "page_count": d.page_count,
                "uploaded_at": d.uploaded_at.isoformat(),
            }
            for d in docs
        ],
    }


@app.get("/documents/{document_id}/chunks")
async def get_chunks(document_id: str):
    """
    List all chunks stored for a document.
    Useful for verifying the chunker worked correctly after upload.
    """
    from sqlmodel import Session, select
    with Session(engine) as session:
        chunks = session.exec(
            select(Chunk).where(Chunk.document_id == document_id)
        ).all()

    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for document {document_id!r}")

    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "breadcrumb": c.breadcrumb,
                "structural_level": c.structural_level,
                "token_estimate": c.token_estimate,
                "page_range": c.page_range_json,
                "ocr_confidence_avg": c.ocr_confidence_avg,
                "has_low_conf_regions": c.has_low_conf_regions,
                "content_preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
            }
            for c in chunks
        ],
    }


@app.post("/query")
async def query_document(body: dict):
    """
    Run hybrid retrieval (BM25 + dense + rerank) against a document.

    Body: { "document_id": "...", "query": "What are the payment terms?" }

    Returns top-5 evidence items with [E1]-[E5] labels, scores, and breadcrumbs.
    If evidence is insufficient, returns a diagnostic message instead.
    """
    document_id = body.get("document_id")
    query = body.get("query", "").strip()

    if not document_id:
        raise HTTPException(status_code=400, detail="'document_id' is required")
    if not query:
        raise HTTPException(status_code=400, detail="'query' is required")

    # Fetch document title for the evidence package
    with Session(engine) as session:
        doc = session.get(Document, document_id)
    doc_title = doc.title if doc else document_id

    # Run the full hybrid retrieval pipeline
    result = retrieve(query, document_id)

    if not result.sufficient:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "query": query,
            "diagnostic": result.diagnostic,
            "evidence": [],
        }

    evidence_items = package_evidence(result.evidence, document_title=doc_title)

    return {
        "status": "ok",
        "query": query,
        "document_id": document_id,
        "retrieval_method": result.retrieval_method,
        "evidence": [e.to_dict() for e in evidence_items],
        "prompt_blocks": [e.to_prompt_block() for e in evidence_items],
    }


@app.post("/draft")
async def generate_document_draft(body: dict):
    """
    Agentic draft pipeline (Phase B): planner → parallel executors → critic
    → refiner loop (max 3) → assembler.

    Body: {
      "document_id": "...",
      "query": "Summarize the compensation and termination terms",
      "draft_type": "case_fact_summary"   (optional, default: case_fact_summary)
    }

    Returns the same JSON schema as Phase A so the UI and feedback loop are
    unaffected. Internally the pipeline is now a LangGraph StateGraph with
    per-section focused retrieval, parallel generation, and critic-gated
    refinement.
    """
    from python_service.agent.graph import run_agent

    document_id   = body.get("document_id")
    query         = body.get("query", "").strip()
    draft_type    = body.get("draft_type", "case_fact_summary")
    skip_patterns = bool(body.get("skip_patterns", False))

    if not document_id:
        raise HTTPException(status_code=400, detail="'document_id' is required")
    if not query:
        raise HTTPException(status_code=400, detail="'query' is required")

    # Validate document exists before handing off to the agent.
    with Session(engine) as session:
        doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found")

    # ── Run the agent ─────────────────────────────────────────────────────────
    # run_agent() is synchronous (LangGraph .invoke()). It runs the full graph
    # — planner, executors, critic, optional refiner, assembler — and returns
    # the final DraftingState.
    trace = TraceBuilder("generate_draft", document_id=document_id)
    try:
        with trace.stage("agent_graph", model="gemini-2.5-flash+nli-deberta+llama-3.3-70b") as _stage:
            state = run_agent(document_id, query, draft_type, skip_patterns=skip_patterns)
            _stage["meta"]["agent_nodes"] = _agent_node_detail(state)
    except Exception as exc:
        logger.error("Agent run failed for document %r: %s", document_id, exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    # ── Guard: assembler must have saved a draft ──────────────────────────────
    if not state.get("final_draft_id"):
        raise HTTPException(
            status_code=500,
            detail="Agent completed but produced no draft. Check server logs.",
        )

    trace.save(draft_id=state["final_draft_id"])

    # ── Map agent state → HTTP response ──────────────────────────────────────
    # Derive grounding_status from the numeric score — same thresholds as Phase A.
    grounding_score = state.get("final_grounding_score", 0.0)
    if grounding_score >= 0.75:
        grounding_status = "HIGH"
    elif grounding_score >= 0.50:
        grounding_status = "MEDIUM"
    else:
        grounding_status = "LOW"

    adherence    = state.get("final_adherence", {})
    judge_scores = state.get("final_judge_scores", {})
    sections     = state.get("final_sections", [])
    patterns     = state.get("patterns", [])

    # Collect all evidence IDs cited across sections
    evidence_used = sorted({
        eid
        for sec in sections
        for eid in sec.get("evidence_ids", [])
    })

    return {
        "status":           "ok",
        "draft_id":         state["final_draft_id"],
        "document_id":      document_id,
        "draft_type":       draft_type,
        "title":            state.get("final_title", query),
        "sections":         sections,
        "grounding_score":  grounding_score,
        "grounding_status": grounding_status,
        "warnings":         [],          # per-section grounding captured in each section's confidence field
        "patterns_applied": len(patterns),
        "adherence_score":  adherence.get("adherence_score", 0.0),
        "adherence_detail": adherence.get("detail", []),
        "judge_scores":     judge_scores,
        "evidence_used":    evidence_used,
        "agent_iterations": state.get("iteration", 0),
        "trace_id":         trace.trace_id,
    }


@app.post("/draft/stream")
async def stream_draft(body: dict):
    """
    SSE streaming endpoint for draft generation.

    Emits server-sent events as each agent node completes so the UI can show
    live progress instead of waiting 10–20 s with a spinner.

    Events (each line: `data: <json>\\n\\n`):
      planner_done  — plan decomposed: {"plan": [...], "patterns_count": N}
      section_ready — one executor finished: [{"section_id", "title", "content", ...}]
      critic_done   — critic ran: {"iteration": N, "weak_count": N}
      refiner_done  — refiner improved queries: {"iteration": N}
      done          — assembler finished: full draft result
      error         — agent raised: {"message": "..."}
    """
    import json as _json
    from python_service.agent.graph import drafting_agent

    document_id   = body.get("document_id")
    query         = body.get("query", "").strip()
    draft_type    = body.get("draft_type", "case_fact_summary")
    skip_patterns = bool(body.get("skip_patterns", False))

    if not document_id:
        raise HTTPException(status_code=400, detail="'document_id' is required")
    if not query:
        raise HTTPException(status_code=400, detail="'query' is required")

    with Session(engine) as session:
        doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found")

    initial_state = {
        "document_id":   document_id,
        "query":         query,
        "draft_type":    draft_type,
        "skip_patterns": skip_patterns,
    }

    def event_stream():
        try:
            for chunk in drafting_agent.stream(initial_state, stream_mode="updates"):
                for node_name, update in chunk.items():
                    payload = _node_to_sse_event(node_name, update)
                    if payload:
                        yield f"data: {_json.dumps(payload)}\n\n"
        except Exception as exc:
            logger.error("Stream draft failed: %s", exc)
            err_payload = {"event": "error", "data": {"message": str(exc)}}
            yield f"data: {_json.dumps(err_payload)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback")
async def submit_feedback(body: dict, background_tasks: BackgroundTasks):
    """
    Submit operator edits for a draft. Triggers pattern learning in the background.

    Body:
    {
      "draft_id": "...",
      "edits": [
        {
          "section_id": "sec_1",
          "section_title": "Company Termination",
          "original_text": "The employee will get 3x salary...",
          "edited_text": "Employee shall receive a lump sum equal to three (3) times Base Compensation..."
        }
      ]
    }

    Returns immediately with edit_ids. Pattern extraction runs in the background.
    Poll GET /patterns to see extracted patterns.
    """
    draft_id = body.get("draft_id")
    edits = body.get("edits", [])

    if not draft_id:
        raise HTTPException(status_code=400, detail="'draft_id' is required")
    if not edits or not isinstance(edits, list):
        raise HTTPException(status_code=400, detail="'edits' must be a non-empty list")

    # Verify draft exists
    with Session(engine) as session:
        draft = session.get(Draft, draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail=f"Draft {draft_id!r} not found")

    # Store raw edits (fast — just SQL INSERTs)
    try:
        edit_ids = store_edits(
            draft_id=draft_id,
            edits=edits,
            operator_id=settings.default_operator_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Classify + extract patterns in the background (slow — Groq LLM calls)
    for edit_id in edit_ids:
        background_tasks.add_task(process_edit, edit_id)

    # Write episodic memory for this feedback session (fast — SQL read + embed)
    background_tasks.add_task(_write_episodic_memory, draft_id, edits)

    return {
        "status": "accepted",
        "draft_id": draft_id,
        "edits_received": len(edits),
        "edits_stored": len(edit_ids),
        "edit_ids": edit_ids,
        "message": f"Pattern extraction running in background for {len(edit_ids)} edit(s).",
    }


@app.get("/patterns")
async def list_patterns():
    """List all active learned patterns stored in SQLite."""
    from sqlmodel import select
    from python_service.db.models import Pattern

    with Session(engine) as session:
        patterns = session.exec(
            select(Pattern).where(Pattern.is_active == True)
        ).all()

    return {
        "pattern_count": len(patterns),
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "rule_type": p.rule_type,
                "description": p.description,
                "few_shot_before": p.few_shot_before,
                "few_shot_after": p.few_shot_after,
                "confidence": p.confidence,
                "frequency": p.frequency,
                "created_at": p.created_at.isoformat(),
            }
            for p in patterns
        ],
    }


@app.get("/metrics")
async def get_metrics():
    """
    System-wide metrics: document/draft/edit/pattern counts and
    average quality scores.  Use this as a dashboard summary.
    """
    import json as _json
    from sqlmodel import select, func
    from python_service.db.models import Document, Chunk, Draft, Edit, Pattern

    with Session(engine) as session:
        doc_count = session.exec(select(func.count()).select_from(Document)).one()
        chunk_count = session.exec(select(func.count()).select_from(Chunk)).one()
        draft_count = session.exec(select(func.count()).select_from(Draft)).one()
        edit_count = session.exec(select(func.count()).select_from(Edit)).one()
        pattern_count = session.exec(
            select(func.count()).select_from(Pattern).where(Pattern.is_active == True)
        ).one()
        drafts = session.exec(select(Draft)).all()

    grounding_scores = [d.grounding_score for d in drafts if d.grounding_score > 0]
    judge_scores = []
    for d in drafts:
        try:
            s = _json.loads(d.judge_scores_json or "{}")
            if s.get("overall"):
                judge_scores.append(float(s["overall"]))
        except (ValueError, TypeError):
            pass

    def _avg(vals):
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "counts": {
            "documents": doc_count,
            "chunks": chunk_count,
            "drafts": draft_count,
            "edits_submitted": edit_count,
            "patterns_active": pattern_count,
        },
        "quality": {
            "avg_grounding_score": _avg(grounding_scores),
            "avg_judge_overall": _avg(judge_scores),
            "drafts_scored": len(judge_scores),
        },
    }


@app.get("/evaluation/improvement-report")
async def improvement_report():
    """
    Compare draft quality before vs after pattern learning.

    Splits drafts into two cohorts:
      - before: no patterns were injected (applied_pattern_ids_json = [])
      - after:  at least one pattern was applied

    Returns average grounding + judge scores per cohort and the delta.
    """
    report = compute_improvement_report()
    return {
        "has_data": report.has_data,
        "message": report.message,
        "before_patterns": {
            "draft_count": report.before.count,
            "avg_grounding_score": report.before.avg_grounding,
            "avg_judge_scores": {
                "groundedness": report.before.avg_groundedness,
                "completeness": report.before.avg_completeness,
                "structure": report.before.avg_structure,
                "overall": report.before.avg_overall,
            },
        },
        "after_patterns": {
            "draft_count": report.after.count,
            "avg_grounding_score": report.after.avg_grounding,
            "avg_judge_scores": {
                "groundedness": report.after.avg_groundedness,
                "completeness": report.after.avg_completeness,
                "structure": report.after.avg_structure,
                "overall": report.after.avg_overall,
            },
        },
        "delta": {
            "grounding_score": report.delta_grounding,
            "overall_judge_score": report.delta_overall,
        },
    }


# ── Trace audit endpoints ──────────────────────────────────────────────────────

@app.get("/traces")
async def list_traces(limit: int = 50):
    """
    List recent pipeline traces, newest first.

    Each trace corresponds to one POST /draft call and records per-stage
    timing and model information. Use GET /traces/{trace_id} for full detail.
    """
    from python_service.db.models import Trace
    from sqlmodel import select as _select

    with Session(engine) as session:
        rows = session.exec(
            _select(Trace).order_by(Trace.created_at.desc()).limit(limit)
        ).all()

    return {
        "count": len(rows),
        "traces": [
            {
                "trace_id":          r.trace_id,
                "request_type":      r.request_type,
                "document_id":       r.document_id,
                "draft_id":          r.draft_id,
                "total_duration_ms": r.total_duration_ms,
                "created_at":        r.created_at.isoformat(),
            }
            for r in rows
        ],
    }


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """
    Full audit record for a single pipeline run.

    Returns per-stage breakdown: stage name, model, duration (ms), and
    any metadata captured at run time (evidence count, patterns injected, etc.).
    """
    import json as _json
    from python_service.db.models import Trace

    with Session(engine) as session:
        row = session.get(Trace, trace_id)

    if not row:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id!r} not found")

    stages = _json.loads(row.stages_json or "[]")

    # Extract agent node detail stored inside the agent_graph stage meta.
    agent_nodes = next(
        (s.get("meta", {}).get("agent_nodes") for s in stages if s.get("stage") == "agent_graph"),
        None,
    )

    return {
        "trace_id":          row.trace_id,
        "request_type":      row.request_type,
        "document_id":       row.document_id,
        "draft_id":          row.draft_id,
        "created_at":        row.created_at.isoformat(),
        "completed_at":      row.completed_at.isoformat() if row.completed_at else None,
        "total_duration_ms": row.total_duration_ms,
        "stages":            stages,
        "agent_nodes":       agent_nodes,
    }


# ── Pattern impact analytics ───────────────────────────────────────────────────

@app.get("/patterns/{pattern_id}/impact")
async def pattern_impact(pattern_id: str):
    """
    How many drafts has this pattern been applied to, and what quality lift
    does it produce?

    Returns frequency, operator consensus, drafts applied, and average
    judge_overall score for drafts that used this pattern.
    Reviewers use this to verify the learning loop is producing value.
    """
    import json as _json
    import statistics
    from sqlmodel import select as _select
    from python_service.db.models import Pattern, Draft

    with Session(engine) as session:
        p = session.get(Pattern, pattern_id)
        if not p:
            raise HTTPException(status_code=404, detail="Pattern not found")
        all_drafts = session.exec(_select(Draft)).all()

    # Find drafts that listed this pattern in their applied_pattern_ids_json
    applied_drafts = [
        d for d in all_drafts
        if pattern_id in _json.loads(d.applied_pattern_ids_json or "[]")
    ]

    judge_scores = []
    for d in applied_drafts:
        try:
            s = _json.loads(d.judge_scores_json or "{}")
            if s.get("overall") is not None:
                judge_scores.append(float(s["overall"]))
        except (ValueError, TypeError):
            pass

    return {
        "pattern_id":          pattern_id,
        "description":         p.description,
        "rule_type":           p.rule_type,
        "frequency":           p.frequency,
        "operator_consensus":  p.operator_consensus,
        "confidence":          p.confidence,
        "is_active":           p.is_active,
        "drafts_applied_to":   len(applied_drafts),
        "avg_judge_overall_when_applied": (
            round(statistics.mean(judge_scores), 3) if judge_scores else None
        ),
        "last_reinforced_at":  p.last_reinforced_at.isoformat(),
        "created_at":          p.created_at.isoformat(),
        "source_edit_ids":     _json.loads(p.source_edit_ids_json or "[]"),
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _node_to_sse_event(node_name: str, update: dict) -> dict | None:
    """
    Convert a LangGraph node update (stream_mode='updates') into an SSE payload.

    Each node returns only its OWN new fields — not the full state — so we
    read exactly what each node writes back:
      planner   → plan, patterns, document_title
      executor  → section_drafts (one element, before operator.add merges it)
      critic    → critique list + iteration
      refiner   → updated plan
      assembler → final_* fields
    """
    if node_name == "planner":
        plan = update.get("plan", [])
        return {
            "event": "planner_done",
            "data": {
                "plan": [
                    {
                        "section_id": s["section_id"],
                        "title":      s["title"],
                        "brief":      s.get("brief", ""),
                    }
                    for s in plan
                ],
                "patterns_count":  len(update.get("patterns", [])),
                "document_title":  update.get("document_title", ""),
            },
        }

    if node_name == "executor":
        drafts = update.get("section_drafts", [])
        return {
            "event": "section_ready",
            "data": [
                {
                    "section_id":      d["section_id"],
                    "title":           d["title"],
                    "content":         d["content"],
                    "confidence":      d["confidence"],
                    "grounding_score": d["grounding_score"],
                }
                for d in drafts
            ],
        }

    if node_name == "critic":
        critique = update.get("critique", [])
        return {
            "event": "critic_done",
            "data": {
                "iteration":  update.get("iteration", 0),
                "weak_count": len(critique),
                "weak_ids":   [c["section_id"] for c in critique],
            },
        }

    if node_name == "refiner":
        return {
            "event": "refiner_done",
            "data": {"iteration": update.get("iteration", 0)},
        }

    if node_name == "assembler":
        adherence = update.get("final_adherence", {})
        return {
            "event": "done",
            "data": {
                "draft_id":         update.get("final_draft_id"),
                "title":            update.get("final_title", ""),
                "sections":         update.get("final_sections", []),
                "grounding_score":  update.get("final_grounding_score", 0.0),
                "judge_scores":     update.get("final_judge_scores", {}),
                "adherence_score":  adherence.get("adherence_score", 1.0),
                "adherence_detail": adherence.get("detail", []),
                "agent_iterations": update.get("iteration", 0),
            },
        }

    return None


def _agent_node_detail(state: dict) -> dict:
    """
    Build a compact agent node-level summary from the final DraftingState.

    Stored inside the agent_graph stage's meta dict so no schema change is
    needed on the Trace model. Surfaced by GET /traces/{id} as `agent_nodes`.
    """
    section_drafts = state.get("section_drafts", [])
    iterations     = state.get("iteration", 0)
    plan           = state.get("plan", [])
    patterns       = state.get("patterns", [])

    n = len(plan)
    # Reconstruct the node execution sequence from observable state.
    # planner fires once; executors fire for every section on first pass and
    # for weak sections only on each refinement pass.
    nodes_run = [f"planner", f"executor × {n}", "critic"]
    for i in range(iterations):
        nodes_run += [f"refiner (iter {i + 1})", "executor (weak sections)", "critic"]
    nodes_run.append("assembler")

    return {
        "nodes_run":              nodes_run,
        "plan_sections":          n,
        "sections_executed":      len(section_drafts),
        "refinement_iterations":  iterations,
        "patterns_injected":      len(patterns),
        "per_section": [
            {
                "section_id":      d["section_id"],
                "title":           d["title"],
                "grounding_score": d["grounding_score"],
                "confidence":      d["confidence"],
            }
            for d in sorted(section_drafts, key=lambda d: d["section_id"])
        ],
    }


def _write_episodic_memory(draft_id: str, edits: list[dict]) -> None:
    """
    Background task: record the (draft, edit-outcome) session as episodic memory.

    Called by the /feedback route. We store:
      - The query that generated the draft (from processing_meta_json)
      - Quality scores (grounding, judge_overall)
      - How much the operator changed the draft (edit distances)

    This lets the planner ask "what happened last time I drafted something
    similar?" and inject that context into its decomposition prompt.

    WHY background task?
    The embedding call (~50ms) and two DB writes would add latency to the
    /feedback response. Since episodic memory is used at the NEXT draft
    (not the current one), it's fine to write asynchronously.
    """
    import json as _json
    import Levenshtein
    from python_service.db.models import EpisodicMemory, Draft
    from python_service.embedder import embed_one
    from python_service.vector.qdrant_store import qdrant_store

    try:
        # Read every attribute we need while the session is open.
        # Accessing ORM attributes after `with Session()` closes causes
        # DetachedInstanceError because SQLAlchemy expires them on commit.
        with Session(engine) as session:
            draft = session.get(Draft, draft_id)
            if not draft:
                logger.warning("EpisodicMemory skipped — draft %r not found", draft_id)
                return
            document_id        = draft.document_id
            draft_type         = draft.draft_type
            grounding_score    = draft.grounding_score
            processing_meta    = draft.processing_meta_json or "{}"
            judge_scores_raw   = draft.judge_scores_json or "{}"

        # Extract query + document_type from the processing metadata the
        # assembler wrote. Falls back gracefully if either is missing.
        meta = _json.loads(processing_meta)
        query = meta.get("query", "")
        document_type = meta.get("documentType", "unknown")

        judge_scores = _json.loads(judge_scores_raw)
        try:
            judge_overall = float(judge_scores["overall"]) if judge_scores.get("overall") else None
        except (TypeError, ValueError):
            judge_overall = None

        # Sum of Levenshtein edit distances across all submitted edits.
        # A high total means the operator heavily rewrote the draft.
        # A decreasing trend over sessions means the system is converging.
        edit_distance_total = sum(
            Levenshtein.distance(
                e.get("original_text", ""),
                e.get("edited_text", ""),
            )
            for e in edits
        )

        memory = EpisodicMemory(
            document_id=document_id,
            document_type=document_type,
            query=query,
            draft_id=draft_id,
            draft_type=draft_type,
            grounding_score=grounding_score,
            judge_overall=judge_overall,
            edit_distance_total=edit_distance_total,
            edit_count=len(edits),
        )

        with Session(engine) as session:
            session.add(memory)
            session.commit()
            # Read memory_id while session is open; commit expires the object
            # and accessing it afterwards causes DetachedInstanceError.
            memory_id = memory.memory_id

        # Embed "query | document_type" so the planner can find similar sessions
        # using cosine similarity (same approach as pattern retrieval).
        embed_text = f"{query} | {document_type}"
        vector = embed_one(embed_text)
        point_id = qdrant_store.upsert_episodic_memory(
            memory_id=memory_id,
            query_vector=vector,
            payload={
                "memory_id":           memory_id,
                "document_id":         document_id,
                "document_type":       document_type,
                "draft_type":          draft_type,
                "query":               query[:200],
                "grounding_score":     grounding_score,
                "judge_overall":       judge_overall,
                "edit_count":          len(edits),
                "edit_distance_total": edit_distance_total,
            },
        )

        with Session(engine) as session:
            mem = session.get(EpisodicMemory, memory_id)
            if mem:
                mem.qdrant_point_id = point_id
                session.add(mem)
                session.commit()

        logger.info(
            "EpisodicMemory %r written for draft %r (edits=%d, total_dist=%d)",
            memory_id, draft_id, len(edits), edit_distance_total,
        )

    except Exception as exc:
        logger.error("_write_episodic_memory(%r) failed: %s", draft_id, exc, exc_info=True)
