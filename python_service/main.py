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
from sqlmodel import Session

from python_service.config import settings
from python_service.db.models import Chunk, Document
from python_service.db.session import create_db_and_tables, engine
from python_service.vector.qdrant_store import qdrant_store
from python_service import jobs as job_store
from python_service.ingestion.pipeline import run_ingestion_pipeline
from python_service.retrieval.hybrid import retrieve
from python_service.retrieval.evidence import package_evidence
from python_service.generation.context import build_prompt
from python_service.generation.gemini import generate_draft
from python_service.generation.grounding import verify_draft
from python_service.evaluation.draft_judge import judge_draft
from python_service.db.models import Draft

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
    Full generation pipeline: retrieve evidence → build prompt → Gemini draft
    → NLI grounding check → judge score → save to SQLite.

    Body: {
      "document_id": "...",
      "query": "Summarize the compensation and termination terms",
      "draft_type": "case_fact_summary"   (optional, default: case_fact_summary)
    }

    Returns the full draft with inline [E1] citations, grounding score,
    and independent judge scores. Refuses to deliver if grounding < 0.50.
    """
    import json as _json

    document_id = body.get("document_id")
    query = body.get("query", "").strip()
    draft_type = body.get("draft_type", "case_fact_summary")

    if not document_id:
        raise HTTPException(status_code=400, detail="'document_id' is required")
    if not query:
        raise HTTPException(status_code=400, detail="'query' is required")

    # ── Step 1: Retrieve evidence ────────────────────────────────────────────
    with Session(engine) as session:
        doc = session.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found")

    retrieval_result = retrieve(query, document_id)
    if not retrieval_result.sufficient:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "diagnostic": retrieval_result.diagnostic,
        }

    evidence_items = package_evidence(retrieval_result.evidence, document_title=doc.title)
    evidence_dicts = [e.to_dict() for e in evidence_items]

    # ── Step 2: Build prompt + generate draft ────────────────────────────────
    prompt = build_prompt(
        evidence_items=evidence_items,
        draft_type=draft_type,
        document_title=doc.title,
    )
    raw_draft = generate_draft(prompt)
    sections = raw_draft.get("sections", [])

    # ── Step 3: Grounding verification ──────────────────────────────────────
    evidence_map = {e.evidence_id: e.content for e in evidence_items}
    grounding = verify_draft(sections, evidence_map)

    if grounding.status == "LOW":
        return {
            "status": "INSUFFICIENT_GROUNDING",
            "grounding_score": grounding.grounding_score,
            "diagnostic": grounding.diagnostic,
            "warnings": [w.warning_type + ": " + w.sentence[:100] for w in grounding.warnings],
        }

    # ── Step 4: Judge score ──────────────────────────────────────────────────
    judge_scores = judge_draft(sections, evidence_dicts)

    # ── Step 5: Save to SQLite ───────────────────────────────────────────────
    draft_id = str(uuid.uuid4())
    with Session(engine) as session:
        draft_row = Draft(
            draft_id=draft_id,
            document_id=document_id,
            draft_type=draft_type,
            title=raw_draft.get("title", query),
            sections_json=_json.dumps(sections),
            grounding_score=grounding.grounding_score,
            warnings_json=_json.dumps([
                {"type": w.warning_type, "sentence": w.sentence, "evidence_id": w.evidence_id}
                for w in grounding.warnings
            ]),
            judge_scores_json=_json.dumps(judge_scores),
            applied_pattern_ids_json="[]",
            processing_meta_json=_json.dumps({
                "generationModel": "gemini-2.5-flash",
                "judgeModel": "llama-3.3-70b-versatile",
                "retrievalMethod": retrieval_result.retrieval_method,
                "evidenceIds": [e.evidence_id for e in evidence_items],
            }),
        )
        session.add(draft_row)
        session.commit()

    return {
        "status": "ok",
        "draft_id": draft_id,
        "document_id": document_id,
        "draft_type": draft_type,
        "title": raw_draft.get("title", ""),
        "sections": sections,
        "grounding_score": grounding.grounding_score,
        "grounding_status": grounding.status,
        "warnings": [
            {"type": w.warning_type, "sentence": w.sentence[:120]}
            for w in grounding.warnings
        ],
        "judge_scores": judge_scores,
        "evidence_used": [e.evidence_id for e in evidence_items],
    }
