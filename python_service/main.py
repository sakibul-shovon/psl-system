"""
FastAPI entrypoint.

`uvicorn python_service.main:app --reload` starts the server.
Routes are added here as each phase is built; all pipeline logic
lives in submodules (ingestion/, retrieval/, generation/, etc.).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from python_service.config import settings
from python_service.db.session import create_db_and_tables
from python_service.vector.qdrant_store import qdrant_store
from python_service import jobs as job_store

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


@app.get("/job/{job_id}")
async def get_job(job_id: str):
    """
    Poll the status of a background pipeline job.

    Returns status (pending|running|done|failed), current stage,
    progress 0-100, and result/error when finished.
    """
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
