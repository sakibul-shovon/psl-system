"""
FastAPI entrypoint.

This file is what `uvicorn python_service.main:app` runs. Right now it only has
a /health endpoint — as we build out the pipeline (ingestion, retrieval,
generation, edit loop) we'll wire each module's routes into this app.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from python_service.config import settings

# ─── Logging setup ────────────────────────────────────────────────────────
# One stdlib logger configured at module load. Subsequent modules just
# `logger = logging.getLogger(__name__)` and inherit this configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan: code that runs on app startup and shutdown ─────────────────
# The async context manager pattern is FastAPI's recommended replacement for
# the older @app.on_event("startup") decorator (which is deprecated).
# Code before `yield` runs at startup; after `yield` runs at shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting PSL Document Intelligence service...")

    # Make sure local data directories exist (idempotent — safe to call every boot)
    for d in [Path("./data"), settings.bm25_dir, settings.uploads_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("PSL service ready. Tesseract: %s", settings.tesseract_cmd)
    yield
    logger.info("PSL service shutting down.")


# ─── App instance ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Pearson Specter Litt — Document Intelligence",
    version="0.1.0",
    description=(
        "Ingest messy legal documents, retrieve grounded evidence, generate "
        "drafts with inline citations, and learn from operator edits."
    ),
    lifespan=lifespan,
)

# Allow the Streamlit UI (default port 8501) to call this API from the browser.
# In production you'd lock this down to a specific origin; for local dev this
# is fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Liveness + environment sanity check.
    Hit this first when debugging — it tells you what config the app loaded
    and which optional integrations are configured.
    """
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
    """Friendly landing message."""
    return {
        "service": "Pearson Specter Litt — Document Intelligence",
        "docs": "/docs",
        "health": "/health",
    }
