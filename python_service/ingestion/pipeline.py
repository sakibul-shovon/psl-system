"""
Ingestion pipeline orchestrator.

Runs the full processing sequence for one uploaded file:
  1. Route file (per-page TEXT_LAYER vs OCR_NEEDED)
  2. Normalize OCR lines (hyphen join, logical line join)
  3. Extract structured fields (Groq 8B)
  4. Chunk the document (stack-based state machine)
  5. Embed each chunk (bge-base-en-v1.5)
  6. Store vectors in Qdrant
  7. Store metadata in SQLite
  8. Build + save BM25 index

This function runs as a FastAPI BackgroundTask — it starts after the
/upload endpoint returns. Job status is updated at each stage so the
UI can show a live progress bar via GET /job/{id}.
"""

import json
import logging
from pathlib import Path

from sqlmodel import Session

from python_service import jobs as job_store
from python_service.chunking.legal_chunker import chunk_document
from python_service.db.models import Chunk, Document
from python_service.db.session import engine
from python_service.embedder import embed_texts
from python_service.ingestion.file_router import route_file
from python_service.ingestion.line_normalizer import normalize_lines
from python_service.ingestion.structured_extractor import extract_structured_fields
from python_service.retrieval.bm25_index import build_index, save_index
from python_service.vector.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(file_path: Path, document_id: str, job_id: str) -> None:
    """
    Full ingestion pipeline. Called as a background task by POST /upload.

    Args:
        file_path:   path to the saved file in data/uploads/
        document_id: pre-created Document row's ID in SQLite
        job_id:      Job record to update with progress
    """
    # We create our own DB session here because this runs in a background task.
    # The request-scoped session from the route handler has already been closed
    # by the time this function executes.
    with Session(engine) as session:
        try:
            _run(file_path, document_id, job_id, session)
        except Exception as exc:
            logger.exception("Pipeline failed for document %s: %s", document_id, exc)
            job_store.fail_job(job_id, str(exc))


def _run(file_path: Path, document_id: str, job_id: str, session: Session) -> None:
    """Inner pipeline — exceptions bubble up to the caller which handles job failure."""

    # ── Stage 1: Route file ─────────────────────────────────────────────────
    job_store.update_job(job_id, status="running", stage="routing", progress=10)
    logger.info("[%s] Stage 1: routing file %s", job_id, file_path.name)

    routed = route_file(file_path)

    # ── Stage 2: Normalize lines ────────────────────────────────────────────
    job_store.update_job(job_id, stage="normalizing", progress=25)
    logger.info("[%s] Stage 2: normalizing %d pages", job_id, routed.page_count)

    normalized_pages: list[tuple[int, str]] = []
    for page in routed.pages:
        if page.routing == "OCR_NEEDED":
            normalized_text = normalize_lines(page.raw_text)
        else:
            normalized_text = page.raw_text   # text-layer pages don't need normalization
        normalized_pages.append((page.page_num, normalized_text))

    # ── Stage 3: Extract structured fields ─────────────────────────────────
    job_store.update_job(job_id, stage="extracting fields", progress=35)
    logger.info("[%s] Stage 3: extracting structured fields", job_id)

    full_text = routed.full_text
    structured_fields = extract_structured_fields(full_text)

    # Update the Document row with extracted metadata
    doc = session.get(Document, document_id)
    if doc:
        doc.structured_fields_json = json.dumps(structured_fields)
        doc.document_type = structured_fields.get("documentType", "unknown")
        doc.page_count = routed.page_count
        session.add(doc)
        session.commit()

    # ── Stage 4: Chunk document ─────────────────────────────────────────────
    job_store.update_job(job_id, stage="chunking", progress=50)
    logger.info("[%s] Stage 4: chunking document", job_id)

    chunks = chunk_document(normalized_pages)
    if not chunks:
        raise ValueError("Document produced no chunks — may be empty or unreadable")

    logger.info("[%s] Produced %d chunks", job_id, len(chunks))

    # ── Stage 5: Embed all chunks ───────────────────────────────────────────
    job_store.update_job(job_id, stage="embedding", progress=65)
    logger.info("[%s] Stage 5: embedding %d chunks", job_id, len(chunks))

    # Embed content and title vectors in two batches (faster than one-by-one)
    content_texts = [c.content for c in chunks]
    title_texts = [c.title for c in chunks]

    content_vectors = embed_texts(content_texts)
    title_vectors = embed_texts(title_texts)

    # ── Stage 6: Store in Qdrant ────────────────────────────────────────────
    job_store.update_job(job_id, stage="storing vectors", progress=75)
    logger.info("[%s] Stage 6: upserting to Qdrant", job_id)

    chunk_db_rows: list[Chunk] = []
    doc_type = structured_fields.get("documentType", "unknown")

    for i, (chunk_data, c_vec, t_vec) in enumerate(zip(chunks, content_vectors, title_vectors)):
        chunk_id = f"{document_id}_chunk_{i:04d}"

        qdrant_point_id = qdrant_store.upsert_chunk(
            chunk_id=chunk_id,
            content_vector=c_vec,
            title_vector=t_vec,
            payload={
                "document_id": document_id,
                "document_type": doc_type,
                "structural_level": chunk_data.structural_level,
                "breadcrumb": chunk_data.breadcrumb,
                "title": chunk_data.title,
                "ocr_confidence_avg": chunk_data.ocr_confidence_avg,
                "page_range": chunk_data.page_range,
            },
        )

        chunk_db_rows.append(Chunk(
            chunk_id=chunk_id,
            document_id=document_id,
            title=chunk_data.title,
            content=chunk_data.content,
            breadcrumb=chunk_data.breadcrumb,
            structural_level=chunk_data.structural_level,
            page_range_json=json.dumps(chunk_data.page_range),
            token_estimate=chunk_data.token_estimate,
            ocr_confidence_avg=chunk_data.ocr_confidence_avg,
            ocr_confidence_min=chunk_data.ocr_confidence_min,
            has_low_conf_regions=chunk_data.has_low_conf_regions,
            qdrant_point_id=qdrant_point_id,
        ))

    # ── Stage 7: Store metadata in SQLite ───────────────────────────────────
    job_store.update_job(job_id, stage="saving metadata", progress=85)
    logger.info("[%s] Stage 7: saving %d chunk rows to SQLite", job_id, len(chunk_db_rows))

    for row in chunk_db_rows:
        session.add(row)
    session.commit()

    # ── Stage 8: Build BM25 index ───────────────────────────────────────────
    job_store.update_job(job_id, stage="building BM25 index", progress=93)
    logger.info("[%s] Stage 8: building BM25 index", job_id)

    bm25_index = build_index([c.content for c in chunks])
    save_index(bm25_index, document_id)

    # ── Done ────────────────────────────────────────────────────────────────
    job_store.update_job(
        job_id,
        status="done",
        stage="complete",
        progress=100,
        result={
            "document_id": document_id,
            "page_count": routed.page_count,
            "chunk_count": len(chunks),
            "ocr_pages": routed.ocr_page_count,
            "document_type": doc_type,
        },
    )
    logger.info(
        "[%s] Pipeline complete: %d chunks, %d pages (%d OCR)",
        job_id, len(chunks), routed.page_count, routed.ocr_page_count,
    )
