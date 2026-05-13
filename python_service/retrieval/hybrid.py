"""
Hybrid retrieval orchestrator — runs the full retrieval pipeline.

Flow:
  query
    ├─► dense search (Qdrant, top-20)
    └─► BM25 search (pickled index, top-20)
          ↓
    RRF fusion → top-20 merged
          ↓
    cross-encoder rerank → top-5
          ↓
    insufficient evidence guard → refuse if best score < 0.35
"""

import logging
from dataclasses import dataclass

from python_service.retrieval.bm25_index import load_index, search as bm25_search
from python_service.retrieval.dense import search_dense
from python_service.retrieval.reranker import is_sufficient, rerank
from python_service.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class RetrievalResult:
    """
    Output of the full hybrid retrieval pipeline.

    If sufficient=False, the generation step must not proceed.
    The evidence list contains up to 5 reranked chunks.
    """
    sufficient: bool
    evidence: list[dict]          # top-5 reranked chunks with scores + payload
    query: str
    retrieval_method: str = "dense+bm25+rerank"
    diagnostic: str = ""          # populated when sufficient=False


def retrieve(
    query: str,
    document_id: str,
    *,
    top_k: int = 5,
) -> RetrievalResult:
    """
    Full hybrid retrieval for a query against one document's chunks.

    Args:
        query:       the user's question.
        document_id: which document to search.
        top_k:       evidence items to return (default 5 → [E1]..[E5]).

    Returns:
        RetrievalResult with evidence list and sufficiency flag.
    """
    # ── Dense search ────────────────────────────────────────────────────────
    dense_results = search_dense(query, limit=20, document_id=document_id)

    # ── BM25 search ─────────────────────────────────────────────────────────
    # Load the chunk texts in the same order they were indexed (needed to map
    # BM25 integer positions back to chunk_ids).
    bm25_results = _bm25_search(query, document_id, limit=20)

    # ── RRF fusion ───────────────────────────────────────────────────────────
    fused = reciprocal_rank_fusion(dense_results, bm25_results, limit=20)

    if not fused:
        return RetrievalResult(
            sufficient=False,
            evidence=[],
            query=query,
            diagnostic="No chunks found for this document. Has it been ingested?",
        )

    # ── Rerank ───────────────────────────────────────────────────────────────
    # The cross-encoder needs the full chunk content — pull it from payload.
    # The payload stored in Qdrant doesn't include content (too large for payload
    # index), so we need to fetch it from SQLite via the chunk_id.
    fused_with_content = _attach_content(fused)
    reranked = rerank(query, fused_with_content, top_k=top_k)

    # ── Insufficient evidence guard ───────────────────────────────────────────
    if not is_sufficient(reranked):
        best = reranked[0]["rerank_score"] if reranked else 0.0
        logger.warning(
            "Insufficient evidence for query %r: best rerank score %.3f < 0.35",
            query[:60], best,
        )
        return RetrievalResult(
            sufficient=False,
            evidence=reranked,
            query=query,
            diagnostic=(
                f"Top relevance score ({best:.3f}) is below the minimum threshold (0.35). "
                "The indexed document may not contain information relevant to this query."
            ),
        )

    logger.info(
        "Retrieval complete: %d evidence items (best score: %.3f)",
        len(reranked), reranked[0]["rerank_score"],
    )
    return RetrievalResult(sufficient=True, evidence=reranked, query=query)


def _bm25_search(query: str, document_id: str, limit: int) -> list[dict]:
    """
    Load the BM25 index for this document and run the query.
    Returns results in the same format as dense_search for RRF compatibility.
    """
    try:
        index = load_index(document_id)
    except FileNotFoundError:
        logger.warning("No BM25 index for document %s — skipping keyword search", document_id)
        return []

    # We need the chunk_ids in the same order they were used when building the index.
    # The pipeline stored chunks with IDs like {document_id}_chunk_0000, _0001, etc.
    # Reconstruct the ordered list from SQLite.
    chunk_ids = _get_ordered_chunk_ids(document_id)
    if not chunk_ids:
        return []

    raw_results = bm25_search(index, query, n=limit)   # [(index, score), ...]

    results = []
    for chunk_index, score in raw_results:
        if chunk_index >= len(chunk_ids):
            continue
        results.append({
            "chunk_id": chunk_ids[chunk_index],
            "score": score,
            "payload": {},   # BM25 doesn't carry payload; RRF fills it from dense results
        })

    return results


def _get_ordered_chunk_ids(document_id: str) -> list[str]:
    """Fetch chunk_ids for a document in ingest order from SQLite."""
    from sqlmodel import Session, select
    from python_service.db.models import Chunk
    from python_service.db.session import engine

    with Session(engine) as session:
        chunks = session.exec(
            select(Chunk.chunk_id)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_id)
        ).all()
    return list(chunks)


def _attach_content(fused: list[dict]) -> list[dict]:
    """
    Fetch the full content text for each chunk from SQLite.
    The reranker needs the actual text to score (query, content) pairs.
    """
    from sqlmodel import Session, select
    from python_service.db.models import Chunk
    from python_service.db.session import engine

    chunk_ids = [item["chunk_id"] for item in fused]

    with Session(engine) as session:
        rows = session.exec(
            select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))
        ).all()

    content_map = {row.chunk_id: row.content for row in rows}

    enriched = []
    for item in fused:
        content = content_map.get(item["chunk_id"], "")
        enriched.append({
            **item,
            "payload": {**item["payload"], "content": content},
        })
    return enriched
