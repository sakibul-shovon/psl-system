"""
Dense vector search — Qdrant wrapper for query-time retrieval.

Embeds the query string and searches the legal_chunks collection.
Returns top-N results with chunk_id, score, and full payload.
"""

import logging

from python_service.embedder import embed_one
from python_service.vector.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)


def search_dense(
    query: str,
    *,
    limit: int = 20,
    document_id: str | None = None,
) -> list[dict]:
    """
    Embed query and search Qdrant for the most semantically similar chunks.

    Args:
        query:       the user's question or search string.
        limit:       max results to return (default 20 — RRF narrows this down).
        document_id: if set, restrict search to one document only.

    Returns:
        List of dicts: {chunk_id, score, payload}
        Ordered by cosine similarity descending.
    """
    query_vector = embed_one(query)

    results = qdrant_store.search_chunks(
        query_vector=query_vector,
        limit=limit,
        document_id=document_id,
        vector_name="content",
    )

    logger.debug("Dense search: %d results for query %r", len(results), query[:60])
    return results
