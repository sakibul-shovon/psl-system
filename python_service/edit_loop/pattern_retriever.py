"""
Pattern retriever — finds relevant learned patterns for a given query.

Embeds the query, searches Qdrant's learned_patterns collection, then
pulls full pattern data from SQLite (Qdrant only stores the vector +
lightweight payload; SQLite has the full few_shot examples).

Only active patterns (is_active=True) above the confidence threshold
are returned.  Patterns are ranked by semantic similarity to the query.
"""

import logging
from typing import Optional

from sqlmodel import Session

from python_service.db.models import Pattern
from python_service.db.session import engine
from python_service.embedder import embed_one
from python_service.vector.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.50
MAX_PATTERNS = 3   # inject at most 3 patterns per draft to avoid prompt bloat


def retrieve_patterns(
    query: str,
    document_type: Optional[str] = None,
    draft_type: Optional[str] = None,
    limit: int = MAX_PATTERNS,
) -> list[dict]:
    """
    Find the most relevant active patterns for a query.

    Args:
        query:         the user's draft query (e.g. "summarize compensation terms").
        document_type: optional filter — only return patterns for this doc type.
        draft_type:    optional filter — only return patterns for this draft type.
        limit:         max patterns to return (default 3).

    Returns:
        List of pattern dicts ready to pass into build_prompt(), sorted by
        relevance score descending.  Empty list if no patterns are stored yet.
    """
    try:
        vector = embed_one(query)
        hits = qdrant_store.search_similar_patterns(query_vector=vector, limit=limit * 2)
    except Exception as exc:
        logger.warning("Qdrant pattern search failed: %s", exc)
        return []

    if not hits:
        return []

    # Collect pattern_ids from Qdrant results
    pattern_ids = [
        h["payload"].get("pattern_id")
        for h in hits
        if h["payload"].get("pattern_id")
    ]

    if not pattern_ids:
        return []

    # Load full rows from SQLite (has the complete few_shot examples)
    patterns: list[dict] = []
    with Session(engine) as session:
        for pid in pattern_ids:
            p = session.get(Pattern, pid)
            if p is None or not p.is_active:
                continue
            if p.confidence < MIN_CONFIDENCE:
                continue
            patterns.append({
                "pattern_id": p.pattern_id,
                "ruleType": p.rule_type,
                "description": p.description,
                "fewShotBefore": p.few_shot_before,
                "fewShotAfter": p.few_shot_after,
                "confidence": p.confidence,
            })

    logger.info(
        "Retrieved %d pattern(s) for query: %s",
        len(patterns[:limit]),
        query[:60],
    )
    return patterns[:limit]
