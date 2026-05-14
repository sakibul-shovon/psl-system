"""
Pattern retriever — finds relevant learned patterns for a given query.

Ranks candidates by a COMPOSITE of four signals, not just semantic similarity:
  - similarity      (Qdrant cosine)            weight 0.40
  - confidence      (extractor confidence)     weight 0.25
  - frequency       (operator reinforcement)   weight 0.20
  - recency         (exponential decay)        weight 0.15

This is what makes the system prefer patterns operators have repeatedly
confirmed over one-off patterns of marginally higher semantic similarity.
Without the composite, a pattern reinforced 5 times by 3 operators loses
to a one-off pattern just because its embedding is 0.02 closer to the query.
"""

import logging
import math
from datetime import datetime
from typing import Optional

from sqlmodel import Session

from python_service.db.models import Pattern
from python_service.db.session import engine
from python_service.embedder import embed_one
from python_service.vector.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.50
MAX_PATTERNS = 3   # inject at most 3 patterns per draft to avoid prompt bloat
RECENCY_HALF_LIFE_DAYS = 30.0

# Composite weight coefficients — must sum to 1.0. Tuning surface for later.
W_SIMILARITY = 0.40
W_CONFIDENCE = 0.25
W_FREQUENCY  = 0.20
W_RECENCY    = 0.15


def _composite_score(
    qdrant_score: float,
    confidence: float,
    frequency: int,
    last_reinforced_at: datetime,
) -> float:
    """
    Weighted blend of relevance + quality + reinforcement signals.

    frequency is normalised by min(freq/10, 1.0) so 10 reinforcements saturates
    the term; 11+ doesn't dominate the formula.

    recency is exp(-days_since_reinforcement / 30) — a pattern reinforced today
    scores 1.0; a pattern last touched 30 days ago scores 0.37; 60 days → 0.14.
    """
    days = max((datetime.utcnow() - last_reinforced_at).days, 0)
    recency = math.exp(-days / RECENCY_HALF_LIFE_DAYS)
    norm_freq = min(frequency / 10.0, 1.0)
    return (
        W_SIMILARITY * qdrant_score
        + W_CONFIDENCE * confidence
        + W_FREQUENCY * norm_freq
        + W_RECENCY * recency
    )


def retrieve_patterns(
    query: str,
    document_type: Optional[str] = None,
    draft_type: Optional[str] = None,
    limit: int = MAX_PATTERNS,
) -> list[dict]:
    """
    Find the most relevant active patterns for a query.

    Over-fetches from Qdrant (limit*4) so the composite re-rank has room
    to surface frequency-rich patterns even when they're not the top
    similarity match.

    Args:
        query:         the user's draft query.
        document_type: if set, restrict patterns whose payload `document_types`
                       list overlaps with this type (avoids leakage across types).
        draft_type:    accepted for API symmetry but not currently used as a filter.
        limit:         max patterns to return (default 3).

    Returns:
        List of pattern dicts ready to pass into build_prompt(), sorted by
        compositeScore descending. Each dict surfaces both `similarity` and
        `compositeScore` so callers can inspect why a pattern ranked high.
    """
    try:
        vector = embed_one(query)
        # Over-fetch for re-ranking headroom
        hits = qdrant_store.search_similar_patterns(
            query_vector=vector,
            limit=limit * 4,
            document_types=[document_type] if document_type else None,
            active_only=True,
        )
    except Exception as exc:
        logger.warning("Qdrant pattern search failed: %s", exc)
        return []

    if not hits:
        return []

    # Load full rows from SQLite (has complete few_shot examples + freshest counters)
    # and compute composite scores per candidate.
    scored: list[tuple[Pattern, float, float]] = []  # (pattern, composite, similarity)
    with Session(engine) as session:
        for hit in hits:
            pid = hit["payload"].get("pattern_id")
            if not pid:
                continue
            p = session.get(Pattern, pid)
            if p is None or not p.is_active:
                continue
            if p.confidence < MIN_CONFIDENCE:
                continue
            similarity = float(hit.get("score", 0.0))
            composite = _composite_score(
                qdrant_score=similarity,
                confidence=p.confidence,
                frequency=p.frequency,
                last_reinforced_at=p.last_reinforced_at,
            )
            scored.append((p, composite, similarity))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:limit]

    patterns = [{
        "pattern_id": p.pattern_id,
        "ruleType": p.rule_type,
        "description": p.description,
        "fewShotBefore": p.few_shot_before,
        "fewShotAfter": p.few_shot_after,
        "confidence": p.confidence,
        "frequency": p.frequency,
        "similarity": round(sim, 4),
        "compositeScore": round(comp, 4),
    } for p, comp, sim in top]

    if patterns:
        logger.info(
            "Retrieved %d pattern(s) for query %r (top composite=%.3f, freq=%d)",
            len(patterns), query[:60],
            patterns[0]["compositeScore"], patterns[0]["frequency"],
        )
    else:
        logger.info("No patterns above confidence threshold for query %r", query[:60])

    return patterns
