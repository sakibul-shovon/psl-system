"""
Cross-encoder reranker — final quality pass over RRF candidates.

The cross-encoder sees (query, chunk_text) as a PAIR and produces a
true relevance score. This is more accurate than embedding similarity
because the model can see direct word-level relationships between the
query and the chunk — not just their independent vector representations.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  ~22MB, ~200ms for 20 pairs on CPU. Good accuracy/speed trade-off.

Loaded as a singleton (same lazy-load pattern as the embedder).
"""

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INSUFFICIENT_EVIDENCE_THRESHOLD = -3.0   # ms-marco raw logits: -10 to +10. -3 = clearly not relevant.

_model: Optional[CrossEncoder] = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        logger.info("Loading reranker model '%s'...", MODEL_NAME)
        _model = CrossEncoder(MODEL_NAME)
        logger.info("Reranker model loaded.")
    return _model


def rerank(
    query: str,
    candidates: list[dict],
    *,
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank a list of candidate chunks for a given query.

    Args:
        query:      the user's query string.
        candidates: list of dicts from RRF, each with chunk_id + payload.
                    payload must contain a "content" or we use chunk_id as fallback.
        top_k:      number of top results to return after reranking.

    Returns:
        List of top_k dicts: {chunk_id, rerank_score, payload}
        Ordered by rerank_score descending.
        Returns empty list if candidates is empty.
    """
    if not candidates:
        return []

    model = _get_model()

    # Build (query, chunk_text) pairs for the cross-encoder
    # The cross-encoder scores each pair independently
    pairs = [
        (query, item["payload"].get("content", item.get("chunk_id", "")))
        for item in candidates
    ]

    # scores is a numpy array of floats, one per pair
    scores = model.predict(pairs)

    # Attach scores to candidates and sort
    scored = [
        {
            "chunk_id": item["chunk_id"],
            "rerank_score": float(score),
            "payload": item["payload"],
        }
        for item, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    top = scored[:top_k]
    logger.debug(
        "Reranker: %d candidates → top %d (best score: %.3f)",
        len(candidates), len(top),
        top[0]["rerank_score"] if top else 0.0,
    )
    return top


def is_sufficient(reranked: list[dict]) -> bool:
    """
    Return True if the top reranked result meets the minimum relevance threshold.
    If False, the generation step should refuse rather than hallucinate.
    """
    if not reranked:
        return False
    return reranked[0]["rerank_score"] >= INSUFFICIENT_EVIDENCE_THRESHOLD
