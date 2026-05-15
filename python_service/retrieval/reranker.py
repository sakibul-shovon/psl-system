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
import threading
from typing import Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INSUFFICIENT_EVIDENCE_THRESHOLD = -3.0   # ms-marco raw logits: -10 to +10. -3 = clearly not relevant.

# Cross-encoder hard limit: 512 BERT tokens total (query + content + 3 special tokens).
# A 15-token query leaves ~494 tokens ≈ 1976 chars for content.
# We use 1800 chars as a safe window to avoid off-by-one truncation.
_CE_WINDOW_CHARS = 1800
_CE_OVERLAP_CHARS = 400   # overlap between consecutive windows

_model: Optional[CrossEncoder] = None
_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                logger.info("Loading reranker model '%s'...", MODEL_NAME)
                _model = CrossEncoder(
                    MODEL_NAME,
                    device="cpu",
                    automodel_args={"low_cpu_mem_usage": False},
                )
                logger.info("Reranker model loaded.")
    return _model


def _windows(text: str) -> list[str]:
    """
    Split text into overlapping windows that each fit the cross-encoder's
    512-token BERT limit.  Returns the original text unchanged when it is
    already short enough.
    """
    if len(text) <= _CE_WINDOW_CHARS:
        return [text]
    step = _CE_WINDOW_CHARS - _CE_OVERLAP_CHARS
    return [text[i: i + _CE_WINDOW_CHARS] for i in range(0, len(text), step)]


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

    # For each candidate, split long content into overlapping windows so the
    # cross-encoder's 512-token limit never silently drops tail content.
    # We track which candidate each pair came from to aggregate scores.
    pairs: list[tuple[str, str]] = []
    pair_to_candidate: list[int] = []   # pairs[i] belongs to candidates[pair_to_candidate[i]]

    for idx, item in enumerate(candidates):
        content = item["payload"].get("content", item.get("chunk_id", ""))
        for window in _windows(content):
            pairs.append((query, window))
            pair_to_candidate.append(idx)

    # scores is a numpy array of floats, one per pair
    raw_scores = model.predict(pairs)

    # Aggregate: take the MAX window score for each candidate
    candidate_scores = [-float("inf")] * len(candidates)
    for pair_idx, cand_idx in enumerate(pair_to_candidate):
        candidate_scores[cand_idx] = max(candidate_scores[cand_idx], float(raw_scores[pair_idx]))

    scores = candidate_scores

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
