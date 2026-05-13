"""
Reciprocal Rank Fusion (RRF) — combines BM25 and dense search rankings.

RRF formula:  score(chunk) = Σ  1 / (k + rank_i)
              summed across all ranking lists that contain the chunk.

k=60 is the established default from the original RRF paper (Cormack 2009).
It dampens the advantage of being ranked #1 vs #2 — small rank differences
matter less than appearing in BOTH lists.

Why ranks instead of raw scores?
  BM25 scores are 0–15. Dense cosine scores are 0–1. These scales are
  incompatible — you can't just add them. Ranks are always comparable:
  position 1 is position 1 regardless of which retrieval method produced it.
"""

import logging

logger = logging.getLogger(__name__)

RRF_K = 60   # standard constant from the original RRF paper


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    limit: int = 20,
) -> list[dict]:
    """
    Fuse multiple ranked result lists into one combined ranking.

    Args:
        *ranked_lists: any number of result lists, each ordered best-first.
                       Each item must have a "chunk_id" key.
                       Items may also carry "payload" — the first seen is kept.
        limit:         max items in the output list.

    Returns:
        Fused list of dicts: {chunk_id, rrf_score, payload}
        Ordered by rrf_score descending.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            chunk_id = item["chunk_id"]
            if chunk_id is None:
                continue

            # RRF contribution from this list: 1 / (k + rank)
            # rank is 0-indexed so rank=0 (best) gives 1/(60+0) = 0.01667
            rrf_score = 1.0 / (RRF_K + rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score

            # Keep the payload from the first list that contains this chunk
            if chunk_id not in payloads and "payload" in item:
                payloads[chunk_id] = item["payload"]

    # Sort by accumulated RRF score, highest first
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = [
        {
            "chunk_id": chunk_id,
            "rrf_score": round(rrf_score, 6),
            "payload": payloads.get(chunk_id, {}),
        }
        for chunk_id, rrf_score in sorted_chunks[:limit]
    ]

    logger.debug(
        "RRF fusion: %d lists → %d unique chunks → top %d",
        len(ranked_lists),
        len(scores),
        len(result),
    )
    return result
