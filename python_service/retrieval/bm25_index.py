"""
BM25 keyword index — build, save, load, search.

One BM25 index is built per document at ingest time, pickled to disk,
and loaded at retrieval time. BM25 handles exact keyword matches that
dense vector search misses: case numbers, party names, statute citations.

rank_bm25.BM25Okapi is picklable (standard Python serialization),
which is why we chose it over Elasticsearch (300MB JVM) or Whoosh.
"""

import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from python_service.config import settings

logger = logging.getLogger(__name__)


def build_index(texts: list[str]) -> BM25Okapi:
    """
    Build a BM25 index from a list of chunk texts.

    Tokenization is simple whitespace split + lowercase.
    This is intentional — legal text has important tokens like
    "Section", "4.2(b)", "$500,000" that complex tokenizers might split.

    Args:
        texts: list of chunk content strings, one per chunk.

    Returns:
        A fitted BM25Okapi index.
    """
    tokenized = [text.lower().split() for text in texts]
    index = BM25Okapi(tokenized)
    logger.debug("BM25 index built: %d documents", len(tokenized))
    return index


def save_index(index: BM25Okapi, document_id: str) -> Path:
    """
    Pickle the BM25 index to data/bm25/{document_id}.pkl.
    Returns the path it was saved to.
    """
    path = settings.bm25_dir / f"{document_id}.pkl"
    with open(path, "wb") as f:
        pickle.dump(index, f)
    logger.info("BM25 index saved: %s", path)
    return path


def load_index(document_id: str) -> BM25Okapi:
    """
    Load a previously saved BM25 index from disk.

    Raises:
        FileNotFoundError: if no index exists for this document_id.
    """
    path = settings.bm25_dir / f"{document_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No BM25 index found for document {document_id!r} at {path}")
    with open(path, "rb") as f:
        index = pickle.load(f)
    logger.debug("BM25 index loaded: %s", path)
    return index


def search(index: BM25Okapi, query: str, n: int = 20) -> list[tuple[int, float]]:
    """
    Search the BM25 index.

    Args:
        index: a loaded BM25Okapi index.
        query: the search query string.
        n: max results to return.

    Returns:
        List of (chunk_index, score) tuples, sorted by score descending.
        chunk_index maps back to the list of texts used when building the index.
        Scores of 0.0 are filtered out (no match at all).
    """
    tokens = query.lower().split()
    scores = index.get_scores(tokens)   # returns array of floats, one per document

    # Pair each score with its position, filter zeros, sort descending
    results = [
        (i, float(score))
        for i, score in enumerate(scores)
        if score > 0.0
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]
