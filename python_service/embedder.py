"""
Sentence-transformers embedding model — singleton.

BAAI/bge-base-en-v1.5 produces 768-dimensional vectors.
We load it once at first use and reuse it for every embed call.

Why lazy loading (not at import time)?
  Importing this module doesn't load the model. The model loads on the
  first call to embed_texts(). This means the FastAPI server starts instantly
  even before the 438MB model is downloaded/cached. The first upload request
  will be slower (~5s extra) while the model loads; all subsequent ones are fast.

Why normalize_embeddings=True?
  Qdrant uses cosine similarity. Cosine similarity between two normalized
  vectors equals their dot product — faster computation. bge-base-en-v1.5
  is specifically trained for normalized cosine similarity retrieval.
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' (first use — may take a moment)...", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded. Vector dim: %d", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts. Returns one 768-dim vector per text.

    Batching is handled internally by sentence-transformers.
    Passing all texts at once is faster than calling embed_one() in a loop.

    Args:
        texts: list of strings to embed.

    Returns:
        List of float lists, same length as input.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    """Embed a single string. Use embed_texts() for batches."""
    return embed_texts([text])[0]
