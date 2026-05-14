"""
Qdrant vector store — collection management and search helpers.

Two collections:
  - legal_chunks: stores chunk content+title vectors (768-dim from bge-base-en-v1.5)
  - learned_patterns: stores pattern principle vectors (768-dim)

Named vectors let us embed both content and title per chunk and query either
independently (e.g., "title-boost" query searches title vectors for section headers).

This module is imported by:
  - main.py lifespan (bootstrap collections)
  - ingestion pipeline (upsert chunks)
  - retrieval module (search chunks)
  - edit_loop (upsert + search patterns)
"""

import logging
from typing import Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from python_service.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768   # BAAI/bge-base-en-v1.5 output dimension

CHUNKS_COLLECTION = "legal_chunks"
PATTERNS_COLLECTION = "learned_patterns"
EPISODIC_COLLECTION = "episodic_memory"


class QdrantStore:
    """
    Thin wrapper around QdrantClient that manages our two collections
    and exposes typed upsert/search methods.
    """

    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
        logger.info("Qdrant client connected to %s", settings.qdrant_url)

    # ── Collection bootstrap ────────────────────────────────────────────────

    def ensure_collections(self) -> None:
        """
        Create collections if they don't exist. Safe to call on every startup.
        Existing collections and their data are left untouched.
        """
        self._ensure_chunks_collection()
        self._ensure_patterns_collection()
        self._ensure_episodic_collection()

    def _ensure_chunks_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if CHUNKS_COLLECTION in existing:
            logger.info("Collection '%s' already exists", CHUNKS_COLLECTION)
            return

        # Named vectors: we embed both the chunk content and its section title.
        # `content` vector is the primary search target.
        # `title` vector enables title-only queries (useful for structured retrieval).
        self.client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            vectors_config={
                "content": qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
                "title": qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            },
        )

        # Payload indexes — enable fast server-side filtering.
        # Without these, Qdrant scans every point; with them it uses an index.
        for field, schema_type in [
            ("document_id", qmodels.PayloadSchemaType.KEYWORD),
            ("document_type", qmodels.PayloadSchemaType.KEYWORD),
            ("structural_level", qmodels.PayloadSchemaType.INTEGER),
            ("ocr_confidence_avg", qmodels.PayloadSchemaType.FLOAT),
        ]:
            self.client.create_payload_index(
                collection_name=CHUNKS_COLLECTION,
                field_name=field,
                field_schema=schema_type,
            )

        logger.info("Created collection '%s' with payload indexes", CHUNKS_COLLECTION)

    def _ensure_patterns_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if PATTERNS_COLLECTION in existing:
            logger.info("Collection '%s' already exists", PATTERNS_COLLECTION)
            return

        self.client.create_collection(
            collection_name=PATTERNS_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )

        for field, schema_type in [
            ("pattern_id", qmodels.PayloadSchemaType.KEYWORD),
            ("rule_type", qmodels.PayloadSchemaType.KEYWORD),
            ("is_active", qmodels.PayloadSchemaType.BOOL),
        ]:
            self.client.create_payload_index(
                collection_name=PATTERNS_COLLECTION,
                field_name=field,
                field_schema=schema_type,
            )

        logger.info("Created collection '%s'", PATTERNS_COLLECTION)

    def _ensure_episodic_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if EPISODIC_COLLECTION in existing:
            logger.info("Collection '%s' already exists", EPISODIC_COLLECTION)
            return

        # Each episodic memory is embedded as "query | document_type" (768-dim).
        # At retrieval time we search by the new draft's query+doc_type to find
        # the most similar past sessions — essentially "how did I handle this before?"
        self.client.create_collection(
            collection_name=EPISODIC_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )

        for field, schema_type in [
            ("memory_id", qmodels.PayloadSchemaType.KEYWORD),
            ("document_type", qmodels.PayloadSchemaType.KEYWORD),
            ("draft_type", qmodels.PayloadSchemaType.KEYWORD),
        ]:
            self.client.create_payload_index(
                collection_name=EPISODIC_COLLECTION,
                field_name=field,
                field_schema=schema_type,
            )

        logger.info("Created collection '%s'", EPISODIC_COLLECTION)

    # ── Chunk operations ────────────────────────────────────────────────────

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        content_vector: list[float],
        title_vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        """
        Insert or overwrite a chunk point in Qdrant.
        Returns the qdrant_point_id (a UUID string stored as payload too).
        """
        point_id = str(uuid4())
        payload["qdrant_point_id"] = point_id
        payload["chunk_id"] = chunk_id

        self.client.upsert(
            collection_name=CHUNKS_COLLECTION,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector={"content": content_vector, "title": title_vector},
                    payload=payload,
                )
            ],
        )
        return point_id

    def search_chunks(
        self,
        *,
        query_vector: list[float],
        limit: int = 20,
        document_id: Optional[str] = None,
        vector_name: str = "content",
    ) -> list[dict[str, Any]]:
        """
        Dense vector search over legal_chunks.

        Args:
            query_vector: embedded query (768-dim).
            limit: max results.
            document_id: if set, filter to one document only.
            vector_name: "content" (default) or "title".

        Returns:
            List of dicts: {chunk_id, score, payload}
        """
        query_filter = None
        if document_id:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id",
                        match=qmodels.MatchValue(value=document_id),
                    )
                ]
            )

        results = self.client.search(
            collection_name=CHUNKS_COLLECTION,
            query_vector=qmodels.NamedVector(name=vector_name, vector=query_vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {"chunk_id": r.payload.get("chunk_id"), "score": r.score, "payload": r.payload}
            for r in results
        ]

    # ── Pattern operations ──────────────────────────────────────────────────

    def upsert_pattern(
        self,
        *,
        pattern_id: str,
        principle_vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        """Insert or overwrite a pattern point. Returns qdrant_point_id."""
        point_id = str(uuid4())
        payload["qdrant_point_id"] = point_id
        payload["pattern_id"] = pattern_id

        self.client.upsert(
            collection_name=PATTERNS_COLLECTION,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=principle_vector,
                    payload=payload,
                )
            ],
        )
        return point_id

    def search_similar_patterns(
        self,
        *,
        query_vector: list[float],
        limit: int = 10,
        document_types: Optional[list[str]] = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search learned_patterns. Used for:
          - Dedup check at storage time (look for cosine > 0.85)
          - Pattern retrieval at generation time (top-10 → composite re-rank → top-5)

        `document_types` filter restricts results to patterns whose stored
        `document_types` payload field overlaps with the provided list.
        Without this, a pattern learned from a lease would be retrieved for
        an employment contract draft.
        """
        must_conditions = []

        if active_only:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="is_active",
                    match=qmodels.MatchValue(value=True),
                )
            )

        if document_types:
            # MatchAny matches when the payload list contains ANY of the values.
            # We use it because the stored payload is a list ("document_types"),
            # not a single keyword.
            must_conditions.append(
                qmodels.FieldCondition(
                    key="document_types",
                    match=qmodels.MatchAny(any=document_types),
                )
            )

        query_filter = qmodels.Filter(must=must_conditions) if must_conditions else None

        results = self.client.search(
            collection_name=PATTERNS_COLLECTION,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {"pattern_id": r.payload.get("pattern_id"), "score": r.score, "payload": r.payload}
            for r in results
        ]

    def update_pattern_payload(self, point_id: str, payload_update: dict[str, Any]) -> None:
        """Partial payload update — used when reinforcing an existing pattern."""
        self.client.set_payload(
            collection_name=PATTERNS_COLLECTION,
            payload=payload_update,
            points=[point_id],
        )

    def delete_chunks_for_document(self, document_id: str) -> None:
        """Remove all chunk vectors for a document (for re-ingestion)."""
        self.client.delete(
            collection_name=CHUNKS_COLLECTION,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted chunk vectors for document %s", document_id)

    # ── Episodic memory operations ──────────────────────────────────────────

    def upsert_episodic_memory(
        self,
        *,
        memory_id: str,
        query_vector: list[float],
        payload: dict[str, Any],
    ) -> str:
        """
        Store an episodic memory point.

        The vector is the embedding of "query | document_type". At retrieval
        time we search by the same embedding of the new draft's context, so
        the planner gets examples of how similar queries played out before.
        """
        point_id = str(uuid4())
        payload["qdrant_point_id"] = point_id
        payload["memory_id"] = memory_id

        self.client.upsert(
            collection_name=EPISODIC_COLLECTION,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload=payload,
                )
            ],
        )
        return point_id

    def search_episodic_memories(
        self,
        *,
        query_vector: list[float],
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Find the top-k most similar past draft sessions.

        Returns dicts with keys: memory_id, score, payload.
        Used by the planner to inject prior-session context into its prompt.
        """
        results = self.client.search(
            collection_name=EPISODIC_COLLECTION,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {"memory_id": r.payload.get("memory_id"), "score": r.score, "payload": r.payload}
            for r in results
        ]


# Module-level singleton — import this everywhere.
# Constructed at module load; if Qdrant is down, connection errors
# appear as clear exceptions the first time a method is called.
qdrant_store = QdrantStore()
