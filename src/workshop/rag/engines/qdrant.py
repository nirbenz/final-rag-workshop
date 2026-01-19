# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
RAG Context Engine - Phase 3.

This engine uses Qdrant for persistent vector storage with ANN search.
Embedding generation uses pydantic-ai's Embedder with automatic batching.

Workshop participants implement this after SimilarityContextEngine to learn about:
- Vector database architecture
- Approximate Nearest Neighbor (ANN) search
- Persistent storage and indexing
"""

import shutil
from typing import List, Optional, Sequence
from uuid import uuid4

from pydantic_ai import Embedder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from workshop.llm import get_embeddings_sync
from workshop.rag.engines.types import ChunkEmbedding, ChunkObject


class RAGContextEngine:
    """
    Phase 3 engine: Qdrant vector database with ANN search.

    Uses:
    - Pydantic-AI Embedder for embedding generation (handles batching internally)
    - Qdrant local mode for vector storage (no Docker required)

    Key behavior:
    - add_context(): Stores chunks in Qdrant with vector index
    - get_relevant_context(): ANN search for top-k most similar chunks
    - Persistent storage survives process restart
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        db_path: str = ".qdrant",
        collection_name: str = "chunks",
        max_tokens: int = 8192,
    ):
        """
        Initialize Qdrant engine.

        Args:
            embedder: Pydantic-AI Embedder instance for generating embeddings
            db_path: Path to Qdrant database directory
            collection_name: Name of the collection to store chunks
            max_tokens: Maximum number of tokens for the embedding model
        """
        if embedder is None:
            raise ValueError(
                "RAGContextEngine requires 'embedder' (pydantic_ai.Embedder). "
                "\nExample: embedder=Embedder('openai:text-embedding-3-small')"
            )

        self._embedder = embedder
        self._max_tokens = max_tokens
        # Qdrant local mode - file-based, no server needed
        self._db_path = db_path
        self._client = QdrantClient(path=db_path)
        self._collection_name = collection_name
        self._embed_dim: Optional[int] = None

    def _embed_sync(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """
        Get embeddings using centralized get_embeddings_sync from workshop.llm.

        This provides consistent batching and error handling across all engines,
        and works correctly in both sync and async contexts (like NiceGUI).
        """
        return get_embeddings_sync(self._embedder, texts, self.max_tokens, input_type)

    def _ensure_collection(self, dim: int) -> None:
        """Create collection if it doesn't exist or has wrong dimensions."""
        needs_recreate = False

        if self._client.collection_exists(self._collection_name):
            info = self._client.get_collection(self._collection_name)
            existing_dim = info.config.params.vectors.size  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
            if existing_dim != dim:
                needs_recreate = True
                self._client.delete_collection(self._collection_name)
        else:
            needs_recreate = True

        if needs_recreate:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        self._embed_dim = dim

    def add_context(
        self, context: Sequence[ChunkObject], embeddings: Optional[Sequence[ChunkEmbedding]] = None
    ) -> None:
        """
        Store chunks in Qdrant with their embeddings.

        Args:
            context: Sequence of ChunkObjects to store
            embeddings: Optional pre-computed embeddings. If None, embeddings will be computed.
        """
        if not context:
            return

        if embeddings is None:
            texts = [chunk.text for chunk in context]
            embeddings = self._embed_sync(texts, input_type="document")  # pyright: ignore[reportAssignmentType]

        # Ensure collection exists with correct dimension
        self._ensure_collection(len(embeddings[0]))  # pyright: ignore[reportOptionalSubscript]

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=list(emb) if hasattr(emb, "__iter__") else emb,
                payload={"chunk_json": chunk.model_dump_json()},
            )
            for chunk, emb in zip(context, embeddings)  # pyright: ignore[reportArgumentType]
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)

    def get_relevant_context(self, query: str, top_k: int = 10) -> Sequence[ChunkObject]:
        """
        Retrieve chunks using Qdrant ANN search.

        Args:
            query: User query text
            top_k: Maximum number of chunks to return

        Returns:
            Top-k most similar chunks from ANN search
        """
        if not self._client.collection_exists(self._collection_name):
            return []

        info = self._client.get_collection(self._collection_name)
        if info.points_count == 0:
            return []

        query_vec = self._embed_sync([query], input_type="query")[0]

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vec,
            limit=top_k,
        )

        return [ChunkObject.model_validate_json(hit.payload["chunk_json"]) for hit in results.points]  # pyright: ignore[reportOptionalSubscript]

    def clear(self) -> None:
        """Clear all stored chunks by deleting and recreating collection."""
        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

    def delete_storage(self) -> None:
        """Delete the entire Qdrant storage directory."""
        self._client.close()
        shutil.rmtree(self._db_path, ignore_errors=True)
        self._client = QdrantClient(path=self._db_path)

    @property
    def context(self) -> Sequence[ChunkObject]:
        """Get all stored chunks."""
        if not self._client.collection_exists(self._collection_name):
            return []

        # Scroll through all points
        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=10000,
            with_payload=True,
        )

        return [ChunkObject.model_validate_json(r.payload["chunk_json"]) for r in records]  # pyright: ignore[reportOptionalSubscript]

    @property
    def context_count(self) -> int:
        """Get total number of stored chunks (for pagination)."""
        if not self._client.collection_exists(self._collection_name):
            return 0
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    def get_context_page(self, offset: int = 0, limit: int = 20) -> Sequence[ChunkObject]:
        """
        Get a page of stored chunks using Qdrant scroll.

        Args:
            offset: Number of chunks to skip
            limit: Maximum number of chunks to return

        Returns:
            Sequence of ChunkObjects for the requested page
        """
        if not self._client.collection_exists(self._collection_name):
            return []

        # Qdrant scroll doesn't support offset directly, so we scroll and skip
        # For better performance with large offsets, we could use scroll with offset_id
        records, _ = self._client.scroll(
            collection_name=self._collection_name,
            limit=offset + limit,
            with_payload=True,
        )

        # Skip the offset records and take only limit
        page_records = records[offset : offset + limit]
        return [ChunkObject.model_validate_json(r.payload["chunk_json"]) for r in page_records]  # pyright: ignore[reportOptionalSubscript]

    @property
    def max_tokens(self) -> int:
        """Get the maximum number of tokens for the embedding model."""

        # TODO don't use model's max tokens, use the max tokens passed to the engine
        # if self._max_tokens is not None:
        #     # Check model's maximum input tokens (returns None if unknown)
        #     async def async_task() -> int | None:
        #         max_tokens = await self._embedder.max_input_tokens()
        #         return max_tokens

        #     max_tokens = asyncio.run(async_task())
        #     self._max_tokens = max_tokens

        return self._max_tokens
