# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Naive context engine - Phase 1 baseline.

This is a simple passthrough engine that stores all chunks and returns all
chunks on retrieval (ignoring the query entirely). It demonstrates that
chunking and retrieval are separable concerns.

Workshop participants always use this engine in Phase 1 to focus on chunking
strategies without worrying about retrieval quality.
"""

from typing import Dict, Optional, Sequence

from workshop.rag.engines.types import ChunkEmbedding, ChunkObject


class NaiveContextEngine:
    """
    Baseline context engine: stores chunks, returns ALL on retrieval.

    This is the passthrough engine that participants always use in Phase 1.
    It demonstrates that chunking and retrieval are separable concerns.

    Key behavior:
    - add_context(): Stores chunks keyed by chunk.id (upsert semantics)
    - get_relevant_context(): Ignores query, returns ALL chunks
    - No embeddings, no indexing, no filtering
    """

    def __init__(self):
        """Initialize empty context storage."""
        self._context: Dict[str, ChunkObject] = {}

    def add_context(
        self, context: Sequence[ChunkObject], embeddings: Optional[Sequence[ChunkEmbedding]] = None
    ) -> None:
        """
        Store chunks by chunk.id with upsert semantics (embeddings ignored).

        Args:
            context: Sequence of ChunkObjects to store
            embeddings: Ignored (no embeddings needed for naive engine)
        """
        for chunk in context:
            self._context[chunk.id] = chunk

    def get_relevant_context(self, query: str, top_k: int = 10) -> Sequence[ChunkObject]:
        """
        Retrieve chunks (query ignored, returns all).

        Args:
            query: User query text (ignored)
            top_k: Maximum chunks to return (ignored, returns all)

        Returns:
            All stored chunks (entire conversation context)
        """
        return list(self._context.values())

    @property
    def context(self) -> Sequence[ChunkObject]:
        """Get all stored chunks."""
        return list(self._context.values())

    @property
    def context_count(self) -> int:
        """Get total number of stored chunks (for pagination)."""
        return len(self._context)

    def get_context_page(self, offset: int = 0, limit: int = 20) -> Sequence[ChunkObject]:
        """
        Get a page of stored chunks.

        Args:
            offset: Number of chunks to skip
            limit: Maximum number of chunks to return

        Returns:
            Sequence of ChunkObjects for the requested page
        """
        chunks = list(self._context.values())
        return chunks[offset : offset + limit]

    def clear(self) -> None:
        """Clear all stored chunks."""
        self._context = {}
