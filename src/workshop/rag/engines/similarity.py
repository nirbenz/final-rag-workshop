# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com


"""
Similarity-based context engine - Phase 2.

This engine uses cosine similarity with NumPy for retrieval. It demonstrates
embedding-based retrieval without the complexity of a full vector database.

Workshop participants implement this after NaiveContextEngine to learn about:
- Embedding generation
- Similarity computation
- Top-k retrieval with thresholds
"""

from typing import List, Optional, Sequence

import numpy as np
from pydantic_ai import Embedder

from workshop.embeddings import get_embeddings_sync

# Import per-exercise toggle (separate file to avoid circular imports)
from workshop.exercise_toggles import USE_SIMILARITY_SOLUTION
from workshop.rag.engines.types import ChunkEmbedding, ChunkObject

if USE_SIMILARITY_SOLUTION:
    from workshop.rag.solutions.similarity import cosine_similarity, get_top_k
else:
    from workshop.rag.exercises.similarity import cosine_similarity, get_top_k


class SimilarityContextEngine:
    """
    Phase 2 engine: cosine similarity retrieval with NumPy.

    This engine demonstrates embedding-based retrieval without requiring
    a full vector database. Participants implement this to understand:
    - How embeddings enable semantic search
    - Cosine similarity computation
    - Top-k ranking and threshold filtering

    Key behavior:
    - add_context(): Stores chunks and their embeddings
    - get_relevant_context(): Computes similarity between query and chunks
    - Returns top-k most similar chunks above threshold
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        similarity_threshold: float = 0.0,
        max_tokens: int = 8192,
        top_k: int = 10,
    ):
        """
        Initialize similarity engine.

        Args:
            embedder: Pydantic-AI Embedder instance for generating embeddings
            similarity_threshold: Minimum similarity score (0-1) for retrieval
            max_tokens: Maximum number of tokens for the embedding model
            top_k: Default number of results to return (can be overridden per query)
        """
        if embedder is None:
            raise ValueError(
                "SimilarityContextEngine requires 'embedder' (pydantic_ai.Embedder). "
                "\nExample: embedder=Embedder('openai:text-embedding-3-small')"
            )

        self._chunks: List[ChunkObject] = []
        self._embeddings: List[ChunkEmbedding] = []
        self._similarity_threshold = similarity_threshold
        self._embedder = embedder
        self._max_tokens = max_tokens
        self._top_k = top_k

    def _embed_sync(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """
        Get embeddings using centralized get_embeddings_sync from workshop.llm.

        This provides consistent batching and error handling across all engines,
        and works correctly in both sync and async contexts (like NiceGUI).
        """
        return get_embeddings_sync(self._embedder, texts, self.max_tokens, input_type)

    def add_context(
        self, context: Sequence[ChunkObject], embeddings: Optional[Sequence[ChunkEmbedding]] = None
    ) -> None:
        """
        Store chunks and their embeddings.

        Args:
            context: Sequence of ChunkObjects to store
            embeddings: Optional pre-computed embeddings (if None, will embed chunks)
        """
        if embeddings is None:
            texts = [chunk.text for chunk in context]
            computed = self._embed_sync(texts, input_type="document")
            embeddings = [np.array(emb) for emb in computed]

        self._chunks.extend(context)
        self._embeddings.extend(embeddings)

    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> Sequence[ChunkObject]:
        """
        Retrieve chunks using cosine similarity.

        Args:
            query: User query text
            top_k: Maximum number of chunks to return (defaults to instance top_k)

        Returns:
            Top-k most similar chunks above similarity_threshold
        """
        if not self._chunks:
            return []

        if top_k is None:
            top_k = self._top_k

        query_embedding = np.array(self._embed_sync([query], input_type="query")[0])
        embeddings_matrix = np.array(self._embeddings)

        # Use imported functions (from exercises or solutions)
        similarities = cosine_similarity(query_embedding, embeddings_matrix)
        top_indices = get_top_k(similarities, self._similarity_threshold, top_k)

        return [self._chunks[i] for i in top_indices]

    @property
    def context(self) -> Sequence[ChunkObject]:
        """Get all stored chunks."""
        return self._chunks

    @property
    def context_count(self) -> int:
        """Get total number of stored chunks (for pagination)."""
        return len(self._chunks)

    def get_context_page(self, offset: int = 0, limit: int = 20) -> Sequence[ChunkObject]:
        """
        Get a page of stored chunks.

        Args:
            offset: Number of chunks to skip
            limit: Maximum number of chunks to return

        Returns:
            Sequence of ChunkObjects for the requested page
        """
        return self._chunks[offset : offset + limit]

    def clear(self) -> None:
        """Clear all stored chunks and embeddings."""
        self._chunks = []
        self._embeddings = []

    @property
    def max_tokens(self) -> int:
        """Get the maximum number of tokens for the embedding model."""
        return self._max_tokens
