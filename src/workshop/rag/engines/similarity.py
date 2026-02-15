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

import importlib
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pydantic_ai import Embedder

from workshop.embeddings import get_embeddings_sync
from workshop.rag.engines.types import ChunkEmbedding, ChunkObject


def _load_similarity_functions() -> Tuple[Callable, Callable]:
    """
    Dynamically reload and return similarity exercise functions.

    Reloads the exercise_toggles and appropriate similarity module to
    pick up code changes without restarting the app.

    Returns:
        Tuple of (cosine_similarity, get_top_k) functions
    """
    import workshop.exercise_toggles as toggles_mod

    importlib.reload(toggles_mod)

    if toggles_mod.USE_SIMILARITY_SOLUTION:
        import workshop.rag.solutions.similarity as mod
    else:
        import workshop.rag.exercises.similarity as mod

    importlib.reload(mod)
    return mod.cosine_similarity, mod.get_top_k


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

    requires_embedder: bool = True

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
        self._chunk_index: Dict[str, int] = {}
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
        Store chunks and their embeddings with upsert semantics by chunk.id.

        Args:
            context: Sequence of ChunkObjects to store
            embeddings: Optional pre-computed embeddings (if None, will embed chunks)
        """
        if embeddings is None:
            texts = [chunk.text for chunk in context]
            computed = self._embed_sync(texts, input_type="document")
            embeddings = [np.array(emb) for emb in computed]

        for chunk, emb in zip(context, embeddings):
            if chunk.id in self._chunk_index:
                idx = self._chunk_index[chunk.id]
                self._chunks[idx] = chunk
                self._embeddings[idx] = emb
            else:
                self._chunk_index[chunk.id] = len(self._chunks)
                self._chunks.append(chunk)
                self._embeddings.append(emb)

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

        cosine_similarity, get_top_k = _load_similarity_functions()
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
        self._chunk_index = {}

    @property
    def max_tokens(self) -> int:
        """Get the maximum number of tokens for the embedding model."""
        return self._max_tokens
