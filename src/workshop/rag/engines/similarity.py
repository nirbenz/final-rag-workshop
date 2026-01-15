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
from numpy.typing import NDArray
from pydantic_ai import Embedder

from workshop.llm import get_embeddings_sync
from workshop.rag.engines.types import ChunkEmbedding, ChunkObject


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
    ):
        """
        Initialize similarity engine.

        Args:
            embedder: Pydantic-AI Embedder instance for generating embeddings
            similarity_threshold: Minimum similarity score (0-1) for retrieval
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

    def _embed_sync(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """
        Get embeddings using centralized get_embeddings_sync from workshop.llm.

        This provides consistent batching and error handling across all engines,
        and works correctly in both sync and async contexts (like NiceGUI).
        """
        return get_embeddings_sync(self._embedder, texts, input_type)

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

    def get_relevant_context(self, query: str, top_k: int = 10) -> Sequence[ChunkObject]:
        """
        Retrieve chunks using cosine similarity.

        Args:
            query: User query text
            top_k: Maximum number of chunks to return

        Returns:
            Top-k most similar chunks above similarity_threshold
        """
        if not self._chunks:
            return []

        query_embedding = np.array(self._embed_sync([query], input_type="query")[0])
        embeddings_matrix = np.array(self._embeddings)

        similarities = self._cosine_similarity(query_embedding, embeddings_matrix)

        mask = similarities >= self._similarity_threshold
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            return []

        filtered_similarities = similarities[filtered_indices]
        sorted_indices = filtered_indices[np.argsort(-filtered_similarities)]
        top_indices = sorted_indices[:top_k]

        return [self._chunks[i] for i in top_indices]

    def _cosine_similarity(self, query_vec: NDArray, chunk_vecs: NDArray) -> NDArray:
        """
        Compute cosine similarity between query and all chunk vectors.

        Args:
            query_vec: Query embedding vector (shape: [dim])
            chunk_vecs: Chunk embedding matrix (shape: [num_chunks, dim])

        Returns:
            Similarity scores array (shape: [num_chunks])
        """
        query_norm = np.linalg.norm(query_vec)
        chunk_norms = np.linalg.norm(chunk_vecs, axis=1)

        if query_norm == 0:
            return np.zeros(len(chunk_vecs))

        dot_products = np.dot(chunk_vecs, query_vec)
        similarities = dot_products / (query_norm * chunk_norms + 1e-10)

        return similarities

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
