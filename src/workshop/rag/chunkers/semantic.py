# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com


"""
Semantic chunker - Extension stub.

This chunker uses embedding similarity to detect topic boundaries, creating
chunks that are semantically coherent rather than fixed-size.

Workshop participants implement this to learn about:
- Embedding-based topic segmentation
- Cosine similarity for boundary detection
- Dynamic chunk sizing based on content
"""

from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from pydantic_ai import Embedder

from workshop.chat import WhatsappMessage
from workshop.llm import get_embeddings_sync
from workshop.rag.chunkers.types import BaseChunkerParams
from workshop.rag.engines.types import ChunkObject


class SemanticChunkerParams(BaseChunkerParams):
    """
    Hyperparameters for semantic chunker.

    Inherits conversation windowing from BaseChunkerParams.
    Adds semantic similarity-based chunking parameters.
    """

    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        json_schema_extra={"step": 0.05, "label": "Topic Similarity Threshold"},
    )
    min_chunk_size: int = Field(
        default=3,
        ge=1,
        le=20,
        json_schema_extra={"step": 1, "label": "Min Messages per Chunk"},
    )


class SemanticChunker:
    """
    Topic-aware chunker using embedding similarity.

    Implementation strategy:
    1. Embed all messages upfront
    2. Compute cosine similarities between consecutive message embeddings
    3. Find "topic boundaries" where similarity < threshold
    4. Create chunks between boundaries
    5. Merge small chunks if below min_chunk_size

    Example with similarity_threshold=0.7, min_chunk_size=3:
    Messages: ["Hi", "How are you?", "Let's talk about Python", "I love Python", "What about lunch?"]
    Similarities: [-, 0.9, 0.5, 0.85, 0.4]
    Boundaries: [2, 4]  # Indices where similarity drops below 0.7
    Chunks:
      Chunk 0: messages[0:2] - greeting topic
      Chunk 1: messages[2:4] - Python topic
      Chunk 2: messages[4:5] - lunch topic (would merge if < min_chunk_size)

    Participants need to:
    - Compute pairwise similarities between consecutive messages
    - Identify boundaries where similarity drops
    - Create chunks and merge small ones
    - Track message_ids for traceability
    """

    def __init__(
        self,
        params: Optional[SemanticChunkerParams] = None,
        embedder: Optional[Embedder] = None,
    ):
        """
        Initialize chunker with hyperparameters.

        Args:
            params: Hyperparameters (default: SemanticChunkerParams())
            embedder: Pydantic-AI Embedder instance for generating embeddings
        """
        if embedder is None:
            raise ValueError(
                "SemanticChunker requires 'embedder' (pydantic_ai.Embedder). "
                "\nExample: embedder=Embedder('openai:text-embedding-3-small')"
            )

        self.params = params or SemanticChunkerParams()
        self._embedder = embedder

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using centralized get_embeddings_sync from workshop.llm.

        This provides consistent batching and error handling across all components,
        and works correctly in both sync and async contexts (like NiceGUI).
        """
        return get_embeddings_sync(self._embedder, texts, input_type="document")

    def _embed_texts(self, texts: Sequence[str]) -> NDArray:
        """Embed multiple texts, returns matrix of shape [num_texts, dim]."""
        embeddings = self._embed_sync(list(texts))
        return np.array(embeddings)

    def chunk_messages(self, messages: Sequence[WhatsappMessage]) -> Sequence[ChunkObject]:
        """
        Transform messages into semantically coherent chunks.

        Args:
            messages: Sequence of WhatsappMessage objects from conversation

        Returns:
            Sequence of ChunkObjects with:
            - text: Combined message text for semantically coherent topic
            - message_ids: Indices of messages in this semantic chunk
            - metadata: start_idx, end_idx, timestamps, speakers

        Implementation steps:
        1. Embed all messages upfront
        2. Compute cosine similarities between consecutive embeddings
        3. Find boundaries where similarity < threshold
        4. Create chunks between boundaries
        5. Merge chunks smaller than min_chunk_size with neighbors
        6. Build ChunkObjects with proper metadata

        TODO for participants: Implement the full logic
        """
        raise NotImplementedError(
            "Participants implement this. "
            "Hints: Use self._embed_texts() to get embeddings, "
            "compute cosine similarity between consecutive messages, "
            "and use a simple greedy merging strategy for small chunks."
        )
