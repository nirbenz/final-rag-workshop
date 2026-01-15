# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field

ChunkEmbedding = npt.NDArray[np.float32 | np.float64]


class ChunkObject(BaseModel):
    """
    A chunk of conversation with traceability to original messages.

    Chunks are the atomic units for embedding and retrieval. They combine
    multiple messages into coherent text segments while maintaining references
    to the original messages for highlighting and analysis.

    Standard metadata keys:
    - start_idx: int - First message index in original conversation
    - end_idx: int - Last message index (exclusive)
    - start_time: datetime - Timestamp of first message
    - end_time: datetime - Timestamp of last message
    - speakers: List[str] - Unique speakers in this chunk
    - segment_id: int (optional) - Conversation segment identifier
    - embedding_context: str (optional) - Additional context for embedding only
      (stripped before returning from get_relevant_context for contextual embeddings)
    """

    id: str
    text: str  # Combined text for embedding/retrieval
    message_ids: List[int]  # Indices in original message list for traceability
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_messages(self, all_messages: Sequence) -> Sequence:
        """
        Retrieve original messages that formed this chunk.

        Args:
            all_messages: The full sequence of MessageModel objects

        Returns:
            Sequence of MessageModel objects referenced by message_ids
        """
        return [all_messages[i] for i in self.message_ids]


class ContextEngineProtocol(Protocol):
    """
    Protocol for context engines (storage + retrieval).

    Engines store ChunkObjects and retrieve relevant ones for queries.
    They handle embedding, indexing, and similarity search.

    Implementations:
    - NaiveContextEngine: Stores all, returns all (baseline passthrough)
    - SimilarityContextEngine: Cosine similarity with NumPy
    - RAGContextEngine: Qdrant vector database with ANN search
    """

    def __init__(self, *args: Any, **kwargs: Any): ...

    def add_context(
        self, context: Sequence[ChunkObject], embeddings: Optional[Sequence[ChunkEmbedding]] = None
    ) -> None:
        """
        Store chunks (and optionally pre-computed embeddings).

        Args:
            context: Sequence of ChunkObjects to store
            embeddings: Optional pre-computed embeddings (avoids re-embedding)
        """
        ...

    def get_relevant_context(self, query: str, top_k: int = 10) -> Sequence[ChunkObject]:
        """
        Retrieve relevant chunks for query.

        Args:
            query: User query text
            top_k: Maximum number of chunks to return

        Returns:
            Sequence of most relevant ChunkObjects
        """
        ...

    @property
    def context(self) -> Sequence[ChunkObject]:
        """Get all stored chunks."""
        ...

    @property
    def context_count(self) -> int:
        """Get total number of stored chunks (for pagination)."""
        ...

    def get_context_page(self, offset: int = 0, limit: int = 20) -> Sequence[ChunkObject]:
        """
        Get a page of stored chunks.

        Args:
            offset: Number of chunks to skip
            limit: Maximum number of chunks to return

        Returns:
            Sequence of ChunkObjects for the requested page
        """
        ...

    def clear(self) -> None:
        """Clear all stored chunks and embeddings."""
        ...
