# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Chunker protocol and types for RAG workshop.

Chunkers transform sequences of messages into ChunkObjects with metadata.
They expose hyperparameters as Pydantic BaseModel for automatic GUI rendering.

All chunkers inherit from BaseChunkerParams which provides conversation-level
windowing parameters (date range, token limits). This gives chunkers full
responsibility for input processing: filtering + chunking.
"""

from typing import List, Protocol, Sequence, Tuple

from pydantic import BaseModel, Field

from workshop.chat import WhatsappMessage
from workshop.rag.engines.types import ChunkObject


class BaseChunkerParams(BaseModel):
    """
    Base parameters for all chunkers - conversation-level filtering.

    These fields control which messages get chunked (windowing/filtering).
    All specific chunker params should inherit from this class.

    Chunkers are responsible for:
    1. Filtering messages by token limits and time windows
    2. Applying chunking strategy (message-count, sentence-aware, semantic, etc.)

    Note: Date range is determined from the messages after applying max_tokens/max_days
    filters and displayed in the context stats (not as input parameters).
    """

    max_tokens: int = Field(
        default=25_000,
        ge=10_000,
        le=200_000,
        json_schema_extra={
            "step": 10_000,
            "label": "Max Context Tokens",
            "tooltip": "Maximum tokens in context window",
        },
    )
    max_days: int = Field(
        default=25,
        ge=1,
        le=360,
        json_schema_extra={
            "step": 1,
            "label": "Max Days of History",
            "tooltip": "Maximum days of history to include",
        },
    )


class MessageChunkerProtocol(Protocol):
    """
    Protocol for message chunkers.

    Chunkers transform raw messages into ChunkObjects suitable for embedding
    and retrieval. They must expose a `params` attribute (Pydantic BaseModel)
    for automatic GUI rendering via render_model_controls().

    Implementations:
    - MessageCountChunker: Simple sliding window over messages
    - SentenceBoundaryChunker: Respects sentence boundaries
    - SemanticChunker: Topic-aware chunking with embedding similarity
    - SegmentingChunker: Time-gap segmentation + chunking
    - ContextualChunker: Adds conversation summary for contextual embeddings
    """

    params: BaseModel

    def chunk_messages(self, messages: Sequence[WhatsappMessage]) -> Sequence[ChunkObject]:
        """
        Transform messages into chunks with metadata.

        Args:
            messages: Sequence of MessageModel objects from conversation

        Returns:
            Sequence of ChunkObject with:
            - text: Combined message text for embedding/retrieval
            - message_ids: Indices in original message list for traceability
            - metadata: start_idx, end_idx, timestamps, speakers, etc.
        """
        ...

    def get_chunk_boundaries(self, num_messages: int) -> List[Tuple[int, int]]:
        """
        Get chunk boundaries for preview visualization.

        This is a lightweight method for GUI preview that returns (start, end)
        tuples without creating full ChunkObjects. Useful for fast refreshes
        when users adjust hyperparameters in the UI.

        Args:
            num_messages: Total number of messages in conversation

        Returns:
            List of (start_idx, end_idx) tuples for each chunk
        """
        ...
