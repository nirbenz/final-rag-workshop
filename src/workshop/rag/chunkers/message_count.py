# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Message-count-based chunker with overlap support.

This is the baseline chunker that workshop participants start with. It uses
a simple sliding window over messages with configurable overlap.
"""

from typing import List, Optional, Sequence, Tuple

from pydantic import Field

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers.types import BaseChunkerParams
from workshop.rag.engines.types import ChunkObject


class MessageCountParams(BaseChunkerParams):
    """
    Hyperparameters for message-count chunker.

    Inherits conversation windowing parameters from BaseChunkerParams:
    - max_tokens: Maximum tokens in context window
    - max_days: Maximum days of history
    - start_date: Optional start date filter
    - end_date: Optional end date filter

    Adds chunking-specific parameters:
    - chunk_length: Number of messages per chunk
    - chunk_overlap: Number of overlapping messages between chunks

    The json_schema_extra metadata controls the UI widget type and constraints.
    """

    chunk_length: int = Field(default=6, ge=1, le=50, json_schema_extra={"step": 1, "label": "Messages per Chunk"})
    chunk_overlap: int = Field(default=4, ge=0, le=10, json_schema_extra={"step": 1, "label": "Overlap Messages"})


class MessageCountChunker:
    """
    Message-count-based chunker with overlap support.

    Implements a sliding window over messages with configurable chunk size
    and overlap. This is the simplest chunking strategy and serves as the
    baseline for workshop participants.

    Chunking strategy:
    - Window of N messages moves across conversation
    - Each window becomes a chunk
    - Windows overlap by M messages
    - Stride = chunk_length - chunk_overlap

    Example with chunk_length=5, chunk_overlap=2:
    Messages: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Chunk 0: [0, 1, 2, 3, 4]  # Messages 0-4
    Chunk 1: [3, 4, 5, 6, 7]  # Messages 3-7 (overlap 2)
    Chunk 2: [6, 7, 8, 9]     # Messages 6-9 (last chunk)
    """

    def __init__(self, params: Optional[MessageCountParams] = None):
        """
        Initialize chunker with hyperparameters.

        Args:
            params: Hyperparameters (default: MessageCountParams())
        """
        self.params = params or MessageCountParams()

    def chunk_messages(self, messages: Sequence[WhatsappMessage]) -> Sequence[ChunkObject]:
        """
        Transform messages into chunks using sliding window.

        Args:
            messages: Sequence of WhatsappMessage objects from conversation

        Returns:
            Sequence of ChunkObjects with:
            - text: Combined message text (user: message\\n...)
            - message_ids: Indices of messages in this chunk
            - metadata: start_idx, end_idx, start_time, end_time, speakers
        """
        if not messages:
            return []

        chunks: List[ChunkObject] = []
        chunk_length = max(1, self.params.chunk_length)
        overlap = max(0, min(self.params.chunk_overlap, chunk_length - 1))
        stride = chunk_length - overlap

        i = 0
        chunk_id = 0

        while i < len(messages):
            end = min(i + chunk_length, len(messages))
            chunk_messages = messages[i:end]

            # Combine message text (format: "user: message")
            text_parts = []
            for msg in chunk_messages:
                # Use compact_form() if available, otherwise format manually
                if hasattr(msg, "compact_form"):
                    text_parts.append(msg.compact_form())
                else:
                    text_parts.append(f"{msg.user}: {msg.text}")

            text = "\n".join(text_parts)

            # Extract metadata
            speakers = {msg.user for msg in chunk_messages}

            # Convert timestamps to datetime (handle pandas Timestamp objects)
            start_time = chunk_messages[0].timestamp
            end_time = chunk_messages[-1].timestamp
            if hasattr(start_time, "to_pydatetime"):
                start_time = start_time.to_pydatetime()  # pyright: ignore[reportAttributeAccessIssue]
            if hasattr(end_time, "to_pydatetime"):
                end_time = end_time.to_pydatetime()  # pyright: ignore[reportAttributeAccessIssue]

            chunk = ChunkObject(
                id=f"chunk_{chunk_id}",
                text=text,
                message_ids=list(range(i, end)),
                metadata={
                    "start_idx": i,
                    "end_idx": end,
                    "start_time": start_time,
                    "end_time": end_time,
                    "speakers": list(speakers),
                },
            )
            chunks.append(chunk)
            chunk_id += 1

            i += stride
            if i >= len(messages):
                break

        return chunks

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

        Example:
            With chunk_length=5, chunk_overlap=2, num_messages=12:
            Returns [(0, 5), (3, 8), (6, 11), (9, 12)]
        """
        if num_messages == 0:
            return []

        chunk_length = max(1, self.params.chunk_length)
        overlap = max(0, min(self.params.chunk_overlap, chunk_length - 1))
        stride = chunk_length - overlap

        boundaries: List[Tuple[int, int]] = []
        i = 0
        while i < num_messages:
            end = min(i + chunk_length, num_messages)
            boundaries.append((i, end))
            i += stride
            if i >= num_messages:
                break

        return boundaries
