# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Shared utilities for chunkers.

This module provides reusable chunking logic that can be shared across
different chunker implementations. Key functionality:

- Sliding window boundary computation
- Message-to-ChunkObject conversion with metadata
- Chunk boundary calculation for UI preview

These utilities enable chunkers like SegmentingChunker to reuse the
message-count chunking logic without duplicating code.
"""

from typing import List, Sequence, Set, Tuple

from workshop.chat import WhatsappMessage
from workshop.rag.engines.types import ChunkObject


def compute_sliding_window_boundaries(
    num_messages: int,
    chunk_length: int,
    chunk_overlap: int,
) -> List[Tuple[int, int]]:
    """
    Compute chunk boundaries using sliding window algorithm.

    This is a pure function that calculates (start, end) tuples for each chunk
    without needing access to the actual messages. Useful for:
    - Fast UI preview updates
    - Pre-computing chunk structure before processing

    Args:
        num_messages: Total number of messages
        chunk_length: Number of messages per chunk
        chunk_overlap: Number of overlapping messages between chunks

    Returns:
        List of (start_idx, end_idx) tuples for each chunk

    Example:
        >>> compute_sliding_window_boundaries(12, chunk_length=5, chunk_overlap=2)
        [(0, 5), (3, 8), (6, 11), (9, 12)]
    """
    if num_messages == 0:
        return []

    chunk_length = max(1, chunk_length)
    overlap = max(0, min(chunk_overlap, chunk_length - 1))
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


def messages_to_chunk(
    messages: Sequence[WhatsappMessage],
    start_idx: int,
    end_idx: int,
    chunk_id: str,
    extra_metadata: dict | None = None,
) -> ChunkObject:
    """
    Convert a slice of messages to a ChunkObject with metadata.

    This function handles:
    - Text formatting (user: message format)
    - Timestamp conversion (pandas Timestamp -> datetime)
    - Speaker extraction
    - Message ID tracking for traceability

    Args:
        messages: Full sequence of messages (for slicing)
        start_idx: Start index in messages
        end_idx: End index in messages (exclusive)
        chunk_id: Unique identifier for the chunk
        extra_metadata: Additional metadata to include (e.g., segment_id)

    Returns:
        ChunkObject with text, message_ids, and metadata
    """
    chunk_messages = messages[start_idx:end_idx]

    # Combine message text (format: "user: message")
    text_parts = []
    for msg in chunk_messages:
        if hasattr(msg, "compact_form"):
            text_parts.append(msg.compact_form())
        else:
            text_parts.append(f"{msg.user}: {msg.text}")

    text = "\n".join(text_parts)

    # Extract speakers (filter out None values)
    speakers: Set[str] = {msg.user for msg in chunk_messages if msg.user is not None}

    # Convert timestamps to datetime (handle pandas Timestamp objects)
    start_time = chunk_messages[0].timestamp
    end_time = chunk_messages[-1].timestamp
    if hasattr(start_time, "to_pydatetime"):
        start_time = start_time.to_pydatetime()  # pyright: ignore[reportAttributeAccessIssue]
    if hasattr(end_time, "to_pydatetime"):
        end_time = end_time.to_pydatetime()  # pyright: ignore[reportAttributeAccessIssue]

    # Build metadata
    metadata = {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_time": start_time,
        "end_time": end_time,
        "speakers": list(speakers),
    }

    # Merge extra metadata if provided
    if extra_metadata:
        metadata.update(extra_metadata)

    return ChunkObject(
        id=chunk_id,
        text=text,
        message_ids=list(range(start_idx, end_idx)),
        metadata=metadata,
    )


def chunk_messages_sliding_window(
    messages: Sequence[WhatsappMessage],
    chunk_length: int,
    chunk_overlap: int,
    chunk_id_prefix: str = "chunk",
    global_offset: int = 0,
    extra_metadata: dict | None = None,
) -> List[ChunkObject]:
    """
    Chunk messages using sliding window algorithm.

    This is the core chunking function that can be reused by multiple chunkers.
    It applies the sliding window algorithm and creates ChunkObjects with
    proper metadata.

    Args:
        messages: Sequence of messages to chunk
        chunk_length: Number of messages per chunk
        chunk_overlap: Number of overlapping messages between chunks
        chunk_id_prefix: Prefix for chunk IDs
        global_offset: Offset to add to message indices (for tracking global position)
        extra_metadata: Additional metadata to include in all chunks

    Returns:
        List of ChunkObjects

    Example:
        >>> # Chunk segment 2 (messages 10-20) with segment_id in metadata
        >>> chunks = chunk_messages_sliding_window(
        ...     messages[10:20],
        ...     chunk_length=5,
        ...     chunk_overlap=2,
        ...     chunk_id_prefix="seg2_chunk",
        ...     global_offset=10,
        ...     extra_metadata={"segment_id": 2}
        ... )
    """
    if not messages:
        return []

    boundaries = compute_sliding_window_boundaries(len(messages), chunk_length, chunk_overlap)

    chunks: List[ChunkObject] = []
    for idx, (start, end) in enumerate(boundaries):
        chunk = messages_to_chunk(
            messages,
            start_idx=start,
            end_idx=end,
            chunk_id=f"{chunk_id_prefix}_{idx}",
            extra_metadata=extra_metadata,
        )

        # Adjust message_ids for global offset if chunking a segment
        if global_offset > 0:
            chunk.message_ids = [mid + global_offset for mid in chunk.message_ids]
            chunk.metadata["start_idx"] = start + global_offset
            chunk.metadata["end_idx"] = end + global_offset

        chunks.append(chunk)

    return chunks
