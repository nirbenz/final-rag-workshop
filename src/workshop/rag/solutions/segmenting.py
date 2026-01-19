# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Segmenting Functions - Reference Solutions

These are the working implementations for the segmenting exercise.
Compare your implementation with these solutions after completing the exercise.
"""

from datetime import timedelta
from typing import List, Sequence

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers.utils import chunk_messages_sliding_window
from workshop.rag.engines.types import ChunkObject


def segment_by_time_gaps(
    messages: Sequence[WhatsappMessage],
    time_gap_hours: float,
) -> List[List[WhatsappMessage]]:
    """
    Split messages into segments based on time gaps.

    Args:
        messages: Sequence of WhatsappMessage objects
        time_gap_hours: Minimum gap (in hours) to start a new segment

    Returns:
        List of message lists, each representing a continuous segment
    """
    if not messages:
        return []

    threshold = timedelta(hours=time_gap_hours)
    segments: List[List[WhatsappMessage]] = []
    current_segment: List[WhatsappMessage] = [messages[0]]

    for i in range(1, len(messages)):
        prev_time = messages[i - 1].timestamp
        curr_time = messages[i].timestamp

        # Handle pandas Timestamp objects
        if hasattr(prev_time, "to_pydatetime"):
            prev_time = prev_time.to_pydatetime()  # pyright: ignore[reportAttributeAccessIssue]
        if hasattr(curr_time, "to_pydatetime"):
            curr_time = curr_time.to_pydatetime()  # pyright: ignore[reportAttributeAccessIssue]

        time_delta = curr_time - prev_time

        if time_delta > threshold:
            # Gap exceeds threshold - start new segment
            segments.append(current_segment)
            current_segment = [messages[i]]
        else:
            # Continue current segment
            current_segment.append(messages[i])

    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)

    return segments


def chunk_segments(
    messages: Sequence[WhatsappMessage],
    segments: List[List[WhatsappMessage]],
    chunk_length: int,
    chunk_overlap: int,
) -> List[ChunkObject]:
    """
    Apply sliding-window chunking within each segment.

    Args:
        messages: Original full message sequence (for computing global indices)
        segments: List of segments from segment_by_time_gaps()
        chunk_length: Number of messages per chunk
        chunk_overlap: Number of overlapping messages between chunks

    Returns:
        List of ChunkObjects with segment_id in metadata
    """
    all_chunks: List[ChunkObject] = []
    global_offset = 0

    for segment_id, segment in enumerate(segments):
        # Chunk this segment using shared utility
        segment_chunks = chunk_messages_sliding_window(
            messages=segment,
            chunk_length=chunk_length,
            chunk_overlap=chunk_overlap,
            chunk_id_prefix=f"seg{segment_id}_chunk",
            global_offset=global_offset,
            extra_metadata={"segment_id": segment_id},
        )

        all_chunks.extend(segment_chunks)
        global_offset += len(segment)

    return all_chunks
