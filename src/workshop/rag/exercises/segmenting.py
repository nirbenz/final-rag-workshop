# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

# pyright: ignore
# ruff: ignore

"""
Segmenting Functions Exercise - Phase 4 Extension

This exercise teaches conversation segmentation and hierarchical chunking:
1. Time-gap based segmentation
2. Applying chunking within segments

After completing this exercise, you'll understand:
- How to detect conversation breaks using timestamps
- Why segmentation improves retrieval quality
- How to combine segmentation with chunking strategies

Implementation tips:
- Use datetime.timedelta for time comparisons
- Track segment boundaries carefully
- Use the shared utils for sliding-window chunking within segments
"""

from typing import List, Sequence

from workshop.chat import WhatsappMessage
from workshop.rag.engines.types import ChunkObject


def segment_by_time_gaps(
    messages: Sequence[WhatsappMessage],
    time_gap_hours: float,
) -> List[List[WhatsappMessage]]:
    """
    Split messages into segments based on time gaps.

    A new segment starts when the gap between consecutive messages
    exceeds time_gap_hours. This helps identify natural conversation
    breaks (e.g., overnight, during work hours).

    Args:
        messages: Sequence of WhatsappMessage objects (chronologically ordered)
        time_gap_hours: Minimum gap (in hours) to start a new segment

    Returns:
        List of message lists, each representing a continuous segment.
        Empty list if no messages provided.

    Examples:
        >>> from datetime import datetime
        >>> # Messages with an 8-hour gap between [2] and [3]
        >>> messages = [
        ...     WhatsappMessage(timestamp=datetime(2024, 1, 1, 9, 0), ...),   # 9 AM
        ...     WhatsappMessage(timestamp=datetime(2024, 1, 1, 9, 5), ...),   # 9:05 AM
        ...     WhatsappMessage(timestamp=datetime(2024, 1, 1, 9, 10), ...),  # 9:10 AM
        ...     WhatsappMessage(timestamp=datetime(2024, 1, 1, 17, 0), ...),  # 5 PM (8h gap)
        ...     WhatsappMessage(timestamp=datetime(2024, 1, 1, 17, 5), ...),  # 5:05 PM
        ... ]
        >>> segments = segment_by_time_gaps(messages, time_gap_hours=6.0)
        >>> len(segments)
        2
        >>> len(segments[0])  # Morning segment
        3
        >>> len(segments[1])  # Evening segment
        2

    Implementation hints:
    1. Handle empty messages case (return [])
    2. Initialize first segment with first message
    3. For each subsequent message:
       - Compute time delta from previous message
       - If delta > timedelta(hours=time_gap_hours): start new segment
       - Otherwise: append to current segment
    4. Return list of all segments

    TODO: Implement this function
    TODO: REMOVE THE EXCEPTION
    """
    raise NotImplementedError(
        "Implement segment_by_time_gaps.\n"
        "Hint: Use datetime.timedelta(hours=time_gap_hours) for threshold comparison."
    )


def chunk_segments(
    messages: Sequence[WhatsappMessage],
    segments: List[List[WhatsappMessage]],
    chunk_length: int,
    chunk_overlap: int,
) -> List[ChunkObject]:
    """
    Apply sliding-window chunking within each segment.

    This function chunks each segment independently, ensuring that:
    - Overlap does NOT cross segment boundaries
    - Each chunk has segment_id in its metadata
    - message_ids track global positions (not segment-local)

    Args:
        messages: Original full message sequence (for computing global indices)
        segments: List of segments from segment_by_time_gaps()
        chunk_length: Number of messages per chunk
        chunk_overlap: Number of overlapping messages between chunks

    Returns:
        List of ChunkObjects with segment_id in metadata

    Examples:
        >>> # With 2 segments of 5 messages each, chunk_length=3, overlap=1
        >>> # Segment 0 (msgs 0-4): chunks at [0:3], [2:5]
        >>> # Segment 1 (msgs 5-9): chunks at [5:8], [7:10]
        >>> # Each chunk has segment_id=0 or segment_id=1

    Implementation hints:
    1. Track global_offset as you iterate through segments
    2. For each segment:
       - Use chunk_messages_sliding_window() from chunkers/utils.py
       - Pass segment_id in extra_metadata
       - Pass global_offset for correct message_ids
    3. Collect all chunks from all segments

    TODO: Implement this function
    TODO: REMOVE THE EXCEPTION
    """
    raise NotImplementedError(
        "Implement chunk_segments.\n" "Hint: Use chunk_messages_sliding_window() from workshop.rag.chunkers.utils"
    )
