# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Segmenting chunker - Phase 4 extension.

This chunker first segments conversations by time gaps (or other criteria),
then applies message-count chunking within each segment. This prevents chunks
from spanning unrelated conversation sessions.

Workshop participants implement this as an advanced extension to learn about:
- Conversation segmentation strategies
- Hierarchical chunking (segment -> chunk)
- Metadata enrichment with segment_id
"""

from typing import List, Optional, Sequence, Tuple

from pydantic import Field

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers.types import BaseChunkerParams
from workshop.rag.chunkers.utils import compute_sliding_window_boundaries
from workshop.rag.engines.types import ChunkObject

# Toggle between exercise stubs and solutions:
# - Use exercises for workshop participants to implement
# - Use solutions for working reference implementation
USE_SOLUTIONS = True

if USE_SOLUTIONS:
    from workshop.rag.solutions.segmenting import chunk_segments, segment_by_time_gaps
else:
    from workshop.rag.exercises.segmenting import chunk_segments, segment_by_time_gaps


class SegmentingChunkerParams(BaseChunkerParams):
    """
    Hyperparameters for segmenting chunker.

    Inherits conversation windowing from BaseChunkerParams.
    Adds segmentation and chunking parameters.
    """

    time_gap_hours: float = Field(
        default=6.0,
        ge=0.5,
        le=48.0,
        json_schema_extra={"step": 0.5, "label": "Time Gap (hours)"},
    )
    chunk_length: int = Field(
        default=6,
        ge=1,
        le=50,
        json_schema_extra={"step": 1, "label": "Messages per Chunk"},
    )
    chunk_overlap: int = Field(
        default=4,
        ge=0,
        le=10,
        json_schema_extra={"step": 1, "label": "Overlap Messages"},
    )


class SegmentingChunker:
    """
    Chunker that first segments conversation by time gaps, then chunks within segments.

    Key implementation considerations:
    - Segment by finding gaps > threshold in message timestamps
    - Each segment is processed independently (message-count chunking)
    - segment_id added to metadata for each chunk
    - Overlap does NOT cross segment boundaries

    Implementation strategy:
    1. Scan timestamps, find gaps > time_gap_hours
    2. Split messages into segments at gap boundaries
    3. For each segment: run message-count chunking (reuse MessageCountChunker logic)
    4. Add segment_id to each chunk's metadata

    Why this is useful:
    - Conversations often have natural breaks (sleep, work hours, topic shifts)
    - Prevents chunks from mixing unrelated conversation contexts
    - Enables filtering by segment (e.g., "show me yesterday's conversation")
    - Improves retrieval quality by respecting conversation structure

    Example with time_gap_hours=6, chunk_length=3, chunk_overlap=1:
    Messages:
      [0] 9:00 AM - "Good morning"
      [1] 9:05 AM - "How are you?"
      [2] 9:10 AM - "I'm good!"
      [3] 5:00 PM - "Back from work"  # 8-hour gap
      [4] 5:05 PM - "How was your day?"
      [5] 5:10 PM - "Great!"

    Segments:
      Segment 0: messages[0:3] (morning conversation)
      Segment 1: messages[3:6] (evening conversation)

    Chunks:
      Chunk 0: messages[0:3], segment_id=0
      Chunk 1: messages[2:5], segment_id=1 (starts at boundary)
      Chunk 2: messages[4:6], segment_id=1

    Participants need to:
    - Implement time-gap-based segmentation
    - Reuse message-count chunking logic within segments
    - Add segment_id to metadata
    - Ensure overlap respects segment boundaries
    - Track message_ids for traceability
    """

    def __init__(self, params: Optional[SegmentingChunkerParams] = None):
        """
        Initialize chunker with hyperparameters.

        Args:
            params: Hyperparameters (default: SegmentingChunkerParams())
        """
        self.params = params or SegmentingChunkerParams()

    def _segment_by_time_gaps(self, messages: Sequence[WhatsappMessage]) -> List[List[WhatsappMessage]]:
        """
        Split messages into segments based on time gaps.

        Args:
            messages: Sequence of WhatsappMessage objects

        Returns:
            List of message lists, each representing a continuous segment

        Uses the imported segment_by_time_gaps function (from exercises or solutions).
        """
        return segment_by_time_gaps(messages, self.params.time_gap_hours)

    def chunk_messages(self, messages: Sequence[WhatsappMessage]) -> Sequence[ChunkObject]:
        """
        Transform messages into chunks with conversation segmentation.

        Args:
            messages: Sequence of WhatsappMessage objects from conversation

        Returns:
            Sequence of ChunkObjects with:
            - text: Combined message text
            - message_ids: Indices in original message list
            - metadata: start_idx, end_idx, timestamps, speakers, segment_id

        Uses the imported functions (from exercises or solutions):
        1. segment_by_time_gaps() to split into conversation segments
        2. chunk_segments() to apply sliding-window chunking within each segment
        """
        if not messages:
            return []

        # Segment by time gaps
        segments = self._segment_by_time_gaps(messages)

        # Chunk within each segment
        return chunk_segments(
            messages=messages,
            segments=segments,
            chunk_length=self.params.chunk_length,
            chunk_overlap=self.params.chunk_overlap,
        )

    def get_chunk_boundaries(self, num_messages: int) -> List[Tuple[int, int]]:
        """
        Get chunk boundaries for preview visualization.

        Note: This is an approximation that doesn't account for time-gap segmentation,
        since we don't have access to actual timestamps here. It uses simple
        sliding-window boundaries for UI preview purposes.

        Args:
            num_messages: Total number of messages in conversation

        Returns:
            List of (start_idx, end_idx) tuples for each chunk
        """
        return compute_sliding_window_boundaries(
            num_messages,
            self.params.chunk_length,
            self.params.chunk_overlap,
        )
