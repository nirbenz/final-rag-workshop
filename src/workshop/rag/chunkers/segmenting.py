# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Segmenting chunker - Phase 4 extension stub.

This chunker first segments conversations by time gaps (or other criteria),
then applies message-count chunking within each segment. This prevents chunks
from spanning unrelated conversation sessions.

Workshop participants implement this as an advanced extension to learn about:
- Conversation segmentation strategies
- Hierarchical chunking (segment -> chunk)
- Metadata enrichment with segment_id
"""

from typing import List, Optional, Sequence

from pydantic import Field

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers.types import BaseChunkerParams
from workshop.rag.engines.types import ChunkObject


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

        Implementation:
        1. Initialize first segment with first message
        2. For each subsequent message:
           - Compute time gap from previous message
           - If gap > time_gap_hours: start new segment
           - Otherwise: add to current segment
        3. Return list of segments

        TODO for participants: Implement segmentation logic
        """
        raise NotImplementedError(
            "Participants implement this. "
            "Hint: Use message.timestamp to compute time deltas, "
            "and datetime.timedelta for threshold comparison."
        )

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

        Implementation steps:
        1. Segment messages by time gaps
        2. For each segment:
           - Apply message-count chunking (sliding window)
           - Add segment_id to metadata
        3. Ensure overlap does NOT cross segment boundaries
        4. Track global message_ids (not segment-local indices)

        TODO for participants: Implement full pipeline
        """
        raise NotImplementedError(
            "Extension for advanced participants. "
            "Hint: Reuse MessageCountChunker logic but apply per-segment, "
            "and track global message indices across segments."
        )
