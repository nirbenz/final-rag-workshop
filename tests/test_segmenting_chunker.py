"""
Tests for SegmentingChunker - time-gap-based conversation segmentation.
"""

from datetime import datetime, timedelta

import pytest

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers import SegmentingChunker, SegmentingChunkerParams


@pytest.fixture
def messages_with_gap():
    """
    Create messages with a time gap in the middle.

    Morning session: 9:00-9:10 (3 messages)
    Evening session: 17:00-17:10 (3 messages) - 8 hour gap
    """
    base_date = datetime(2024, 1, 15)
    messages = [
        WhatsappMessage(
            id=0,
            timestamp=base_date.replace(hour=9, minute=0),
            user="Alice",
            text="Good morning!",
        ),
        WhatsappMessage(
            id=1,
            timestamp=base_date.replace(hour=9, minute=5),
            user="Bob",
            text="Morning! How are you?",
        ),
        WhatsappMessage(
            id=2,
            timestamp=base_date.replace(hour=9, minute=10),
            user="Alice",
            text="Doing great, thanks!",
        ),
        WhatsappMessage(
            id=3,
            timestamp=base_date.replace(hour=17, minute=0),
            user="Bob",
            text="Back from work!",
        ),
        WhatsappMessage(
            id=4,
            timestamp=base_date.replace(hour=17, minute=5),
            user="Alice",
            text="How was your day?",
        ),
        WhatsappMessage(
            id=5,
            timestamp=base_date.replace(hour=17, minute=10),
            user="Bob",
            text="Pretty good!",
        ),
    ]
    return messages


@pytest.fixture
def continuous_messages():
    """Create messages without significant time gaps."""
    base_time = datetime(2024, 1, 15, 10, 0)
    messages = []
    for i in range(10):
        messages.append(
            WhatsappMessage(
                id=i,
                timestamp=base_time + timedelta(minutes=i * 5),
                user=f"User{i % 2}",
                text=f"Message {i}",
            )
        )
    return messages


class TestSegmentingChunkerBasic:
    """Basic functionality tests."""

    def test_chunker_initialization(self):
        """Test chunker initializes with default params."""
        chunker = SegmentingChunker()

        assert chunker.params.time_gap_hours == 6.0
        assert chunker.params.chunk_length == 6
        assert chunker.params.chunk_overlap == 4

    def test_chunker_custom_params(self):
        """Test chunker accepts custom params."""
        params = SegmentingChunkerParams(
            time_gap_hours=2.0,
            chunk_length=3,
            chunk_overlap=1,
        )
        chunker = SegmentingChunker(params=params)

        assert chunker.params.time_gap_hours == 2.0
        assert chunker.params.chunk_length == 3
        assert chunker.params.chunk_overlap == 1

    def test_empty_messages(self):
        """Test chunker handles empty input."""
        chunker = SegmentingChunker()
        chunks = chunker.chunk_messages([])

        assert chunks == []


class TestTimeGapSegmentation:
    """Tests for time-gap-based segmentation."""

    def test_detects_time_gap(self, messages_with_gap):
        """Test that time gaps create separate segments."""
        params = SegmentingChunkerParams(
            time_gap_hours=6.0,
            chunk_length=10,
            chunk_overlap=0,
        )
        chunker = SegmentingChunker(params=params)

        chunks = chunker.chunk_messages(messages_with_gap)

        assert len(chunks) == 2, "Should create 2 chunks (one per segment)"

        first_chunk = chunks[0]
        second_chunk = chunks[1]

        assert first_chunk.metadata["segment_id"] == 0
        assert second_chunk.metadata["segment_id"] == 1

    def test_no_gap_single_segment(self, continuous_messages):
        """Test continuous messages stay in one segment."""
        params = SegmentingChunkerParams(
            time_gap_hours=1.0,
            chunk_length=20,
            chunk_overlap=0,
        )
        chunker = SegmentingChunker(params=params)

        chunks = chunker.chunk_messages(continuous_messages)

        assert len(chunks) == 1, "Continuous messages should be in one chunk"
        assert chunks[0].metadata["segment_id"] == 0

    def test_smaller_gap_threshold(self, messages_with_gap):
        """Test smaller gap threshold creates more segments."""
        params = SegmentingChunkerParams(
            time_gap_hours=0.5,
            chunk_length=10,
            chunk_overlap=0,
        )
        chunker = SegmentingChunker(params=params)

        chunks = chunker.chunk_messages(messages_with_gap)

        assert len(chunks) >= 2, "Small gap threshold should detect gap"


class TestChunkingWithinSegments:
    """Tests for chunking behavior within segments."""

    def test_chunking_respects_segments(self, messages_with_gap):
        """Test that chunking respects segment boundaries."""
        params = SegmentingChunkerParams(
            time_gap_hours=6.0,
            chunk_length=2,
            chunk_overlap=1,
        )
        chunker = SegmentingChunker(params=params)

        chunks = chunker.chunk_messages(messages_with_gap)

        segment_0_chunks = [c for c in chunks if c.metadata["segment_id"] == 0]
        segment_1_chunks = [c for c in chunks if c.metadata["segment_id"] == 1]

        assert len(segment_0_chunks) > 0
        assert len(segment_1_chunks) > 0

        for chunk in segment_0_chunks:
            assert all(mid < 3 for mid in chunk.message_ids), "Segment 0 chunks should only contain morning messages"

        for chunk in segment_1_chunks:
            assert all(mid >= 3 for mid in chunk.message_ids), "Segment 1 chunks should only contain evening messages"

    def test_overlap_within_segment(self, continuous_messages):
        """Test overlap works within a segment."""
        params = SegmentingChunkerParams(
            time_gap_hours=24.0,
            chunk_length=4,
            chunk_overlap=2,
        )
        chunker = SegmentingChunker(params=params)

        chunks = chunker.chunk_messages(continuous_messages)

        if len(chunks) >= 2:
            first_ids = set(chunks[0].message_ids)
            second_ids = set(chunks[1].message_ids)
            overlap = first_ids & second_ids

            assert len(overlap) > 0, "Consecutive chunks should overlap"


class TestChunkMetadata:
    """Tests for chunk metadata."""

    def test_metadata_completeness(self, messages_with_gap):
        """Test chunks have all required metadata."""
        chunker = SegmentingChunker()
        chunks = chunker.chunk_messages(messages_with_gap)

        for chunk in chunks:
            assert "start_idx" in chunk.metadata
            assert "end_idx" in chunk.metadata
            assert "start_time" in chunk.metadata
            assert "end_time" in chunk.metadata
            assert "speakers" in chunk.metadata
            assert "segment_id" in chunk.metadata

    def test_message_ids_present(self, messages_with_gap):
        """Test chunks have message_ids for traceability."""
        chunker = SegmentingChunker()
        chunks = chunker.chunk_messages(messages_with_gap)

        for chunk in chunks:
            assert hasattr(chunk, "message_ids")
            assert len(chunk.message_ids) > 0

    def test_speakers_tracked(self, messages_with_gap):
        """Test speakers are tracked in metadata."""
        chunker = SegmentingChunker()
        chunks = chunker.chunk_messages(messages_with_gap)

        for chunk in chunks:
            speakers = chunk.metadata["speakers"]
            assert isinstance(speakers, list)
            assert all(s in ["Alice", "Bob"] for s in speakers)


class TestGetChunkBoundaries:
    """Tests for get_chunk_boundaries preview method."""

    def test_boundaries_returned(self):
        """Test get_chunk_boundaries returns valid boundaries."""
        params = SegmentingChunkerParams(chunk_length=3, chunk_overlap=1)
        chunker = SegmentingChunker(params=params)

        boundaries = chunker.get_chunk_boundaries(10)

        assert len(boundaries) > 0
        assert all(isinstance(b, tuple) and len(b) == 2 for b in boundaries)

    def test_boundaries_cover_all_messages(self):
        """Test boundaries cover all message indices."""
        params = SegmentingChunkerParams(chunk_length=3, chunk_overlap=1)
        chunker = SegmentingChunker(params=params)

        boundaries = chunker.get_chunk_boundaries(10)

        covered_indices = set()
        for start, end in boundaries:
            covered_indices.update(range(start, end))

        assert covered_indices == set(range(10))
