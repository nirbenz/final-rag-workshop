"""Tests for MessageCountChunker implementation."""


from workshop.rag.chunkers import MessageCountChunker, MessageCountParams


def test_chunk_messages_default_params(test_messages):
    """Test chunking with default parameters."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    assert len(chunks) > 0
    assert all(hasattr(c, "id") for c in chunks)
    assert all(hasattr(c, "text") for c in chunks)
    assert all(hasattr(c, "message_ids") for c in chunks)
    assert all(hasattr(c, "metadata") for c in chunks)


def test_chunk_metadata(test_messages):
    """Test chunk metadata is complete."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    first_chunk = chunks[0]
    assert "start_idx" in first_chunk.metadata
    assert "end_idx" in first_chunk.metadata
    assert "start_time" in first_chunk.metadata
    assert "end_time" in first_chunk.metadata
    assert "speakers" in first_chunk.metadata


def test_custom_chunking_params(small_test_messages):
    """Test custom chunk_length and overlap."""
    chunker = MessageCountChunker(
        params=MessageCountParams(chunk_length=3, chunk_overlap=1)
    )
    chunks = chunker.chunk_messages(small_test_messages)

    # With length=3, overlap=1, stride=2: [0:3], [2:5], [4:7], [6:9], [8:10]
    assert len(chunks) == 5


def test_chunk_overlap(small_test_messages):
    """Test that overlap works correctly."""
    chunker = MessageCountChunker(
        params=MessageCountParams(chunk_length=3, chunk_overlap=1)
    )
    chunks = chunker.chunk_messages(small_test_messages)

    second_chunk = chunks[1]
    assert 2 in second_chunk.message_ids  # Should include message 2 (overlap)


def test_get_chunk_boundaries():
    """Test get_chunk_boundaries for preview."""
    chunker = MessageCountChunker(
        params=MessageCountParams(chunk_length=5, chunk_overlap=2)
    )

    boundaries = chunker.get_chunk_boundaries(12)

    expected = [(0, 5), (3, 8), (6, 11), (9, 12)]
    assert boundaries == expected


def test_get_chunk_boundaries_empty():
    """Test get_chunk_boundaries with zero messages."""
    chunker = MessageCountChunker()
    boundaries = chunker.get_chunk_boundaries(0)

    assert boundaries == []
