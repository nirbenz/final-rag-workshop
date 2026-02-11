"""
Integration tests for the full RAG pipeline using the example chat file.

Tests the complete flow: file -> parse -> chunk -> engine -> query
"""

from pathlib import Path

import pytest

from workshop.chat import ChatContext, WhatsappMessage, load_whatsapp_chat
from workshop.rag.chunkers import MessageCountChunker, MessageCountParams
from workshop.rag.engines import NaiveContextEngine

EXAMPLE_CHAT_PATH = Path(__file__).parent.parent / "chats" / "example_chat.txt"


class TestChatLoading:
    """Tests for loading and parsing WhatsApp chat files."""

    def test_load_whatsapp_chat(self):
        """Test loading the example chat file."""
        messages = load_whatsapp_chat(EXAMPLE_CHAT_PATH)

        assert len(messages) > 0, "Should load messages from example chat"
        assert all(isinstance(m, WhatsappMessage) for m in messages)

    def test_message_structure(self):
        """Test that loaded messages have correct structure."""
        messages = load_whatsapp_chat(EXAMPLE_CHAT_PATH)

        first_msg = messages[0]
        assert first_msg.timestamp is not None
        assert first_msg.text is not None

    def test_chat_context_initialization(self):
        """Test ChatContext loads and filters correctly."""
        ctx = ChatContext(str(EXAMPLE_CHAT_PATH))

        assert ctx.num_messages > 0
        assert ctx.start_time is not None
        assert ctx.end_time is not None
        assert ctx.start_time <= ctx.end_time

    def test_chat_context_text_output(self):
        """Test ChatContext produces text context."""
        ctx = ChatContext(str(EXAMPLE_CHAT_PATH))

        text = ctx.text_context
        assert len(text) > 0
        assert "NovaMind" in text or "Dana" in text or "Marco" in text


class TestFullPipeline:
    """Integration tests for the complete RAG pipeline."""

    @pytest.fixture
    def loaded_messages(self):
        """Load messages from example chat."""
        return load_whatsapp_chat(EXAMPLE_CHAT_PATH)

    @pytest.fixture
    def chunker(self):
        """Create a chunker with default params."""
        return MessageCountChunker(params=MessageCountParams(chunk_length=5, chunk_overlap=2))

    @pytest.fixture
    def engine(self):
        """Create a naive context engine."""
        return NaiveContextEngine()

    def test_pipeline_messages_to_chunks(self, loaded_messages, chunker):
        """Test chunking loaded messages."""
        chunks = chunker.chunk_messages(loaded_messages)

        assert len(chunks) > 0, "Should create chunks from messages"
        assert all(hasattr(c, "text") for c in chunks)
        assert all(hasattr(c, "metadata") for c in chunks)
        assert all(hasattr(c, "message_ids") for c in chunks)

    def test_pipeline_chunks_to_engine(self, loaded_messages, chunker, engine):
        """Test storing chunks in engine."""
        chunks = chunker.chunk_messages(loaded_messages)
        engine.add_context(chunks)

        assert len(engine.context) == len(chunks)

    def test_pipeline_query_retrieval(self, loaded_messages, chunker, engine):
        """Test querying the engine returns relevant chunks."""
        chunks = chunker.chunk_messages(loaded_messages)
        engine.add_context(chunks)

        results = engine.get_relevant_context("chunking strategies", top_k=3)

        assert len(results) > 0, "Should return results for query"

    def test_pipeline_chunk_traceability(self, loaded_messages, chunker, engine):
        """Test that chunks can be traced back to original messages."""
        chunks = chunker.chunk_messages(loaded_messages)
        engine.add_context(chunks)

        results = engine.get_relevant_context("embeddings", top_k=1)
        assert len(results) > 0

        first_result = results[0]
        reconstructed = first_result.get_messages(loaded_messages)

        assert len(reconstructed) > 0, "Should reconstruct messages from chunk"
        assert all(isinstance(m, WhatsappMessage) for m in reconstructed)

    def test_pipeline_metadata_preserved(self, loaded_messages, chunker, engine):
        """Test that chunk metadata is preserved through pipeline."""
        chunks = chunker.chunk_messages(loaded_messages)
        engine.add_context(chunks)

        results = engine.get_relevant_context("vector database", top_k=1)
        assert len(results) > 0

        chunk = results[0]
        assert "start_idx" in chunk.metadata
        assert "end_idx" in chunk.metadata
        assert "start_time" in chunk.metadata
        assert "end_time" in chunk.metadata
        assert "speakers" in chunk.metadata

    def test_full_flow_with_chat_context(self, chunker, engine):
        """Test complete flow using ChatContext."""
        ctx = ChatContext(str(EXAMPLE_CHAT_PATH))
        messages = ctx.get_context()

        chunks = chunker.chunk_messages(messages)
        engine.add_context(chunks)

        results = engine.get_relevant_context("RAG workshop", top_k=5)

        assert len(results) > 0
        assert ctx.num_messages == len(messages)
        assert len(engine.context) == len(chunks)


class TestChunkContent:
    """Tests verifying chunk content quality."""

    @pytest.fixture
    def chunks(self):
        """Create chunks from example chat."""
        messages = load_whatsapp_chat(EXAMPLE_CHAT_PATH)
        chunker = MessageCountChunker(params=MessageCountParams(chunk_length=5, chunk_overlap=2))
        return chunker.chunk_messages(messages)

    def test_chunks_contain_text(self, chunks):
        """Test that all chunks have non-empty text."""
        for chunk in chunks:
            assert chunk.text, f"Chunk {chunk.id} has empty text"

    def test_chunks_have_valid_boundaries(self, chunks):
        """Test that chunk boundaries are valid."""
        for chunk in chunks:
            start = chunk.metadata["start_idx"]
            end = chunk.metadata["end_idx"]
            assert start < end, f"Invalid boundaries: {start} >= {end}"
            assert start >= 0, f"Negative start index: {start}"

    def test_chunks_have_speakers(self, chunks):
        """Test that chunks track speakers."""
        for chunk in chunks:
            speakers = chunk.metadata.get("speakers", [])
            assert len(speakers) > 0, f"Chunk {chunk.id} has no speakers"

    def test_chunk_overlap_works(self):
        """Test that overlapping chunks share messages."""
        messages = load_whatsapp_chat(EXAMPLE_CHAT_PATH)
        chunker = MessageCountChunker(params=MessageCountParams(chunk_length=5, chunk_overlap=2))
        chunks = chunker.chunk_messages(messages)

        if len(chunks) >= 2:
            first_ids = set(chunks[0].message_ids)
            second_ids = set(chunks[1].message_ids)
            overlap = first_ids & second_ids

            assert len(overlap) > 0, "Consecutive chunks should overlap"
