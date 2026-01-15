"""Tests for NaiveContextEngine implementation."""


from workshop.rag.chunkers import MessageCountChunker
from workshop.rag.engines import NaiveContextEngine


def test_engine_add_context(small_test_messages):
    """Test adding chunks to engine."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(small_test_messages)

    engine = NaiveContextEngine()
    engine.add_context(chunks)

    stored = engine.context
    assert len(stored) == len(chunks)


def test_engine_get_relevant_context(small_test_messages):
    """Test retrieval returns all chunks (naive behavior)."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(small_test_messages)

    engine = NaiveContextEngine()
    engine.add_context(chunks)

    retrieved = engine.get_relevant_context("test query", top_k=5)

    # NaiveContextEngine returns all chunks regardless of query
    assert len(retrieved) == len(chunks)
