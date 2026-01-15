"""Tests for RAGContextEngine."""

import shutil
import tempfile

from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel
import pytest

from workshop.rag.chunkers import MessageCountChunker
from workshop.rag.engines import RAGContextEngine


@pytest.fixture
def mock_embedder() -> Embedder:
    """Mock embedder for tests - returns deterministic embeddings."""
    return Embedder(TestEmbeddingModel())


def test_rag_engine_add_context(test_messages, mock_embedder):
    """Test adding chunks to RAG engine with Qdrant."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    temp_dir = tempfile.mkdtemp()
    try:
        engine = RAGContextEngine(embedder=mock_embedder, db_path=temp_dir)

        engine.add_context(chunks)

        assert len(engine.context) == len(chunks)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rag_engine_retrieval(test_messages, mock_embedder):
    """Test RAG engine ANN search retrieval."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    temp_dir = tempfile.mkdtemp()
    try:
        engine = RAGContextEngine(embedder=mock_embedder, db_path=temp_dir)

        engine.add_context(chunks)

        retrieved = engine.get_relevant_context("test query", top_k=3)

        assert len(retrieved) <= 3
        assert len(retrieved) > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rag_engine_empty_query(mock_embedder):
    """Test querying empty RAG engine."""
    temp_dir = tempfile.mkdtemp()
    try:
        engine = RAGContextEngine(embedder=mock_embedder, db_path=temp_dir)

        retrieved = engine.get_relevant_context("test query", top_k=5)

        assert len(retrieved) == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rag_engine_metadata_preserved(test_messages, mock_embedder):
    """Test that RAG engine preserves chunk metadata correctly."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    temp_dir = tempfile.mkdtemp()
    try:
        engine = RAGContextEngine(embedder=mock_embedder, db_path=temp_dir)

        engine.add_context(chunks)

        stored = engine.context
        assert len(stored) == len(chunks)

        # Create lookup by chunk id (order not guaranteed by vector DB)
        original_by_id = {c.id: c for c in chunks}

        for stored_chunk in stored:
            original_chunk = original_by_id[stored_chunk.id]
            assert stored_chunk.text == original_chunk.text
            assert stored_chunk.message_ids == original_chunk.message_ids
            assert "start_time" in stored_chunk.metadata
            assert "end_time" in stored_chunk.metadata
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_rag_engine_requires_embedder():
    """Test RAGContextEngine requires embedder."""
    temp_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(ValueError, match="requires 'embedder'"):
            RAGContextEngine(db_path=temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
