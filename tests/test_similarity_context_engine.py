"""Tests for SimilarityContextEngine."""

from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel
import pytest

from workshop.rag.chunkers import MessageCountChunker
from workshop.rag.engines import SimilarityContextEngine


@pytest.fixture
def mock_embedder() -> Embedder:
    """Mock embedder for tests - returns deterministic embeddings."""
    return Embedder(TestEmbeddingModel())


def test_similarity_engine_add_context(test_messages, mock_embedder):
    """Test adding chunks to similarity engine."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    engine = SimilarityContextEngine(embedder=mock_embedder, similarity_threshold=0.0)
    engine.add_context(chunks)

    assert len(engine.context) == len(chunks)


def test_similarity_engine_retrieval(test_messages, mock_embedder):
    """Test retrieval returns top-k by similarity."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    engine = SimilarityContextEngine(embedder=mock_embedder, similarity_threshold=0.0)
    engine.add_context(chunks)

    retrieved = engine.get_relevant_context("test query", top_k=3)

    assert len(retrieved) <= 3
    assert len(retrieved) > 0


def test_similarity_threshold(test_messages, mock_embedder):
    """Test similarity threshold filtering."""
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(test_messages)

    engine = SimilarityContextEngine(embedder=mock_embedder, similarity_threshold=0.9)
    engine.add_context(chunks)

    retrieved = engine.get_relevant_context("completely different unrelated text", top_k=10)

    assert len(retrieved) <= len(chunks)


def test_similarity_engine_requires_embedder():
    """Test SimilarityContextEngine requires embedder."""
    with pytest.raises(ValueError, match="requires 'embedder'"):
        SimilarityContextEngine()
