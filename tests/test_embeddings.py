"""Tests for embeddings module."""

import pytest

from workshop.embeddings import (
    APIInitializationError,
    _prepare_texts_for_embedding,
    get_embedding_model,
)


class TestGetEmbeddingModel:
    """Tests for embedding model creation."""

    def test_valid_model_name_format(self):
        """Valid model name with provider:model format."""
        config = {"model_name": "openai:text-embedding-3-small", "kwargs": {}}
        embedder = get_embedding_model(config)
        assert embedder is not None

    def test_invalid_model_name_no_colon(self):
        """Model name without colon raises error."""
        config = {"model_name": "text-embedding-3-small", "kwargs": {}}
        with pytest.raises(APIInitializationError) as exc_info:
            get_embedding_model(config)
        assert "must be in format <provider>:<model_name>" in str(exc_info.value)

    def test_model_with_kwargs(self):
        """Model creation passes kwargs correctly."""
        config = {
            "model_name": "openai:text-embedding-3-small",
            "kwargs": {"dimensions": 512},
        }
        embedder = get_embedding_model(config)
        assert embedder is not None

    def test_model_with_empty_kwargs(self):
        """Model creation works with empty kwargs."""
        config = {"model_name": "openai:text-embedding-3-small"}
        embedder = get_embedding_model(config)
        assert embedder is not None


class TestPrepareTextsForEmbedding:
    """Tests for text preparation/batching logic."""

    def test_single_text_under_limit(self):
        """Single text under limit creates one batch."""
        texts = ["hello world"]
        token_counts = [2]
        max_tokens = 100

        batches = _prepare_texts_for_embedding(texts, token_counts, max_tokens)

        assert len(batches) == 1
        assert batches[0] == ["hello world"]

    def test_multiple_texts_single_batch(self):
        """Multiple texts fitting in one batch."""
        texts = ["hello", "world", "test"]
        token_counts = [1, 1, 1]
        max_tokens = 10

        batches = _prepare_texts_for_embedding(texts, token_counts, max_tokens)

        assert len(batches) == 1
        assert batches[0] == ["hello", "world", "test"]

    def test_multiple_texts_multiple_batches(self):
        """Texts split across multiple batches."""
        texts = ["text1", "text2", "text3", "text4"]
        token_counts = [5, 5, 5, 5]
        max_tokens = 10

        batches = _prepare_texts_for_embedding(texts, token_counts, max_tokens)

        # Each pair should be in separate batch (5+5=10, then next pair)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2

    def test_long_text_split(self):
        """Text exceeding max_tokens gets split."""
        long_text = "word " * 1000  # ~1000 words
        texts = [long_text]
        token_counts = [1000]
        max_tokens = 100

        batches = _prepare_texts_for_embedding(texts, token_counts, max_tokens)

        # Should be split into multiple chunks
        total_chunks = sum(len(batch) for batch in batches)
        assert total_chunks > 1

    def test_empty_texts(self):
        """Empty text list returns empty batches."""
        batches = _prepare_texts_for_embedding([], [], 100)
        assert batches == []

    def test_max_tokens_required(self):
        """max_tokens=None raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _prepare_texts_for_embedding(["text"], [1], None)
        assert "max_tokens is required" in str(exc_info.value)

    def test_batch_boundary_exact(self):
        """Texts exactly at batch boundary handled correctly."""
        texts = ["a", "b", "c"]
        token_counts = [5, 5, 5]
        max_tokens = 10

        batches = _prepare_texts_for_embedding(texts, token_counts, max_tokens)

        # First two fit exactly, third starts new batch
        assert len(batches) == 2
        assert batches[0] == ["a", "b"]
        assert batches[1] == ["c"]

    def test_single_large_text_per_batch(self):
        """Single text that fills a batch entirely."""
        texts = ["large", "small"]
        token_counts = [10, 2]
        max_tokens = 10

        batches = _prepare_texts_for_embedding(texts, token_counts, max_tokens)

        # Large text alone in first batch, small in second
        assert len(batches) == 2
        assert batches[0] == ["large"]
        assert batches[1] == ["small"]


class TestAPIInitializationError:
    """Tests for the custom exception."""

    def test_exception_message(self):
        """Exception carries message correctly."""
        error = APIInitializationError("test message")
        assert str(error) == "test message"

    def test_exception_inheritance(self):
        """Exception is a proper Exception subclass."""
        error = APIInitializationError("test")
        assert isinstance(error, Exception)
