"""Tests for embedding cache module."""

from unittest.mock import patch

import pytest

from workshop import embedding_cache
from workshop.embedding_cache import (
    _cache_key,
    cache_embedding,
    clear_cache,
    get_cache_stats,
    get_cached_embedding,
    save_cache,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset global cache state before and after each test."""
    embedding_cache._cache = None
    yield
    embedding_cache._cache = None


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Redirect cache to temp directory."""
    cache_dir = tmp_path / ".embedding_cache"
    cache_file = cache_dir / "embeddings.json"

    with patch.object(embedding_cache, "_CACHE_DIR", cache_dir):
        with patch.object(embedding_cache, "_CACHE_FILE", cache_file):
            yield cache_dir, cache_file


class TestCacheKey:
    """Tests for cache key generation."""

    def test_cache_key_deterministic(self):
        """Same inputs produce same key."""
        key1 = _cache_key("model:name", "test text")
        key2 = _cache_key("model:name", "test text")
        assert key1 == key2

    def test_cache_key_different_models(self):
        """Different models produce different keys."""
        key1 = _cache_key("openai:text-embedding-3-small", "test")
        key2 = _cache_key("openai:text-embedding-3-large", "test")
        assert key1 != key2

    def test_cache_key_different_texts(self):
        """Different texts produce different keys."""
        key1 = _cache_key("model", "hello")
        key2 = _cache_key("model", "world")
        assert key1 != key2

    def test_cache_key_length(self):
        """Cache key has expected length (32 hex chars = 128 bits)."""
        key = _cache_key("model", "text")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)


class TestCacheOperations:
    """Tests for basic cache operations."""

    def test_cache_miss_returns_none(self, temp_cache_dir):
        """Getting uncached embedding returns None."""
        result = get_cached_embedding("model", "uncached text")
        assert result is None

    def test_cache_hit_returns_embedding(self, temp_cache_dir):
        """Getting cached embedding returns the embedding."""
        embedding = [0.1, 0.2, 0.3]
        cache_embedding("model", "text", embedding)

        result = get_cached_embedding("model", "text")
        assert result == embedding

    def test_cache_different_models_isolated(self, temp_cache_dir):
        """Embeddings for different models are isolated."""
        embedding1 = [0.1, 0.2]
        embedding2 = [0.3, 0.4]

        cache_embedding("model1", "text", embedding1)
        cache_embedding("model2", "text", embedding2)

        assert get_cached_embedding("model1", "text") == embedding1
        assert get_cached_embedding("model2", "text") == embedding2

    def test_cache_overwrite(self, temp_cache_dir):
        """Caching same key overwrites previous value."""
        cache_embedding("model", "text", [0.1])
        cache_embedding("model", "text", [0.2])

        result = get_cached_embedding("model", "text")
        assert result == [0.2]


class TestCachePersistence:
    """Tests for cache persistence to disk."""

    def test_save_and_load(self, temp_cache_dir):
        """Cache persists across resets."""
        cache_dir, cache_file = temp_cache_dir

        # Save some embeddings
        cache_embedding("model", "text1", [0.1, 0.2])
        cache_embedding("model", "text2", [0.3, 0.4])
        save_cache()

        # Verify file exists
        assert cache_file.exists()

        # Reset in-memory cache
        embedding_cache._cache = None

        # Load should restore
        result1 = get_cached_embedding("model", "text1")
        result2 = get_cached_embedding("model", "text2")

        assert result1 == [0.1, 0.2]
        assert result2 == [0.3, 0.4]

    def test_save_creates_directory(self, temp_cache_dir):
        """Save creates cache directory if missing."""
        cache_dir, cache_file = temp_cache_dir

        cache_embedding("model", "text", [0.1])
        save_cache()

        assert cache_dir.exists()
        assert cache_file.exists()

    def test_load_handles_missing_file(self, temp_cache_dir):
        """Loading when file doesn't exist initializes empty cache."""
        result = get_cached_embedding("model", "text")
        assert result is None

    def test_load_handles_corrupted_file(self, temp_cache_dir):
        """Loading corrupted file initializes empty cache."""
        cache_dir, cache_file = temp_cache_dir

        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("not valid json {{{")

        # Should not raise, just start with empty cache
        result = get_cached_embedding("model", "text")
        assert result is None


class TestClearCache:
    """Tests for cache clearing."""

    def test_clear_removes_memory_cache(self, temp_cache_dir):
        """Clear removes in-memory cache."""
        cache_embedding("model", "text", [0.1])
        assert get_cached_embedding("model", "text") is not None

        clear_cache()

        assert get_cached_embedding("model", "text") is None

    def test_clear_removes_disk_cache(self, temp_cache_dir):
        """Clear removes disk cache file."""
        cache_dir, cache_file = temp_cache_dir

        cache_embedding("model", "text", [0.1])
        save_cache()
        assert cache_file.exists()

        clear_cache()

        assert not cache_file.exists()


class TestCacheStats:
    """Tests for cache statistics."""

    def test_stats_empty_cache(self, temp_cache_dir):
        """Stats for empty cache."""
        stats = get_cache_stats()
        assert stats["cached_embeddings"] == 0

    def test_stats_with_entries(self, temp_cache_dir):
        """Stats reflect cached entries."""
        cache_embedding("model", "text1", [0.1])
        cache_embedding("model", "text2", [0.2])

        stats = get_cache_stats()
        assert stats["cached_embeddings"] == 2

    def test_stats_cache_size(self, temp_cache_dir):
        """Stats include file size after save."""
        cache_dir, cache_file = temp_cache_dir

        cache_embedding("model", "text", [0.1, 0.2, 0.3])
        save_cache()

        stats = get_cache_stats()
        assert stats["cache_size_bytes"] > 0
