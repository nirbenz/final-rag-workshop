# Created by Nir Ben-Zvi
# Embedding cache for workshop
# me@nirbnzvi.com

"""
Disk-based embedding cache for workshop.

Caches embeddings by (model_name, text) to avoid redundant API calls.
Cache persists across restarts in .embedding_cache/embeddings.json.

This cache operates at the final text level (after any splitting/batching),
so cache hits are maximized regardless of batching strategy.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

_CACHE_DIR = Path(".embedding_cache")
_CACHE_FILE = _CACHE_DIR / "embeddings.json"
_cache: Optional[Dict[str, List[float]]] = None


def _ensure_loaded() -> Dict[str, List[float]]:
    """Load cache from disk if not already loaded."""
    global _cache
    if _cache is None:
        _CACHE_DIR.mkdir(exist_ok=True)
        if _CACHE_FILE.exists():
            try:
                loaded: Dict[str, List[float]] = json.loads(_CACHE_FILE.read_text())
                _cache = loaded
                logger.info(f"Loaded {len(loaded)} cached embeddings from disk")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                _cache = {}
        else:
            _cache = {}
    assert _cache is not None
    return _cache


def _cache_key(model_name: str, text: str) -> str:
    """Create cache key from model + text hash."""
    content = f"{model_name}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def get_cached_embedding(model_name: str, text: str) -> Optional[List[float]]:
    """
    Get embedding from cache.

    Args:
        model_name: Embedding model identifier (e.g., "openai:text-embedding-3-small")
        text: Text that was embedded

    Returns:
        Cached embedding vector, or None if not cached
    """
    cache = _ensure_loaded()
    return cache.get(_cache_key(model_name, text))


def cache_embedding(model_name: str, text: str, embedding: List[float]) -> None:
    """
    Store embedding in cache (memory only, call save_cache to persist).

    Args:
        model_name: Embedding model identifier
        text: Text that was embedded
        embedding: Embedding vector to cache
    """
    cache = _ensure_loaded()
    cache[_cache_key(model_name, text)] = embedding


def save_cache() -> None:
    """Persist cache to disk."""
    if _cache is not None:
        try:
            _CACHE_FILE.write_text(json.dumps(_cache))
            logger.debug(f"Saved {len(_cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")


def clear_cache() -> None:
    """Clear all cached embeddings (memory and disk)."""
    global _cache
    _cache = {}
    if _CACHE_FILE.exists():
        try:
            _CACHE_FILE.unlink()
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.warning(f"Failed to delete cache file: {e}")


def get_cache_stats() -> Dict[str, int]:
    """Get cache statistics."""
    cache = _ensure_loaded()
    return {
        "cached_embeddings": len(cache),
        "cache_size_bytes": _CACHE_FILE.stat().st_size if _CACHE_FILE.exists() else 0,
    }
