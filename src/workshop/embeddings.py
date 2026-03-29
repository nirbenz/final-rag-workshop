# Workshop embeddings module
# Embedding generation and caching utilities

import textwrap
from typing import List, Optional

from loguru import logger
import pydantic_ai
from pydantic_ai import Embedder
import tqdm

from workshop.types import LLMConfig


class APIInitializationError(Exception):
    """Error raised when model initialization fails."""

    pass


def get_embedding_model(model_config: LLMConfig) -> Embedder:
    """
    Create an embedding model from configuration.

    Args:
        model_config: Configuration with model_name as "provider:model"

    Returns:
        Configured Embedder instance

    Raises:
        APIInitializationError: If model name format is invalid
    """
    import os

    if ":" not in model_config["model_name"]:
        raise APIInitializationError(
            f"Model name {model_config['model_name']} must be in format <provider>:<model_name>"
        )

    # Debug: Log API key status
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    logger.info(f"Creating embedder with model={model_config['model_name']}")
    logger.info(f"OPENAI_API_KEY={'set ('+api_key[:10]+'...)' if api_key else 'NOT SET'}")
    logger.info(f"OPENAI_BASE_URL={base_url or 'NOT SET'}")

    # Get embedding settings from config
    settings_dict = model_config.get("kwargs", {}).copy()

    # For litellm: models, explicitly set the base_url and api_key
    if model_config["model_name"].startswith("litellm:"):
        if base_url:
            settings_dict["base_url"] = base_url
        if api_key:
            settings_dict["api_key"] = api_key
            logger.info(f"Explicitly passing API key to Embedder for litellm model")

    settings = pydantic_ai.EmbeddingSettings(**settings_dict)

    return Embedder(model=model_config["model_name"], settings=settings)


def _prepare_texts_for_embedding(
    texts: List[str],
    token_counts: List[int],
    max_tokens: int,
) -> List[List[str]]:
    """
    Preprocess texts for embedding: split long texts and batch by token limit.

    This is a pure function that prepares texts for the embedding API.
    It splits texts that exceed max_tokens and groups them into batches.

    Args:
        texts: Original texts to embed
        token_counts: Token count for each text (same length as texts)
        max_tokens: Maximum tokens per text/batch

    Returns:
        List of text batches, each batch fits within max_tokens
    """
    if max_tokens is None:
        raise ValueError("max_tokens is required")

    # Split long texts into chunks
    processed_texts = []
    processed_counts = []
    for text, token_count in zip(texts, token_counts):
        if token_count > max_tokens:
            # Split by characters (approximation, textwrap uses chars not tokens)
            text_chunks = textwrap.wrap(text, max_tokens * 4)  # ~4 chars per token
            for chunk in text_chunks:
                processed_texts.append(chunk)
                # Approximate token count for chunk
                processed_counts.append(len(chunk) // 4)
        else:
            processed_texts.append(text)
            processed_counts.append(token_count)

    # Batch texts to fit within max_tokens per batch
    text_batches = []
    current_batch: List[str] = []
    current_batch_count = 0

    for text, token_count in zip(processed_texts, processed_counts):
        if current_batch and current_batch_count + token_count > max_tokens:
            text_batches.append(current_batch)
            current_batch = [text]
            current_batch_count = token_count
        else:
            current_batch.append(text)
            current_batch_count += token_count

    if current_batch:
        text_batches.append(current_batch)

    return text_batches


async def _count_tokens(embedder: Embedder, texts: List[str]) -> List[int]:
    """
    Count tokens for each text using the embedder.

    Args:
        embedder: Pydantic-AI Embedder instance
        texts: List of texts to count tokens for

    Returns:
        List of token counts (same length as texts)
    """
    counts = []
    for text in texts:
        try:
            count = await embedder.count_tokens(text)
        except Exception:
            # Fallback for non-OpenAI models: approximate 4 chars per token
            count = len(text) // 4
        counts.append(count)
    return counts


async def get_embeddings(
    embedder: Embedder, texts: List[str], max_tokens: int, input_type: str = "document"
) -> List[List[float]]:
    """
    Get embeddings for a list of texts (async version) with caching.

    Args:
        embedder: Pydantic-AI Embedder instance
        texts: List of texts to embed
        max_tokens: Maximum tokens per batch
        input_type: "document" for storage, "query" for retrieval

    Returns:
        List of embedding vectors
    """
    from workshop.embedding_cache import cache_embedding, get_cached_embedding, save_cache

    # Count tokens and prepare batches
    token_counts = await _count_tokens(embedder, texts)
    text_batches = _prepare_texts_for_embedding(texts, token_counts, max_tokens)

    total_texts = sum(len(batch) for batch in text_batches)
    logger.info(f"Prepared {len(text_batches)} batches with {total_texts} texts")

    model_name = str(embedder.model_name) if hasattr(embedder, "model_name") else "unknown"  # pyright: ignore[reportAttributeAccessIssue]

    results = []
    cache_hits = 0
    cache_misses = 0

    for text_batch in tqdm.tqdm(text_batches, desc="Embedding batches"):
        batch_results: list = [None] * len(text_batch)
        uncached_texts = []
        uncached_positions = []

        # Check cache for each text in batch
        for i, text in enumerate(text_batch):
            cached = get_cached_embedding(model_name, text)
            if cached is not None:
                batch_results[i] = cached
                cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_positions.append(i)
                cache_misses += 1

        # Only call API for uncached texts
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)}/{len(text_batch)} uncached texts")
            if input_type == "query":
                result = await embedder.embed_query(uncached_texts)
            else:
                result = await embedder.embed_documents(uncached_texts)

            # Fill in results and cache new embeddings
            for pos, emb in zip(uncached_positions, result.embeddings):
                emb_list = list(emb)
                batch_results[pos] = emb_list
                cache_embedding(model_name, text_batch[pos], emb_list)
        else:
            logger.info(f"Cache hit: all {len(text_batch)} texts cached")

        results.extend(batch_results)

    # Persist cache to disk after all batches
    save_cache()

    logger.info(
        f"Got {len(results)} embeddings for {len(texts)} texts " f"(cache: {cache_hits} hits, {cache_misses} misses)"
    )
    return results


def get_embeddings_sync(
    embedder: Embedder, texts: List[str], max_tokens: int, input_type: str = "document"
) -> List[List[float]]:
    """
    Get embeddings for a list of texts (sync version).

    This is the recommended function for workshop participants. It runs
    the async embedding call in a separate thread with its own event loop,
    making it safe to call from any context (sync or async).

    Args:
        embedder: Pydantic-AI Embedder instance
        texts: List of texts to embed
        max_tokens: Maximum tokens per batch
        input_type: "document" for storage, "query" for retrieval

    Returns:
        List of embedding vectors

    Raises:
        Exception: Re-raises any exception from the embedding thread
    """
    import asyncio
    import threading

    result: List[List[float]] = []
    error: Optional[BaseException] = None

    def run_async() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(get_embeddings(embedder, texts, max_tokens, input_type))
        except BaseException as e:
            error = e

    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()
    thread.join()

    if error is not None:
        raise error

    return result
