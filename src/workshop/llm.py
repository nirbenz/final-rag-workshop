# Workshop LLM module
# Simplified LLM factory for pydantic-ai

import textwrap
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel
import pydantic_ai
from pydantic_ai import Agent, Embedder, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
import tqdm

from workshop.types import LLMConfig

# Toggle between exercise stubs and solutions for prompts:
# - Use exercises for workshop participants to implement
# - Use solutions for working reference implementation
USE_PROMPT_SOLUTIONS = True

if USE_PROMPT_SOLUTIONS:
    from workshop.rag.solutions.prompting import build_full_prompt
else:
    from workshop.rag.exercises.prompting import build_full_prompt


class APIInitializationError(Exception):
    """Error raised when model initialization fails."""

    pass


class ModelMessageError(Exception):
    """Error raised when a model message is invalid."""

    pass


def _route_model(model_config: LLMConfig) -> Model:
    """
    Route to the appropriate model provider based on config.

    Args:
        model_config: Configuration with model_name as "provider:model"

    Returns:
        Configured Model instance

    Raises:
        APIInitializationError: If model initialization fails
    """
    if ":" not in model_config["model_name"]:
        raise APIInitializationError(
            f"Model name {model_config['model_name']} must be in format <provider>:<model_name>"
        )

    model_provider, model_name = model_config["model_name"].split(":")
    settings = pydantic_ai.ModelSettings(**model_config.get("kwargs", {}))

    try:
        if model_provider == "anthropic":
            return AnthropicModel(model_name, settings=settings)
        elif model_provider == "openai":
            return OpenAIChatModel(model_name, settings=settings)
        elif model_provider in ("google", "vertexai", "gemini"):
            return GoogleModel(
                model_name,
                settings=settings,
                provider=GoogleProvider(vertexai=True),
            )
        else:
            raise ValueError(f"Model provider '{model_provider}' not supported")
    except Exception as e:
        raise APIInitializationError(f"Failed to initialize model {model_config['model_name']}: {e}")


def _add_history_handling(agent: Agent[Any, Any]) -> Agent[Any, Any]:
    """
    Add history canonization to convert dict messages to ModelMessage format.

    Args:
        agent: Agent to add history handling to

    Returns:
        Agent with history processor configured
    """

    def canonize_history(messages: List[ModelMessage] | List[Dict[str, str]]) -> List[ModelMessage]:
        res_messages: List[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, (ModelRequest, ModelResponse)):
                res_messages.append(msg)
            elif isinstance(msg, dict):
                if msg["role"] == "user":
                    res_messages.append(ModelRequest(parts=[UserPromptPart(content=msg["content"])]))
                else:
                    res_messages.append(ModelResponse(parts=[TextPart(content=msg["content"])]))
            else:
                raise ModelMessageError(f"Unknown message type: {type(msg)}")
        return res_messages

    agent.history_processors = [canonize_history]
    return agent


def create_agent(
    model_config: LLMConfig,
    *,
    structured_output_type: Optional[Type[BaseModel]] = None,
) -> Agent:
    """
    Create a pydantic-ai Agent from configuration.

    Args:
        model_config: Configuration for the model (must include model_name as "provider:model")
        structured_output_type: Optional Pydantic model for structured output

    Returns:
        Configured Agent instance

    Raises:
        APIInitializationError: If model initialization fails
    """
    model_settings = pydantic_ai.ModelSettings(**model_config.get("kwargs", {}))
    agent = pydantic_ai.Agent(
        model=model_config["model_name"],
        model_settings=model_settings,
        output_type=structured_output_type or str,
        retries=5,
        output_retries=5,
    )
    agent = _add_history_handling(agent)
    return agent


def get_pydantic_agent(
    model_config: LLMConfig, structured_output_type: Optional[Type[BaseModel]] = None
) -> Agent[Any, Any]:
    """
    Create an agent with a context-aware system prompt for RAG.

    This is the main entry point for workshop participants.
    The system prompt is built using prompts from exercises/solutions.

    Args:
        model_config: LLM configuration
        structured_output_type: Pydantic model for structured responses

    Returns:
        Configured Agent with RAG-style system prompt
    """
    agent = create_agent(model_config, structured_output_type=structured_output_type)

    @agent.instructions
    def system_prompt_input(ctx: RunContext[Any]) -> str:
        deps = ctx.deps or {}
        context = deps.get("context", "No context available.")
        return build_full_prompt(context)

    return agent


def get_embedding_model(model_config: LLMConfig) -> Embedder:
    """
    Create an embedding model from configuration.
    """
    if ":" not in model_config["model_name"]:
        raise APIInitializationError(
            f"Model name {model_config['model_name']} must be in format <provider>:<model_name>"
        )

    settings = pydantic_ai.EmbeddingSettings(**model_config.get("kwargs", {}))

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
        count = await embedder.count_tokens(text)
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
        input_type: "document" for storage, "query" for retrieval

    Returns:
        List of embedding vectors
    """
    import asyncio
    import threading

    result: List[List[float]] = []

    def run_async() -> None:
        nonlocal result
        result = asyncio.run(get_embeddings(embedder, texts, max_tokens, input_type))

    thread = threading.Thread(target=run_async, daemon=True)
    thread.start()
    thread.join()
    return result
