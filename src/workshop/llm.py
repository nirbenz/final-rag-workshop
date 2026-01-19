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

    Args:
        model_config: LLM configuration
        structured_output_type: Pydantic model for structured responses

    Returns:
        Configured Agent with RAG-style system prompt
    """
    agent = create_agent(model_config, structured_output_type=structured_output_type)

    context_prompt = textwrap.dedent("""
        Use the following context to answer the question:
        <context>
        {context}
        </context>
    """)

    @agent.instructions
    def system_prompt_input(ctx: RunContext[Any]) -> str:
        deps = ctx.deps or {}
        if not deps.get("context"):
            deps = {"context": "N/A"}
        return context_prompt.format(**deps)

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


async def get_embeddings(
    embedder: Embedder, texts: List[str], max_tokens: int, input_type: str = "document"
) -> List[List[float]]:
    """
    Get embeddings for a list of texts (async version).

    Pydantic-AI handles batching internally. This function adds guardrails
    for token limits and provides consistent error handling.

    Args:
        embedder: Pydantic-AI Embedder instance
        texts: List of texts to embed
        input_type: "document" for storage, "query" for retrieval

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If model doesn't support max_input_tokens
    """
    # Count tokens in text
    token_counts = []
    for text in texts:
        token_count = await embedder.count_tokens(text)
        token_counts.append((text, token_count))

    total_token_counts = sum(token_count for _, token_count in token_counts)

    if max_tokens is None:
        raise ValueError("Model does not support maximum input tokens")

    # split long texts into chunks
    token_counts_updated = []
    for text, token_count in token_counts:
        if len(text) > max_tokens:
            text_chunks = textwrap.wrap(text, max_tokens)
            token_counts_updated.extend(
                [(text_chunk, await embedder.count_tokens(text_chunk)) for text_chunk in text_chunks]
            )
        else:
            token_counts_updated.append((text, token_count))

    token_counts = token_counts_updated
    total_token_counts = sum(token_count for _, token_count in token_counts)

    # batch texts if needed
    text_batches = []
    current_batch = []
    current_batch_count = 0
    for tidx, (text, token_count) in enumerate(token_counts):
        if tidx > 0 and current_batch_count + token_count > max_tokens:
            text_batches.append(current_batch)
            current_batch = [text]
            current_batch_count = token_count
        else:
            current_batch.append(text)
            current_batch_count += token_count
    if current_batch:
        text_batches.append(current_batch)
    logger.info(f"Batched {len(text_batches)} texts into {total_token_counts} tokens")

    results = []
    for text_batch in tqdm.tqdm(text_batches, desc="Embedding batches"):
        logger.info(f"Embedding batch of {len(text_batch)} texts")
        if input_type == "query":
            result = await embedder.embed_query(text_batch)
        else:
            result = await embedder.embed_documents(text_batch)
        results.extend([list(vec) for vec in result.embeddings])

    logger.info(f"Got {len(results)} embeddings for {len(texts)} texts, totalling {token_count} tokens")
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
