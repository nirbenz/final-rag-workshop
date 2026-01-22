# Workshop LLM module
# Simplified LLM factory for pydantic-ai

from typing import Any, Dict, List, Optional, Type
import os

from pydantic import BaseModel
import httpx

# Monkey-patch OpenAI to disable SSL verification if needed
if os.getenv("SSL_VERIFY", "true").lower() == "false":
    import openai

    # Patch sync client
    _original_init = openai.OpenAI.__init__
    def _patched_init(self, *args, **kwargs):
        if "http_client" not in kwargs:
            kwargs["http_client"] = httpx.Client(verify=False)
        return _original_init(self, *args, **kwargs)
    openai.OpenAI.__init__ = _patched_init

    # Patch async client (used by Pydantic-AI)
    _original_async_init = openai.AsyncOpenAI.__init__
    def _patched_async_init(self, *args, **kwargs):
        if "http_client" not in kwargs:
            kwargs["http_client"] = httpx.AsyncClient(verify=False)
        return _original_async_init(self, *args, **kwargs)
    openai.AsyncOpenAI.__init__ = _patched_async_init

import pydantic_ai
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

# Import per-exercise toggle (separate file to avoid circular imports)
from workshop.exercise_toggles import USE_PROMPTING_SOLUTION
from workshop.types import LLMConfig

if USE_PROMPTING_SOLUTION:
    from workshop.rag.solutions.prompting import get_system_prompt
else:
    from workshop.rag.exercises.prompting import get_system_prompt

# Re-export embedding functions for backward compatibility
# Import from workshop.embeddings for new code
from workshop.embeddings import (
    APIInitializationError,
    get_embedding_model,
    get_embeddings,
    get_embeddings_sync,
)

__all__ = [
    "APIInitializationError",
    "ModelMessageError",
    "create_agent",
    "get_pydantic_agent",
    # Re-exported from embeddings module
    "get_embedding_model",
    "get_embeddings",
    "get_embeddings_sync",
]


class ModelMessageError(Exception):
    """Error raised when a model message is invalid."""

    pass


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
    The system prompt template is loaded from exercises/solutions, and
    the context is injected here at runtime.

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
        # Get the template and inject context here
        return get_system_prompt().format(context=context)

    return agent
