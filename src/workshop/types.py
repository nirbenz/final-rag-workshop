# Workshop types module
# Simplified type definitions for the workshop

from typing import Any, Dict, Optional, Type, TypedDict

from pydantic import BaseModel


class LLMConfig(TypedDict):
    """
    Configuration for LLM models.

    Used with pydantic-ai to configure model providers.
    """

    model_name: str
    kwargs: Dict[str, Any]
    structured_output_type: Optional[Type[BaseModel]]


__all__ = ["LLMConfig"]
