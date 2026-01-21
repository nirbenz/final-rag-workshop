# Workshop structured types for LLM responses
# Simplified Pydantic models for RAG outputs

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RAGResponse",
    # Keep legacy exports for backward compatibility
    "RetrievalCoT",
    "RetrievalResult",
]


class RAGResponse(BaseModel):
    """
    Structured response from RAG query.

    This is a flat, simple model for workshop use. The LLM returns:
    - output: The answer to display to the user
    - reasoning: Optional chain-of-thought explanation
    - context_used: Optional context excerpt used for the answer (for highlighting)
    - confidence: Optional confidence level

    The parser in state.py looks for `.output` and falls back to JSON display.
    """

    model_config = ConfigDict(extra="allow")

    output: str = Field(description="The answer to the user's question")
    reasoning: Optional[str] = Field(
        default=None,
        description="Chain-of-thought reasoning or explanation",
    )
    context_used: Optional[str] = Field(
        default=None,
        description="Excerpt from context used to answer (for UI highlighting)",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence level: high, medium, or low",
    )


# Legacy aliases for backward compatibility
# These are deprecated but kept to avoid breaking existing code
RetrievalCoT = RAGResponse
RetrievalResult = RAGResponse
