# Workshop structured types for LLM responses
# Pydantic models for RAG outputs with chain-of-thought reasoning

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RAGResponse",
]


class ReasoningStep(BaseModel):
    """
    A single step in the reasoning process.

    Each step explains what the model is doing and what it found.
    """

    thought: str = Field(
        description="What you are thinking or checking at this step",
    )
    observation: str = Field(
        description="What you found or concluded from this step",
    )


class RAGResponse(BaseModel):
    """
    Structured response from RAG query with chain-of-thought reasoning.

    The LLM fills in each field to show its reasoning process:
    1. First, restate the query to ensure understanding
    2. Walk through reasoning steps examining the context
    3. Identify the specific context that answers the question
    4. Provide the final answer
    5. Rate confidence based on how well context supports the answer

    Example response:
    {
        "query_understanding": "User wants to know when the trip to Paris was discussed",
        "reasoning_steps": [
            {"thought": "Looking for mentions of Paris in the context", "observation": "Found discussion in Chunk 2"},
            {"thought": "Checking who mentioned it and when", "observation": "Alice mentioned it on March 15"}
        ],
        "context_used": ["Alice: Let's plan a trip to Paris next month!", "Bob: I'm not sure about that, I think we should go to London instead."],
        "output": "The trip to Paris was discussed by Alice on March 15th.",
        "confidence": "high"
    }
    """

    model_config = ConfigDict(extra="allow")

    query_understanding: str = Field(
        description="Restate the user's question in your own words to show you understand what they're asking",
    )
    reasoning_steps: List[ReasoningStep] = Field(
        description="Step-by-step reasoning showing how you analyzed the context to find the answer",
        min_length=1,
    )
    context_used: List[str] = Field(
        description="Direct quote from the provided context that supports your answer. Should be a list of exact quotes.",
    )
    output: str = Field(
        description="The final answer to the user's question, based on the context",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level: 'high' if context directly answers, 'medium' if inferred, 'low' if uncertain",
    )

    def __str__(self) -> str:
        """
        Format as readable markdown for display.

        Returns the full response with all fields formatted nicely.
        """
        parts = [f"## Query Understanding\n{self.query_understanding}"]

        # Format reasoning steps
        steps_lines = []
        for i, step in enumerate(self.reasoning_steps, 1):
            steps_lines.append(f"{i}. **Thought:** {step.thought}\n   **Observation:** {step.observation}")
        parts.append("## Reasoning Steps\n" + "\n".join(steps_lines))

        # Format context_used as bullet list
        context_items = "\n".join(f"- {quote}" for quote in self.context_used)
        parts.append(f"## Context Used\n{context_items}")

        parts.append(f"## Output\n{self.output}")
        parts.append(f"## Confidence\n{self.confidence}")

        return "\n\n".join(parts)
