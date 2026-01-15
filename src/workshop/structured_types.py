# Workshop structured types for LLM responses
# Pydantic models for chain-of-thought and retrieval outputs

import textwrap
from typing import Generic, List, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

__all__ = [
    "Step",
    "CoTStage",
    "OneStageAnswer",
    "ComplexAnswer",
    "RetrievalResult",
    "RetrievalCoT",
]


class LLMBaseModel(BaseModel):
    """Base model for LLM structured outputs."""

    model_config = ConfigDict(extra="forbid")

    def to_message(self) -> str:
        """Convert to message string."""
        return self.model_dump_json()


class Step(LLMBaseModel):
    """A single reasoning step with explanation and output."""

    explanation: str
    output: str

    def to_message(self) -> str:
        return "\n\n".join([self.explanation, self.output])


class BaseStage(LLMBaseModel):
    """Base class for reasoning stages."""

    steps: List[Step] = Field(description="Steps taken to answer the question")

    def to_message(self) -> str:
        message = []
        for step in self.steps:
            message.append(f"- {step.explanation}")
            if step.output:
                message.append(f"    {step.output}")
        message = "\n".join(message)
        return textwrap.indent(message, "  ")


class OneStageAnswer(BaseStage):
    """A series of steps and a final answer."""

    output: str = Field(description="The answer to the user's question")

    def to_message(self):
        message = super().to_message()
        message = [message, f"Output: {self.output}\n"]
        return "\n".join(message)


class CoTStage(BaseStage):
    """A major stage in chain of thought reasoning."""

    title: str = Field(description="Title of this reasoning stage")

    def to_message(self) -> str:
        message = []
        message.append(f"### {self.title}")
        message.append(super().to_message())
        message = "\n".join(message)
        return message


class ComplexAnswer(BaseModel, Generic[T]):
    """Chain of thought reasoning with multiple stages and final answer."""

    stages: List[CoTStage] = Field(description="Major stages of reasoning")
    output: T = Field(description="Final synthesized answer")

    def to_message(self) -> str:
        message = []
        for stage in self.stages:
            stage_message = stage.to_message()
            message.append(stage_message)

        if isinstance(self.output, LLMBaseModel):
            message.append(f"\nFinal Output:\n{self.output.to_message()}")
        elif isinstance(self.output, str):
            message.append(f"\nFinal Output:\n{self.output}")
        else:
            raise ValueError(f"Unsupported output type: {type(self.output)}")
        return "\n".join(message)


class RetrievalResult(LLMBaseModel):
    """
    Result from a retrieval-augmented generation query.

    Contains the answer along with context attribution.
    """

    query: str = Field(description="The user's query, re-explained")
    output: str = Field(description="The answer to the user's query")
    explanation: str = Field(description="Explanation for context selection")
    context_used: str = Field(description="Context used to determine the answer")

    def to_message(self) -> str:
        message = []
        message.append(f"- Query: {self.query}")
        message.append(f"- Output: {self.output}")
        message.append(f"- Explanation: {self.explanation}")
        message.append(f"- Context Used: {self.context_used}")
        message = "\n".join(message)
        return textwrap.indent(message, "  ")


class RetrievalCoT(ComplexAnswer[RetrievalResult]):
    """Chain of thought with retrieval result as final answer."""

    pass
