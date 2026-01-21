# Lecture 06: Advanced Prompting Techniques

> **Duration:** 15 minutes (stretch goal content)
> **Phase:** Phase 3B Extension

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand Chain-of-Thought (CoT) prompting and when to use it
2. Know how to calibrate LLM confidence in RAG responses
3. Implement few-shot prompting for consistent output formats
4. Design custom Pydantic models for structured LLM output

---

## Outline

### 1. Chain-of-Thought Prompting (4 minutes)

**The Problem:**

LLMs often jump to conclusions without showing their work. In RAG, this means:
- No visibility into which chunks informed the answer
- Harder to debug incorrect responses
- Less grounded, more hallucinatory outputs

**The Solution: Step-by-Step Reasoning**

```
┌─────────────────────────────────────────────────────────┐
│                WITHOUT CHAIN-OF-THOUGHT                 │
├─────────────────────────────────────────────────────────┤
│ Q: When did they plan to meet?                          │
│ A: Friday at 3pm.                                       │
│                                                         │
│ (Where did this come from? Is it correct?)              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 WITH CHAIN-OF-THOUGHT                   │
├─────────────────────────────────────────────────────────┤
│ Q: When did they plan to meet?                          │
│                                                         │
│ Let me analyze the relevant context:                    │
│ 1. Chunk [2] mentions "Can we move to 3pm?"             │
│ 2. Chunk [2] shows Sarah confirming "that works"        │
│ 3. Chunk [3] mentions "see you Friday"                  │
│                                                         │
│ A: They planned to meet Friday at 3pm [2][3].           │
└─────────────────────────────────────────────────────────┘
```

**Implementation Pattern:**

```python
def get_system_prompt() -> str:
    return """You answer questions about WhatsApp conversations.

IMPORTANT: Before answering, think through these steps:
1. Which chunks contain information relevant to the question?
2. What specific evidence supports the answer?
3. Are there any contradictions or ambiguities?

Show your reasoning, then provide a clear answer with citations."""
```

**When to Use CoT:**

| Use CoT | Skip CoT |
|---------|----------|
| Complex multi-hop questions | Simple factual lookups |
| Debugging retrieval issues | High-throughput applications |
| When citations are critical | When latency matters most |

---

### 2. Confidence Calibration (3 minutes)

**The Problem:**

LLMs are confidently wrong. They don't naturally express uncertainty.

**The Solution: Explicit Confidence Instructions**

```python
def get_output_instructions() -> str:
    return """After your answer, rate your confidence:

- HIGH: The answer is directly and clearly stated in the context
- MEDIUM: The answer requires reasonable inference from the context
- LOW: The context is ambiguous, incomplete, or potentially outdated

If confidence is LOW, explain what additional information would help."""
```

**Example Output:**

```
Q: What time is the meeting tomorrow?

A: The meeting is scheduled for 2pm tomorrow. (Confidence: MEDIUM)

Note: Chunk [3] mentions "2pm works" but doesn't explicitly confirm
this is the final agreed time. There may be later messages with changes.
```

**Why This Matters for RAG:**

- Retrieval might miss relevant chunks (false confidence)
- Context might be outdated (temporal uncertainty)
- Multiple chunks might contradict (conflicting evidence)

---

### 3. Few-Shot Prompting (4 minutes)

**The Problem:**

Instructions alone don't guarantee consistent output format.

**The Solution: Show, Don't Tell**

```python
def get_system_prompt() -> str:
    return """You answer questions about WhatsApp conversations.
Format your responses exactly like these examples:

---
Q: Who suggested the restaurant?
A: John suggested trying the new Italian place on Main Street [2].
   Sarah agreed it sounded good [3].

Q: When is the deadline?
A: The project deadline is Friday at 5pm [1]. Mike confirmed he'd
   have his part done by Thursday [4].

Q: What was the final decision about the trip?
A: I couldn't find a clear final decision in the context. Chunks [2]
   and [5] discuss options (Paris vs Rome) but no conclusion is shown.
---

Now answer the user's question in the same format."""
```

**Key Principles:**

1. **Show the exact format** you want (citations, length, structure)
2. **Include edge cases** (e.g., "couldn't find" example)
3. **Match the domain** (conversation-style examples for chat data)
4. **Keep it short** (2-3 examples is usually enough)

**Prompt Token Budget:**

| Component | Typical Tokens |
|-----------|----------------|
| System instructions | 100-200 |
| Few-shot examples | 200-400 |
| Retrieved context | 1000-4000 |
| User query | 20-100 |

**Trade-off:** More examples = more consistent format, but less room for context.

---

### 4. Structured Output with Pydantic (4 minutes)

**The Problem:**

Free-form text is hard to parse programmatically.

**The Solution: Pydantic Models as Output Schema**

```python
from pydantic import BaseModel, Field
from typing import List, Literal


class RAGResponse(BaseModel):
    """Structured response for RAG queries."""

    # The main answer (displayed to user)
    output: str = Field(
        description="Direct answer to the user's question"
    )

    # Confidence assessment
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident are you in this answer?"
    )

    # Source attribution
    source_chunks: List[int] = Field(
        description="Which chunk numbers support this answer?"
    )

    # Reasoning trace
    reasoning: str = Field(
        description="Brief explanation of how you arrived at this answer"
    )

    # Uncertainty flag
    needs_more_context: bool = Field(
        default=False,
        description="True if more context would significantly improve the answer"
    )
```

**How Pydantic-AI Uses This:**

```python
from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4",
    output_type=RAGResponse,  # Enforces this schema
)

result = await agent.run("When is the meeting?", deps={"context": chunks})
# result.output is a RAGResponse instance, not a string!

print(result.output.output)        # "The meeting is at 3pm Friday"
print(result.output.confidence)    # "high"
print(result.output.source_chunks) # [2, 3]
```

**The Workshop's RetrievalCoT Model:**

```python
# Already defined in src/workshop/structured_types.py

class RetrievalResult(BaseModel):
    query: str       # Rephrased question
    output: str      # The answer
    explanation: str # Why this answer
    context_used: str # Which context was relevant

class RetrievalCoT(ComplexAnswer[RetrievalResult]):
    # Includes reasoning stages + final RetrievalResult
    pass
```

**Custom Model Integration:**

The `extract_llm_response()` function in `state.py` handles any Pydantic model:

1. Looks for `.output` attribute (your answer field)
2. Falls back to JSON serialization
3. Falls back to `str()` as last resort

**Design Tip:** Always include an `output: str` field for the displayable answer.

---

## Combining Techniques

**The Ultimate RAG Prompt:**

```python
def build_full_prompt(context: str) -> str:
    return f"""You are a helpful assistant answering questions about conversations.

## Instructions
1. First, identify which chunks are relevant to the question
2. Extract key evidence from those chunks
3. Synthesize a clear, cited answer
4. Rate your confidence based on evidence quality

## Examples
Q: Who organized the event?
A: Sarah organized the birthday party [1][3]. She coordinated with
   John for decorations [2]. (Confidence: HIGH)

Q: What was decided about the budget?
A: The context discusses budget concerns [4] but no final decision
   is shown. (Confidence: LOW - need more recent messages)

## Context
{context}

## Response Format
Provide your answer with [chunk numbers] citations, followed by
your confidence level (HIGH/MEDIUM/LOW)."""
```

---

## Instructor Notes

- Demo the difference between CoT and non-CoT responses
- Show a real example where confidence calibration caught a wrong answer
- Emphasize: These techniques stack - use them together
- Warning: More structure = more tokens = higher cost

---

## Slides

> **TODO:** Create presentation slides for this lecture

---

## Code References

- Structured types: `src/workshop/structured_types.py`
- Response extraction: `src/nicegui_app/state.py` (`extract_llm_response`)
- Prompt exercises: `src/workshop/rag/exercises/prompting.py`
- LLM integration: `src/workshop/llm.py`

---

## Further Reading

- [Chain-of-Thought Prompting (Wei et al.)](https://arxiv.org/abs/2201.11903)
- [Language Models are Few-Shot Learners (GPT-3 paper)](https://arxiv.org/abs/2005.14165)
- [Pydantic-AI Documentation](https://ai.pydantic.dev/)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
