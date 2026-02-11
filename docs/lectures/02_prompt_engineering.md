# Lecture 02: Prompt Engineering for RAG

> **Phase:** Phase 2 - The "G" in RAG

---

## Where We Are

```
              *** YOU ARE HERE ***
┌──────────┐         ┌─────────────┐         ┌─────┐
│ Chunking │ ──all──>│  PROMPTING  │ ──────> │ LLM │ ──> Response
└──────────┘ chunks  └─────────────┘         └─────┘
  Phase 1             Phase 2
```

*Phase 2 architecture: prompting layer added between chunks and LLM. Retrieval is still naive.*

Retrieval is still naive (all chunks). We are focusing on how the LLM **uses** context.

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand why prompting matters even with good retrieval
2. Know the components of an effective RAG prompt
3. Design context formatting strategies
4. (Advanced) Implement chain-of-thought and structured output

---

## Core Content

### 1. Retrieval is Half the Battle

**The Problem:**

```
Great retrieval + Bad prompt = Bad answer
```

**Side-by-Side:**

```
Retrieved context: "John mentioned the deadline is Friday"

Bad prompt:  "Answer the question: When is the deadline?"
Response:    "The deadline is Friday." (no source, no confidence)

Good prompt: "Based ONLY on the context below, answer the question.
              Cite your sources. If unsure, say so."
Response:    "According to John, the deadline is Friday [1]."
```

**Key Insight:** How you ask matters as much as what you provide.

---

### 2. System Prompt Design

**The Four Components:**

```
┌─────────────────────────────────────────────┐
│              SYSTEM PROMPT                  │
├─────────────────────────────────────────────┤
│ 1. ROLE ASSIGNMENT                          │
│    "You are a helpful assistant that..."    │
│                                             │
│ 2. TASK DESCRIPTION                         │
│    "Answer questions using ONLY the         │
│     provided conversation context..."       │
│                                             │
│ 3. CONSTRAINTS                              │
│    "Be concise. Cite sources. Don't         │
│     make up information..."                 │
│                                             │
│ 4. FALLBACK BEHAVIOR                        │
│    "If the context doesn't contain          │
│     relevant information, say so..."        │
└─────────────────────────────────────────────┘
```

*The four building blocks of a RAG system prompt: role, task, constraints, and fallback behavior.*

> **Slide guidance:** Present the four components as one slide, the example prompt as a separate slide.

**Example (plain text response -- no Pydantic needed):**

```
You are a helpful assistant that answers questions about WhatsApp conversations.

Your task:
- Answer questions using ONLY the provided conversation context
- Be concise and direct (2-3 sentences)
- Quote relevant parts of the conversation when helpful
- Cite which chunk your answer comes from using [1], [2], etc.

If the context doesn't contain enough information to answer:
- Say "I don't have enough information to answer that"
- Don't make up or infer information not in the context

<context>
{context}
</context>
```

This works without any structured output -- the LLM returns a plain string. Start here.

---

### 3. Context Formatting

> **Slide guidance:** Each formatting option (numbered list, XML tags, with metadata) is a separate slide.

**Option 1: Numbered List**

```
Context:
[1] John: Hey, can we move the meeting to 3pm?
    Sarah: Sure, that works for me.

[2] Sarah: Don't forget the deadline is Friday.
    John: Got it, I'll have it ready by Thursday.
```

**Benefit:** Enables citations like "According to [2]..."

**Option 2: XML Tags**

```xml
<context>
<chunk id="1">
John: Hey, can we move the meeting to 3pm?
Sarah: Sure, that works for me.
</chunk>
<chunk id="2">
Sarah: Don't forget the deadline is Friday.
John: Got it, I'll have it ready by Thursday.
</chunk>
</context>
```

**Benefit:** Clear boundaries, works well with Claude and GPT-4.

**Option 3: With Metadata**

```
[Chunk 1 | 2024-01-15 14:30 | Speakers: John, Sarah]
John: Hey, can we move the meeting to 3pm?
Sarah: Sure, that works for me.
```

**Benefit:** Additional context for temporal and speaker questions.

---

### 4. Output Instructions

**Control the Response:**

| Instruction | Effect |
|-------------|--------|
| "Be concise, 2-3 sentences" | Prevents rambling |
| "Quote relevant parts" | Grounds response in context |
| "Rate confidence: High/Medium/Low" | Honest uncertainty |
| "If unsure, say so" | Reduces hallucination |

---

## What Changed: Phase 1 -> Phase 2

| Before (Phase 1) | After (Phase 2) |
|---|---|
| Default system prompt | Your custom prompt |
| Unstructured responses | Controlled format |
| No grounding instructions | "Use ONLY the context" |
| No fallback behavior | Explicit "I don't know" when context is insufficient |

**Next up:** Phase 2 hands-on -- design your own system prompt.

---

## Key Concepts: CoT and Structured Output

### Chain-of-Thought (CoT) Prompting

> **Slide guidance:** The without/with CoT side-by-side is one slide; the "When to Use CoT" table is a second slide.

LLMs often jump to conclusions without showing their work. In RAG, this means:
- No visibility into which chunks informed the answer
- Harder to debug incorrect responses

**The Fix: Step-by-Step Reasoning**

```
┌─────────────────────────────────────────────────────────┐
│                WITHOUT CHAIN-OF-THOUGHT                 │
├─────────────────────────────────────────────────────────┤
│ Q: When did they plan to meet?                          │
│ A: Friday at 3pm.                                       │
│                                                         │
│    (Where did this come from? Is it correct?)           │
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

*Side-by-side: same question answered without CoT (opaque) vs with CoT (traceable, cited).*

**When to Use CoT:**

| Use CoT | Skip CoT |
|---------|----------|
| Complex multi-hop questions | Simple factual lookups |
| Debugging retrieval issues | High-throughput applications |
| When citations are critical | When latency matters most |

---

### Structured Output with Pydantic

> **Slide guidance:** The RAGResponse schema is one slide; the integration code + design tip are a second slide.

**Going beyond plain text:** Force the LLM to return a specific schema.

The workshop uses `RAGResponse` for structured chain-of-thought:

```python
class ReasoningStep(BaseModel):
    thought: str     # What you are checking
    observation: str # What you found

class RAGResponse(BaseModel):
    query_understanding: str              # Restate the question
    reasoning_steps: List[ReasoningStep]  # Step-by-step analysis
    context_used: List[str]               # Exact quotes from context
    output: str                           # Final answer
    confidence: Literal["high", "medium", "low"]
```

**How it integrates:**

```python
agent = Agent(
    model="openai:gpt-4",
    output_type=RAGResponse,  # Enforces this schema
)

result = await agent.run("When is the meeting?")
print(result.output.output)        # "The meeting is at 3pm Friday"
print(result.output.confidence)    # "high"
```

**Key Insight:** Structured outputs make responses predictable and parseable.
The chain-of-thought fields (`reasoning_steps`, `context_used`) help with debugging and traceability.

**Design Tip:** Always include an `output: str` field for the displayable answer.

> More advanced prompting techniques (confidence calibration, few-shot, combining techniques)
> are covered in [Lecture 05: Advanced Topics](05_advanced_topics.md).

---

## Instructor Notes

- Show side-by-side: same retrieval, different prompts, different answers
- Start with plain text prompts, introduce Pydantic only as an upgrade path
- Emphasize experimentation: "There's no one right prompt"
- Warn against over-engineering: "Start simple, iterate"
- CoT and Pydantic are the two key upgrades -- present these if time allows

---

## Code References

- Exercise: `src/workshop/rag/exercises/prompting.py`
- Solution: `src/workshop/rag/solutions/prompting.py`
- LLM integration: `src/workshop/llm.py`
- Structured types: `src/workshop/structured_types.py`

---

## Further Reading

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Chain-of-Thought Prompting (Wei et al.)](https://arxiv.org/abs/2201.11903)
- [Pydantic-AI Documentation](https://ai.pydantic.dev/)
