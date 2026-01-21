# Lecture 04: Prompt Engineering for RAG

> **Duration:** 10 minutes
> **Phase:** Phase 3B - The "G" in RAG

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand why prompting matters even with good retrieval
2. Know the components of an effective RAG prompt
3. Learn context formatting strategies
4. Be ready to design their own prompts

---

## Outline

### 1. Retrieval is Half the Battle (2 minutes)

**The Problem:**

```
Great retrieval + Bad prompt = Bad answer
```

**Example:**

```
Retrieved context: "John mentioned the deadline is Friday"

Bad prompt: "Answer the question: When is the deadline?"
Result: "The deadline is Friday." (no source, no confidence)

Good prompt: "Based ONLY on the context below, answer the question.
              Cite your sources. If unsure, say so."
Result: "According to John, the deadline is Friday [1]."
```

**Key Insight:** How you ask matters as much as what you provide.

---

### 2. System Prompt Design (3 minutes)

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

**Example System Prompt:**

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
```

---

### 3. Context Formatting (3 minutes)

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

**Benefit:** Clear boundaries, works well with Claude/GPT-4.

**Option 3: With Metadata**

```
[Chunk 1 | 2024-01-15 14:30 | Speakers: John, Sarah]
John: Hey, can we move the meeting to 3pm?
Sarah: Sure, that works for me.
```

**Benefit:** Additional context for temporal/speaker questions.

---

### 4. Output Instructions (2 minutes)

**Control the Response:**

| Instruction | Effect |
|-------------|--------|
| "Be concise, 2-3 sentences" | Prevents rambling |
| "Quote relevant parts" | Grounds response in context |
| "Rate confidence: High/Medium/Low" | Honest uncertainty |
| "If unsure, say so" | Reduces hallucination |

**Structured Output Example:**

```python
class RAGResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: List[int]  # Chunk IDs used
    reasoning: str  # Chain-of-thought
```

**The Workshop Uses:** `RetrievalCoT` model for structured responses.

---

## Instructor Notes

- Show side-by-side: same retrieval, different prompts, different answers
- Emphasize experimentation: "There's no one right prompt"
- Warn against over-engineering: "Start simple, iterate"
- Demo: Clear chat history before testing new prompts

---

## Slides

> **TODO:** Create presentation slides for this lecture

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
- [Prompting for RAG](https://www.anthropic.com/news/contextual-retrieval)
