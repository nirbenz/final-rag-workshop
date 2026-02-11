# Lecture 05: Advanced Chunking and Prompting Strategies

> **Phase:** Optional - Advanced Topics (with optional hands-on before wrap-up)

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand why fixed-window chunking is problematic
2. Know data-driven approaches to determine chunk boundaries
3. Understand time-based segmentation
4. Be aware of advanced retrieval models (ColBERT)
5. Know advanced prompting patterns (confidence calibration, few-shot, combining techniques)

---

## Part A: Advanced Chunking

### 1. The Fixed-Window Problem

**What Goes Wrong:**

```
Actual Conversation:                    Fixed-Window Chunks:

Morning topic: Planning trip            [Chunk 1: Trip planning + random]
  "Let's go to Paris"                   [Chunk 2: Random + Dinner plans]
  "I found cheap flights"               [Chunk 3: Dinner + Project work]
  "Book for May?"

Evening topic: Dinner plans             Topics get SPLIT across chunks!
  "Where should we eat?"
  "How about Italian?"

Next day: Project work
  "Did you finish the report?"
  "Almost done"
```

*Fixed windows split topics mid-conversation and mix unrelated content in the same chunk.*

**Problems:**

1. **Cuts mid-conversation**: Topic split between chunks
2. **Mixes unrelated content**: Different topics in same chunk
3. **Overlap is a hack**: Creates redundancy, doesn't fix boundary issues

---

### 2. Data-Driven Chunking Insights

> **Slide guidance:** Each signal (time gaps, histogram, mean messages) is a separate slide.

**Key Question:** How do we find natural conversation boundaries?

**Signal 1: Time Gaps**

```
Messages over time:

|----|||--|||----|--||--|||---|  ...  |--|||-|
      ^         ^              ^
   Active    Active         6-hour
   period    period           gap
                           (new segment!)
```

**Insight:** Silence often indicates topic change.

**Signal 2: Time Gap Histogram**

```
Gap Distribution:
|
|  *
|  *  *
|  *  *  *
|  *  *  *  *
|  *  *  *  *  *        *
+--1m-5m-1h-6h-12h-24h-----> gap duration

Bimodal = natural breakpoints
```

*Histogram of time gaps between messages. Bimodal distribution reveals natural conversation breakpoints.*

**Signal 3: Mean Messages Before Silence**

```python
# Approximate "natural conversation length"
avg_messages_before_gap = total_messages / num_gaps_over_threshold
# Use this as chunk_length hint!
```

---

### 3. Segmentation Strategies

> **Slide guidance:** Each segmentation strategy (time-based, semantic, speaker-based) is a separate slide.

**Strategy 1: Time-Based Segmentation**

```
Rule: Start new segment when gap > N hours

Messages: [M1, M2, M3, ..., M10] [gap: 8 hours] [M11, M12, ..., M20]
                    |                                      |
             Segment 1                              Segment 2

Then chunk WITHIN segments (never cross boundaries)
```

**Parameters:**
- `time_gap_hours`: Threshold for new segment (try 4-8 hours)

**Strategy 2: Semantic Segmentation**

```
Rule: Start new segment when embedding similarity drops

Messages:  M1 -- M2 -- M3 -- M4 -- M5 -- M6 -- M7
Similarity:  0.9   0.85  0.3   0.88  0.82  0.4
                        ^                   ^
                   Topic change!       Topic change!

Segments: [M1-M3] [M4-M6] [M7...]
```

**Parameters:**
- `similarity_threshold`: Minimum similarity to stay in segment

**Strategy 3: Speaker-Based (Heuristic)**

```
Rule: New participant entering might indicate new context

Speakers: [John, Sarah, John, Sarah] [+Mike joins] [Mike, Sarah, John]
                                          ^
                                  Possible segment break
```

---

### 4. Hierarchical Chunking

**The Pattern:**

```
Step 1: Segment by time gaps
        [Segment A: 15 msgs] [Segment B: 8 msgs] [Segment C: 22 msgs]

Step 2: Chunk within segments
        [A1: 6] [A2: 6] [A3: 6]   [B1: 8]   [C1: 6] [C2: 6] [C3: 6] [C4: 6]
        (with overlap)           (no split)  (with overlap)

Result: Chunks respect conversation boundaries!
```

*Two-level approach: segment by time gaps first, then chunk within each segment. Chunks never cross segment boundaries.*

**Benefits:**

- Topics stay together
- Overlap within segments only
- Variable segment sizes (content-aware)

---

### 5. More Advanced Strategies (Lecture Only)

> **Slide guidance:** Each strategy (contextual chunking, day-level aggregation, query expansion) is a separate slide. These are mention-only -- one diagram each, no deep-dive.

**Contextual Chunking (Anthropic Pattern):**

```
1. Generate conversation summary once
2. Prepend summary to each chunk for embedding
3. Store original chunk text only

Embedding: "Summary: Friends planning a trip... | John: Let's go to Paris"
Storage:   "John: Let's go to Paris"
```

**Benefit:** Chunks understand global context for better retrieval.

**Day-Level Aggregation:**

```
After top-k retrieval:
1. Count which day has most matching chunks
2. Return entire day instead of scattered chunks
3. Better coherence, more context
```

**Query Expansion:**

```
Original: "What about the trip?"
Expanded: "vacation travel holiday journey Paris flights booking"
```

---

## Part B: Advanced Retrieval Models

### 6. ColBERT and Late Interaction

> **Slide guidance:** The retrieval spectrum is one slide; the ColBERT token-level scoring example is a second slide.

**The Spectrum of Retrieval Models:**

```
Sparse (BM25)         Bi-Encoder          Cross-Encoder       ColBERT
    |                     |                    |                  |
 Keyword matching    Sentence vectors     Joint encoding    Token-level
 Very fast           Fast, pre-compute    Slow, accurate    Best of both
 No semantics        Good semantics       Best semantics    Great semantics
                                                            + lexical
```

*Spectrum of retrieval models from pure keyword (BM25) to token-level interaction (ColBERT).*

**ColBERT (Contextualized Late Interaction over BERT):**

Instead of compressing each document into a single vector, ColBERT keeps
per-token embeddings and computes similarity at the token level:

```
Query:    "Paris trip"        ->  [vec_Paris, vec_trip]
Document: "vacation in Paris" ->  [vec_vacation, vec_in, vec_Paris]

Score = sum of max similarities:
  vec_Paris  <-> max(vec_vacation, vec_in, vec_Paris) = vec_Paris  (1.0)
  vec_trip   <-> max(vec_vacation, vec_in, vec_Paris) = vec_vacation (0.8)
  Total = 1.8
```

This gives ColBERT both semantic understanding AND lexical precision -- addressing
the exact blind spot we discussed in Lecture 04.

**RAGatouille** makes ColBERT easy to use:

```python
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
results = RAG.search(query="Paris trip", k=10)
```

Qdrant also supports ColBERT-style late-interaction reranking natively in its
multi-stage retrieval pipeline.

Worth exploring post-workshop for production systems where retrieval quality is critical.

---

## Part C: Advanced Prompting Strategies

### 7. Confidence Calibration

> **Slide guidance:** The prompt template is one slide; the example output with "Why This Matters" is a second slide.

LLMs are confidently wrong. They don't naturally express uncertainty.

**Add to your system prompt:**

```
When responding, rate your confidence:
- HIGH: The answer is directly and clearly stated in the context
- MEDIUM: The answer requires reasonable inference from the context
- LOW: The context is ambiguous, incomplete, or potentially outdated
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

### 8. Few-Shot Prompting

> **Slide guidance:** The few-shot code example is one slide; the key principles and token budget table are a second slide.

Instructions alone don't guarantee consistent output. Show, don't tell:

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

**Trade-off:** More examples = more consistent format, but less room for context.

| Component | Typical Tokens |
|-----------|----------------|
| System instructions | 100-200 |
| Few-shot examples | 200-400 |
| Retrieved context | 1000-4000 |
| User query | 20-100 |

---

### 9. Combining Techniques: The Full Prompt

> **Slide guidance:** The stacked system prompt is one slide; the context injection mechanism and "key insight" takeaway are a second slide.

**Stacking CoT + Confidence + Few-Shot + Pydantic:**

The prompting module exports a single function:

```python
def get_system_prompt() -> str:
    """
    Returns the system prompt template with {context} placeholder.
    This is displayed in the UI sidebar for inspection.
    """
    return """You are a helpful assistant that answers questions
about WhatsApp conversations.

<context>
{context}
</context>

When responding, you must provide:
1. query_understanding: Restate what the user is asking
2. reasoning_steps: Your step-by-step analysis
3. context_used: List of exact quotes from the context
4. output: Your final answer to the question
5. confidence: "high", "medium", or "low" """
```

**Context Injection:**

The `{context}` placeholder is filled in by `llm.py` at runtime:

```python
# In llm.py - get_pydantic_agent()
@agent.instructions
def system_prompt_input(ctx: RunContext[Any]) -> str:
    context = ctx.deps.get("context", "No context available.")
    return get_system_prompt().format(context=context)
```

This keeps the prompting module focused on prompt design while `llm.py` handles
the integration with pydantic-ai.

**Key Insight:** These techniques stack. CoT gives reasoning, confidence gives
calibration, few-shot gives format consistency, Pydantic gives structure.
More structure = more tokens = higher cost. Choose what you need.

---

## Instructor Notes

- Show actual time-gap histogram from sample data if possible
- Draw the segmentation visually
- Emphasize: "Let the data tell you where to split"
- ColBERT: mention only, don't demo (post-workshop exploration)
- Advanced prompting: demo CoT vs non-CoT if not already done in Phase 2
- These techniques stack -- use the "combining" section to show how they fit together

---

## Code References

**Chunking:**
- Exercise: `src/workshop/rag/exercises/segmenting.py`
- Solution: `src/workshop/rag/solutions/segmenting.py`
- Chunker: `src/workshop/rag/chunkers/segmenting.py`
- Semantic chunker: `src/workshop/rag/chunkers/semantic.py`
- Contextual chunker: `src/workshop/rag/chunkers/contextual.py`

**Prompting:**
- Exercise: `src/workshop/rag/exercises/prompting.py`
- Structured types: `src/workshop/structured_types.py`
- LLM integration: `src/workshop/llm.py`

---

## Further Reading

- [Chunking Strategies for RAG](https://arxiv.org/abs/2312.06648)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex Chunking Guide](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
- [ColBERT: Efficient and Effective Passage Search (Khattab & Zaharia)](https://arxiv.org/abs/2004.12832)
- [RAGatouille Library](https://github.com/bclavie/RAGatouille)
- [Chain-of-Thought Prompting (Wei et al.)](https://arxiv.org/abs/2201.11903)
- [Language Models are Few-Shot Learners (GPT-3 paper)](https://arxiv.org/abs/2005.14165)
