# Lecture 01: The Baseline System

> **Duration:** 5 minutes
> **Phase:** Phase 1 - Baseline Exploration

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand how `MessageCountChunker` works
2. Understand why `NaiveContextEngine` is problematic
3. Be ready to explore and build intuition

---

## Outline

### 1. MessageCountChunker (2 minutes)

**How it works:**

```
Messages: [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10]

chunk_length = 4
chunk_overlap = 2
stride = chunk_length - chunk_overlap = 2

Chunk 1: [M1, M2, M3, M4]     indices 0-3
Chunk 2: [M3, M4, M5, M6]     indices 2-5  (overlaps M3, M4)
Chunk 3: [M5, M6, M7, M8]     indices 4-7  (overlaps M5, M6)
Chunk 4: [M7, M8, M9, M10]    indices 6-9  (overlaps M7, M8)
```

**Parameters:**

| Parameter | Purpose | Tradeoff |
|-----------|---------|----------|
| `chunk_length` | Messages per chunk | Small = loses context, Large = dilutes relevance |
| `chunk_overlap` | Shared messages between chunks | More = redundancy, Less = boundary issues |

**Key Insight:** This is a simple sliding window. No intelligence about content.

---

### 2. NaiveContextEngine (2 minutes)

**How it works:**

```python
def get_relevant_context(self, query: str, top_k: int):
    # Ignores the query entirely!
    return self.all_chunks[:top_k]
```

**Why this is terrible:**

- O(n) context: All chunks go to LLM
- Irrelevant information dilutes the prompt
- Token limits hit quickly with large chats
- LLM has to find the needle in the haystack

**But it works!** And shows us what we're improving.

---

### 3. What to Observe During Exploration (1 minute)

**Questions to answer:**

1. How do different chunk sizes feel?
2. Where do chunks cut mid-conversation?
3. What information gets lost with small chunks?
4. How does the LLM perform with irrelevant context?

**The "aha moment":** Seeing ALL chunks sent to the LLM regardless of query.

---

## Instructor Notes

- Keep this lecture short - the exploration is the learning
- Draw the sliding window on a whiteboard
- Emphasize that this is intentionally bad
- Prepare for "why would anyone do this?" - explain it's a teaching baseline

---

## Slides

> **TODO:** Create presentation slides for this lecture

---

## Code References

- Chunker: `src/workshop/rag/chunkers/message_count.py`
- Engine: `src/workshop/rag/engines/naive.py`
- Config: `src/nicegui_app/workshop_config.py`
