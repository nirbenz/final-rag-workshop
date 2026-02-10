# Lecture 01: The Baseline System

> **Phase:** Phase 1 - Baseline Exploration

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand how `MessageCountChunker` works
2. Understand why `NaiveContextEngine` is problematic
3. Be ready to explore and build intuition

---

## Outline

### 1. MessageCountChunker

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

*Sliding window with overlap: consecutive chunks share messages at their boundaries (e.g., M3-M4 appear in both Chunk 1 and 2).*

**Parameters:**

| Parameter | Purpose | Tradeoff |
|-----------|---------|----------|
| `chunk_length` | Messages per chunk | Small = loses context, Large = dilutes relevance |
| `chunk_overlap` | Shared messages between chunks | More = redundancy, Less = boundary issues |

**Key Insight:** This is a simple sliding window. No intelligence about content.

---

### 2. NaiveContextEngine

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

### 3. What to Observe During Exploration

**Questions to answer:**

1. How do different chunk sizes feel?
2. Where do chunks cut mid-conversation?
3. What information gets lost with small chunks?
4. How does the LLM perform with irrelevant context?

**The "aha moment":** Seeing ALL chunks sent to the LLM regardless of query.

---

## Where We Are

```
              *** ACTIVE ***
┌──────────┐                    ┌─────┐
│ CHUNKING │ ──── all chunks ──>│ LLM │ ──> Response
└──────────┘                    └─────┘
  MessageCountChunker            NaiveContextEngine
  (sliding window)               (no filtering at all)
```

*Phase 1 architecture: all chunks go directly to the LLM with no retrieval or filtering.*

Everything else in the pipeline is absent. This is the starting point we improve from.

---

## Workshop Mechanics: Before You Start

**Two files control the workshop progression:**

**1. Phase selection** -- determines which chunker and engine are active:

```python
# src/nicegui_app/workshop_config.py
PHASE = 1    # Change to 1, 2, 3, or 4
```

**2. Exercise toggles** -- switch between your code and reference solutions:

```python
# src/workshop/exercise_toggles.py
USE_PROMPTING_SOLUTION = False   # Phase 2: prompt design
USE_SIMILARITY_SOLUTION = False  # Phase 3: cosine similarity + top-k
USE_RERANKING_SOLUTION = False   # Phase 4: re-ranking strategy
USE_SEGMENTING_SOLUTION = False  # Optional: time-based segmentation
```

Set a toggle to `False` when you start an exercise (to use your code).
Set it back to `True` if you get stuck (to fall back to the reference solution).

**After any change:** Restart the app with `uv run python -m nicegui_app.main`

---

## What's Next: Phase 1 Hands-On

Explore the baseline system. No coding yet -- just observe:

1. Load your WhatsApp chat
2. Ask questions, adjust chunk size/overlap
3. Notice that ALL chunks are sent to the LLM regardless of query

**Next up:** Phase 1 exploration, then Lecture 02 on prompt engineering.

---

## Instructor Notes

- Keep this lecture short -- the exploration is the learning
- Draw the sliding window on a whiteboard
- Emphasize that this is intentionally bad
- Prepare for "why would anyone do this?" -- explain it's a teaching baseline
- Make sure everyone understands the two config files before starting hands-on

---

## Code References

- Chunker: `src/workshop/rag/chunkers/message_count.py`
- Engine: `src/workshop/rag/engines/naive.py`
- Phase config: `src/nicegui_app/workshop_config.py`
- Exercise toggles: `src/workshop/exercise_toggles.py`
