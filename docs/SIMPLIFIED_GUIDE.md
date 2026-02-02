# RAG Workshop - Quick Reference

## How It Works

1. Set `PHASE` in [`src/nicegui_app/workshop_config.py`](../src/nicegui_app/workshop_config.py) to advance
2. Set toggles in [`src/workshop/exercise_toggles.py`](../src/workshop/exercise_toggles.py) to switch between your code and solutions
3. Restart the app after changes: `uv run python -m nicegui_app.main`

---

## Phase 1: Baseline Exploration

**Set:** `PHASE = 1`

**What you learn:** How a naive RAG system works. The system splits your chat into chunks and sends ALL of them to the LLM, regardless of what you ask.

**What you see in the GUI:** Every chunk highlighted as "used" context. Long, slow responses because the LLM processes everything.

**What you do:** No coding. Load your chat, ask questions, adjust chunk size/overlap sliders, and observe the problems. Why is sending everything to the LLM bad?

---

## Phase 2: Prompt Engineering

**Set:** `PHASE = 2`

**What you learn:** How prompt design affects LLM answers. Same naive engine (all chunks sent), but now you control how the LLM interprets and uses that context.

**What you see in the GUI:** Same retrieval as Phase 1, but responses change based on your prompt. Compare structured vs freeform answers.

**What you code:**

| File | Function |
|------|----------|
| [`src/workshop/rag/exercises/prompting.py`](../src/workshop/rag/exercises/prompting.py) | `get_system_prompt()` |

**Toggle:** `USE_PROMPTING_SOLUTION = False` to use yours, `True` to see the reference.

---

## Phase 3: Embedding-Based Retrieval

**Set:** `PHASE = 3`

**What you learn:** How embeddings turn text into vectors, and how cosine similarity finds relevant chunks instead of returning all of them.

**What you see in the GUI:** Only relevant chunks highlighted. Faster, more focused responses. Similarity scores shown per chunk.

**What you code:**

| File | Function |
|------|----------|
| [`src/workshop/rag/exercises/similarity.py`](../src/workshop/rag/exercises/similarity.py) | `cosine_similarity()` |
| [`src/workshop/rag/exercises/similarity.py`](../src/workshop/rag/exercises/similarity.py) | `get_top_k()` |

**Toggle:** `USE_SIMILARITY_SOLUTION = False` to use yours, `True` to see the reference.

**Test:** `uv run pytest tests/test_similarity_context_engine.py -v`

---

## Phase 4: Vector Database + Re-ranking

**Set:** `PHASE = 4`

**What you learn:** Why O(n) similarity search does not scale, how ANN (Approximate Nearest Neighbor) search works, and how a re-ranking stage improves precision.

**What you see in the GUI:** Same relevant-chunk retrieval as Phase 3, but powered by Qdrant (a real vector database) instead of brute-force NumPy. Re-ranking reorders candidates for better results.

**What you code:**

| File | Function |
|------|----------|
| [`src/workshop/rag/exercises/reranking.py`](../src/workshop/rag/exercises/reranking.py) | `rerank()` |

Pick any re-ranking strategy: keyword overlap (easy), BM25 (medium), or cross-encoder (advanced). The baseline just truncates -- replace it with something smarter.

**Toggle:** `USE_RERANKING_SOLUTION = False` to use yours, `True` to see the BM25 reference.

---

## All Toggles at a Glance

In [`src/workshop/exercise_toggles.py`](../src/workshop/exercise_toggles.py):

```python
USE_PROMPTING_SOLUTION = True    # Phase 2: prompt design
USE_SIMILARITY_SOLUTION = True   # Phase 3: cosine similarity + top-k
USE_RERANKING_SOLUTION = True    # Phase 4: re-ranking strategy
USE_SEGMENTING_SOLUTION = False  # Optional: time-based segmentation
```

Set to `False` when you start an exercise. Set back to `True` if you get stuck.
