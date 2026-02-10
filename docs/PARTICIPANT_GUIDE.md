# RAG Workshop - Participant Hands-On Guide

> **Prerequisites:** Python 3.12+, basic Python knowledge, familiarity with NumPy

Welcome to the RAG Workshop! In this hands-on session, you will build a complete Retrieval-Augmented Generation system from scratch, using WhatsApp chat exports as your data source. By the end, you will understand how chunking strategies, embedding-based retrieval, and prompt engineering work together to create intelligent Q&A systems over your own conversations.

---

## Table of Contents

1. [Phase 0: Setup](#phase-0-setup)
2. [Phase 1: Baseline Exploration](#phase-1-baseline-exploration)
3. [Phase 2: Prompt Engineering](#phase-2-prompt-engineering)
4. [Phase 3: Embedding-Based Retrieval](#phase-3-embedding-based-retrieval)
5. [Phase 4: Vector Database & Re-ranking](#phase-4-vector-database--re-ranking)
6. [Optional: Advanced Chunking Strategies](#optional-advanced-chunking-strategies)
7. [Appendix: Troubleshooting](#appendix-troubleshooting)

---

## Phase 0: Setup

> **Lecture Notes:** [00_introduction.md](lectures/00_introduction.md)

### 0.1 Environment Setup

First, clone the repository and install dependencies:

```bash
git clone <repository-url>
cd rag-workshop

# Install dependencies using uv (fast Python package manager)
uv sync

# (OR using pip, which can potentially be buggy)
python -m venv .venv
pip install -r requirements.txt
```

Verify the installation:

```bash
uv run python -c "from nicegui_app.main import *; print('Setup complete!')"

# (if using pip)
source .venv/bin/activate
python -c "from nicegui_app.main import *; print('Setup complete!')"
```

### 0.2 Export Your WhatsApp Chat

You will need a WhatsApp chat export to use as your data source. Follow these instructions based on your device:

#### Android

1. Open WhatsApp and navigate to the chat you want to export (preferably a large group chat)
2. Tap the three-dot menu (top right) > **More** > **Export chat**
3. **Important:** Select **Without Media** when prompted
4. Choose how to share (email to yourself, save to Drive, etc.)
5. You will receive a `.txt` file

![Android Export](https://whatstk.readthedocs.io/en/latest/_images/chat-export-android9-wp2.20.123.gif)

#### iPhone (iOS)

1. Open WhatsApp and navigate to the chat you want to export
2. Tap the contact/group name at the top
3. Scroll down and tap **Export Chat**
4. **Important:** Select **Without Media**
5. Choose how to share (AirDrop, email, save to Files, etc.)
6. You will receive a `.zip` file - **unzip it** to get the `.txt` file

![iOS Export](https://whatstk.readthedocs.io/en/latest/_images/chat-export-ios17-wp24.5.75.gif)

#### Place Your Chat File

Copy the exported `.txt` file to the [`chats/`](../chats/) directory:

```bash
cp ~/Downloads/WhatsApp\ Chat\ with\ MyGroup.txt chats/
```

> **Note:** Sample chats are provided in [`chats/`](../chats/) if you do not have your own export.

### 0.3 Configure API Keys

Create a `.env` file in the project root with your API keys:

```bash
# At least one of these is required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# For Vertex AI (optional)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

### 0.4 Launch the Workshop UI

```bash
uv run python -m nicegui_app.main
```

or

```bash
python -m nicegui_app.main
```

Open your browser to `http://localhost:8080`. You should see the workshop interface.

---

## Phase 1: Baseline Exploration

> **Lecture Notes:** [01_baseline_system.md](lectures/01_baseline_system.md)
>
> **Goal:** Understand the baseline system and build intuition for RAG pipelines.

In this phase, you will explore the existing baseline implementation without writing any code. The goal is to understand how chunking and retrieval work together.

### 1.1 Understanding the Baseline

The baseline system uses two simple components:

| Component   | Class                 | Behavior                                             |
| ----------- | --------------------- | ---------------------------------------------------- |
| **Chunker** | `MessageCountChunker` | Splits messages into fixed-size windows with overlap |
| **Engine**  | `NaiveContextEngine`  | Returns ALL chunks regardless of query               |

This is intentionally naive - it shows you what we are improving!

### 1.2 Hands-On: Load Your Chat

1. In the UI sidebar, click **Load Chat**
2. Select your WhatsApp export file from [`chats/`](../chats/)
3. Observe the messages appearing in the context panel

### 1.3 Experiment: Adjust Chunking Parameters

Use the sidebar sliders to adjust chunker parameters:

| Parameter       | Description            | Try These Values |
| --------------- | ---------------------- | ---------------- |
| `chunk_length`  | Messages per chunk     | 3, 6, 10, 20     |
| `chunk_overlap` | Overlap between chunks | 0, 2, 4          |

**Observe:**
- How do chunk boundaries change in the visualization?
- What happens to context when chunks are too small? Too large?

### 1.4 Experiment: Ask Questions

In the chat panel, ask questions about your conversation:

```
What were we talking about on [date]?
Who suggested [topic]?
What did [person] say about [subject]?
```

**Key Observation:** Notice that ALL chunks are sent to the LLM regardless of your query. This is the problem we will solve in Phase 3!

### 1.5 Reflection Questions

Before moving on, consider:

1. What chunk size feels "natural" for a conversation?
2. When chunks are too small, what information gets lost?
3. When chunks are too large, what problems emerge?
4. Why is sending ALL chunks to the LLM problematic?

---

## Phase 2: Prompt Engineering

> **Lecture Notes:** [02_prompt_engineering.md](lectures/02_prompt_engineering.md)
>
> **Goal:** Design effective prompts that ground LLM responses in retrieved context.

Great retrieval means nothing with bad prompts. Before we improve retrieval, let's optimize the "G" in RAG -- the generation step. This gives you a foundation for evaluating retrieval quality in later phases.

### 2.1 Advance to Phase 2

```python
# In src/nicegui_app/workshop_config.py
PHASE = 2
```

Restart the app. The engine is still `NaiveContextEngine` (all chunks sent to LLM), which is fine -- we are focusing on how the LLM uses context, not on retrieval quality yet.

### 2.2 Concept Review

**Prompt Components for RAG:**

1. **System Prompt**: Sets the assistant's role and behavior
2. **Context Formatting**: How retrieved chunks are presented
3. **Output Instructions**: Constraints on the response format

### 2.3 Exercise: Design Your Prompts

Open [`src/workshop/rag/exercises/prompting.py`](../src/workshop/rag/exercises/prompting.py):

```python
def get_system_prompt() -> str:
    """
    Design a system prompt template for the RAG assistant.

    The prompt should:
    1. Define the assistant's role (helpful, factual, grounded in context)
    2. Explain how to use the <context>...</context> block
    3. Specify the output format to match RAGResponse:
       - query_understanding: Restate the question
       - reasoning_steps: List of {thought, observation} steps
       - context_used: List of exact quotes from context
       - output: The final answer
       - confidence: high/medium/low
    4. Handle cases where context is insufficient

    The {context} placeholder will be replaced with retrieved chunks by llm.py.

    Returns:
        System prompt string with {context} placeholder
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_system_prompt")
```

**How Context Injection Works:**

Your `get_system_prompt()` returns a template with `{context}` placeholder.
The [`llm.py`](../src/workshop/llm.py) module handles injecting the actual context at runtime:

```python
# In llm.py - this is done for you
return get_system_prompt().format(context=context)
```

### 2.4 Enable Your Implementation

```python
# In src/workshop/exercise_toggles.py
USE_PROMPTING_SOLUTION = False  # Use your prompts (True = use solution)
```

Restart the app and test with a fresh chat history.

### 2.5 Experiment with Prompts

Try different prompt variations and observe how responses change:

| Variation       | Try This                                                     |
| --------------- | ------------------------------------------------------------ |
| **Persona**     | "You are a friend who knows this person well..."             |
| **Constraints** | "Never mention exact dates"                                  |
| **Format**      | Use XML tags vs numbered list                                |
| **Length**      | "Answer in one sentence" vs "Provide a detailed explanation" |

**Key Insight:** Same retrieval + different prompts = very different answers!

### 2.6 Stretch Goals: Advanced Prompting Techniques

For participants who finish early, explore these advanced patterns:

#### Chain-of-Thought (CoT) Prompting

Make the LLM reason step-by-step before answering:

```python
def get_system_prompt() -> str:
    return """You are a helpful assistant answering questions about conversations.

Before answering, follow these steps:
1. Identify which chunks are most relevant to the question
2. Extract the key information from those chunks
3. Synthesize a clear answer based on the evidence

Think through your reasoning before providing the final answer."""
```

#### Confidence Calibration

Ask the LLM to express uncertainty:

```python
def get_output_instructions() -> str:
    return """After your answer, rate your confidence:
- HIGH: Answer is directly stated in context
- MEDIUM: Answer requires some inference
- LOW: Context is ambiguous or incomplete

Format: [Answer] (Confidence: HIGH/MEDIUM/LOW)"""
```

#### Few-Shot Prompting

Provide example Q&A pairs to guide the format:

```python
def get_system_prompt() -> str:
    return """You answer questions about WhatsApp conversations.

Example:
Q: When did they plan to meet?
A: They planned to meet on Friday at 3pm [1]. Sarah confirmed the time works for her.

Q: Who suggested the restaurant?
A: John suggested Italian food [2], and the group agreed.

Now answer the user's question using the same format."""
```

#### Structured Output with Custom Pydantic Models

The workshop uses `RAGResponse` for structured chain-of-thought responses:

```python
# Defined in src/workshop/structured_types.py
from pydantic import BaseModel, Field
from typing import List, Literal


class ReasoningStep(BaseModel):
    thought: str     # What you are checking
    observation: str # What you found


class RAGResponse(BaseModel):
    query_understanding: str           # Restate the question
    reasoning_steps: List[ReasoningStep]  # Step-by-step analysis
    context_used: List[str]            # Exact quotes from context
    output: str                        # Final answer
    confidence: Literal["high", "medium", "low"]
```

The `extract_llm_response()` function in [`state.py`](../src/nicegui_app/state.py) handles any Pydantic model
with an `output` field - the `output` becomes the displayed answer, and the full
response is available in `raw_output` for debugging.

`RAGResponse` also provides a `__str__()` method that formats the full response
as readable markdown, shown in the "Raw Output" section of the UI.

**Key Insight:** Structured outputs make LLM responses predictable and parseable.
The chain-of-thought fields (`reasoning_steps`, `context_used`) help with debugging
and traceability.

### 2.7 Stuck? Use the Solution

```python
# In src/workshop/exercise_toggles.py
USE_PROMPTING_SOLUTION = True  # Fall back to reference solution
```

---

## Phase 3: Embedding-Based Retrieval

> **Lecture Notes:** [03_embeddings_similarity.md](lectures/03_embeddings_similarity.md)
>
> **Goal:** Implement semantic search using vector embeddings.

Now we will make retrieval intelligent! Instead of returning all chunks, we will return only the most relevant ones based on semantic similarity.

### 3.1 Advance to Phase 3

```python
# In src/nicegui_app/workshop_config.py
PHASE = 3
```

Restart the app. The engine is now `SimilarityContextEngine`.

### 3.2 Concept Review

**Embeddings** convert text into dense vectors where similar meanings map to nearby points in vector space.

**Cosine Similarity** measures the angle between two vectors:

```
cos(theta) = (A . B) / (||A|| * ||B||)
```

- Returns values in `[-1, 1]`
- `1.0` = identical direction (most similar)
- `0.0` = perpendicular (unrelated)
- `-1.0` = opposite direction (most dissimilar)

### 3.3 Exercise: Implement Similarity Functions

Open [`src/workshop/rag/exercises/similarity.py`](../src/workshop/rag/exercises/similarity.py):

```python
import numpy as np
from numpy.typing import NDArray


def cosine_similarity(
    query_embedding: NDArray[np.float32],
    chunk_embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute cosine similarity between a query and multiple chunks.

    Args:
        query_embedding: Shape (embedding_dim,) - the query vector
        chunk_embeddings: Shape (num_chunks, embedding_dim) - chunk vectors

    Returns:
        Shape (num_chunks,) - similarity scores in [-1, 1]

    Hints:
        1. Normalize both query and chunk vectors (divide by L2 norm)
        2. Compute dot product between normalized vectors
        3. Handle edge case: zero-magnitude vectors (add small epsilon)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement cosine_similarity")


def get_top_k(
    similarities: NDArray[np.float32],
    threshold: float,
    k: int,
) -> NDArray[np.intp]:
    """
    Return indices of top-k chunks above the similarity threshold.

    Args:
        similarities: Shape (num_chunks,) - similarity scores
        threshold: Minimum similarity to include
        k: Maximum number of results

    Returns:
        Indices of top-k chunks, sorted by similarity (highest first)

    Hints:
        1. Filter: keep only indices where similarity >= threshold
        2. Sort: by similarity in descending order
        3. Truncate: return at most k indices
        4. np.argsort returns ascending order - you need descending!
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement get_top_k")
```

### 3.4 Enable Your Implementation

```python
# In src/workshop/exercise_toggles.py
USE_SIMILARITY_SOLUTION = False  # Use your code (True = use solution)
```

### 3.5 Test Your Implementation

Run the tests to verify your implementation:

```bash
uv run pytest tests/test_similarity_context_engine.py -v
```

**Common Issues:**

| Problem          | Solution                                            |
| ---------------- | --------------------------------------------------- |
| Division by zero | Add epsilon: `norm = max(np.linalg.norm(v), 1e-10)` |
| Wrong sort order | Use `np.argsort(-similarities)` or `[::-1]`         |
| Shape mismatch   | Check broadcasting: query is 1D, chunks are 2D      |

### 3.6 Experiment: Compare Engines

Ask the same question with different phases:

1. Set `PHASE = 1` - observe ALL chunks used (NaiveContextEngine)
2. Set `PHASE = 3` - observe RELEVANT chunks used (SimilarityContextEngine)

**Discussion:**
- What similarity threshold works well for your data?
- What happens with very short or vague queries?
- Why is this still O(n) complexity? (Hint: we compare against ALL chunks)

### 3.7 Stuck? Use the Solution

```python
# In src/workshop/exercise_toggles.py
USE_SIMILARITY_SOLUTION = True  # Fall back to reference solution
```

---

## Phase 4: Vector Database & Re-ranking

> **Lecture Notes:** [04_vector_databases.md](lectures/04_vector_databases.md)
>
> **Goal:** Scale retrieval with ANN search and add re-ranking for precision.

The similarity engine compares against ALL chunks -- O(n) per query. Vector databases use Approximate Nearest Neighbor (ANN) algorithms to achieve sub-linear search. We also add a re-ranking stage to combine semantic and lexical relevance.

### 4.1 Advance to Phase 4

```python
# In src/nicegui_app/workshop_config.py
PHASE = 4
```

Restart the app. The engine is now `RAGContextEngine` (Qdrant ANN search).

### 4.2 Concept Review

**Two-Stage Retrieval Pattern:**

```
Query -> [ANN Search: 50 candidates] -> [Re-ranker: top 10] -> LLM
         Fast but approximate           More accurate
```

**Why Two Stages?**
- Stage 1 (ANN): Trade accuracy for speed -- retrieve rough candidates
- Stage 2 (Re-rank): Apply scoring to the small candidate set

### 4.3 Exercise: Implement Re-ranking

Open [`src/workshop/rag/exercises/reranking.py`](../src/workshop/rag/exercises/reranking.py):

The baseline just truncates to top_k. Replace it with something smarter.

**Implementation options (easiest to hardest):**

| Option | Approach | Difficulty |
|--------|----------|------------|
| 1 | **Recency**: sort by `chunk.metadata["end_time"]` (newest first) | Easy |
| 2 | **Keyword overlap**: count query words that appear in chunk text | Easy |
| 3 | **Combined**: multiply keyword score by recency boost | Medium |
| 4 | **BM25**: TF-IDF with length normalization (reference solution) | Medium |
| 5 | **Cross-encoder**: sentence-transformers joint scoring | Advanced |

**Recency example** (good starting point for chat data):

```python
def rerank(query, chunks, top_k=5):
    if not chunks:
        return []
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.metadata.get("end_time", ""),
        reverse=True,
    )
    return sorted_chunks[:top_k]
```

**Keyword overlap example:**

```python
def rerank(query, chunks, top_k=5):
    if not chunks:
        return []
    query_terms = set(query.lower().split())

    def score(chunk):
        chunk_terms = set(chunk.text.lower().split())
        return len(query_terms & chunk_terms)

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:top_k]
```

### 4.4 Enable Your Implementation

```python
# In src/workshop/exercise_toggles.py
USE_RERANKING_SOLUTION = False  # Use your code (True = use BM25 solution)
```

### 4.5 Experiment: Compare Results

Ask the same 3 questions across multiple phases and compare:

| Question | Phase 1 (Naive) | Phase 3 (Similarity) | Phase 4 (ANN + Re-rank) |
|----------|-----------------|---------------------|--------------------------|
| ...      | All chunks sent | Relevant chunks     | Relevant + re-ranked     |

For each, note:
- How many of the returned chunks actually help answer the question?
- Did the engine miss any chunks you know are relevant?
- What is the rough precision and recall?

This manual process is exactly what you would automate in production with
frameworks like RAGAS.

### 4.6 Stuck? Use the Solution

```python
# In src/workshop/exercise_toggles.py
USE_RERANKING_SOLUTION = True  # Fall back to BM25 reference solution
```

---

## Optional: Advanced Chunking Strategies

> **Lecture Notes:** [05_advanced_topics.md](lectures/05_advanced_topics.md)
>
> **Goal:** Implement data-aware chunking that respects conversation boundaries.
>
> This section is a take-home assignment. Work through it after the workshop at your own pace.

### The Problem with Fixed Windows

Fixed-window chunking has fundamental problems:

- Cuts conversations mid-topic
- No respect for natural boundaries (time gaps, speaker changes)
- Overlap is a hack, not a solution

**Better Approach:** Segment first, then chunk within segments.

### Exercise: Time-Based Segmentation

Open [`src/workshop/rag/exercises/segmenting.py`](../src/workshop/rag/exercises/segmenting.py) and implement:

1. `segment_by_time_gaps()` -- split messages into segments when the gap between consecutive messages exceeds a threshold
2. `chunk_segments()` -- apply sliding-window chunking within each segment (chunks should NOT cross segment boundaries)

### Test Your Implementation

```bash
uv run pytest tests/test_segmenting_chunker.py -v
```

### Enable Segmenting Chunker

Uncomment the segmenting chunker section in [`src/nicegui_app/workshop_config.py`](../src/nicegui_app/workshop_config.py):

```python
from workshop.rag.chunkers import SegmentingChunker
CHUNKER_CLASS = SegmentingChunker
ENGINE_CLASS = RAGContextEngine

CHUNKER_DEFAULTS = {
    "time_gap_hours": 6.0,
    "chunk_length": 6,
    "chunk_overlap": 2,
}
```

### Extension: Sentence Boundary Chunker

For an additional challenge, explore [`src/workshop/rag/chunkers/sentence_boundary.py`](../src/workshop/rag/chunkers/sentence_boundary.py):

- Chunk by token count instead of message count
- Respect sentence boundaries (no mid-sentence cuts)
- Use `tiktoken` for accurate token counting

---

## Wrap-Up: Key Takeaways

### The RAG Pipeline

```
┌──────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐
│ Chunking │-->│ Embedding │-->│  Indexing  │-->│ Retrieval │-->│ Re-ranking │-->│ Prompting │--> LLM
└──────────┘   └───────────┘   └───────────┘   └───────────┘   └────────────┘   └───────────┘
     |              |              |                 |               |                |
  Affects        Creates        Enables           Finds          Refines          Formats
  what can       semantic       fast ANN        candidates       accuracy         context
  be found       vectors        search
```

### Tradeoffs Everywhere

| Decision             | Too Little           | Too Much          |
| -------------------- | -------------------- | ----------------- |
| Chunk size           | Loses context        | Loses precision   |
| Similarity threshold | Misses relevant      | Includes noise    |
| Re-rank candidates   | May miss best        | Slower processing |
| Prompt constraints   | Unpredictable output | Overly rigid      |

### What We Did Not Cover

- Fine-tuning embedding models for your domain
- Evaluation metrics automation (RAGAS, faithfulness scores)
- Query expansion and HyDE
- Production concerns (caching, rate limiting, cost)

---

## Appendix: Troubleshooting

### Setup Issues

| Problem            | Solution                                              |
| ------------------ | ----------------------------------------------------- |
| `uv sync` fails    | Check Python version: `python --version` (need 3.12+) |
| App will not start | Check `.env` has API keys                             |
| Port 8080 in use   | Kill existing process or change port                  |

### Runtime Issues

| Problem             | Solution                                         |
| ------------------- | ------------------------------------------------ |
| "No API key" error  | Check `.env` file and restart                    |
| Embeddings fail     | Verify network connectivity and API key          |
| LLM returns garbage | Check if context is being retrieved (look at UI) |
| Retrieval empty     | Lower similarity threshold or check indexing     |

### Exercise Issues

| Problem                  | Solution                                            |
| ------------------------ | --------------------------------------------------- |
| NumPy broadcasting error | Check array shapes: query is 1D, chunks is 2D       |
| Division by zero         | Add epsilon to normalization                        |
| Wrong sort order         | `np.argsort` returns ascending, use `-` or `[::-1]` |
| Datetime errors          | Use `timedelta` for comparisons, not raw numbers    |

### Getting Unstuck

Each exercise has a reference solution. Enable it by setting the toggle in [`src/workshop/exercise_toggles.py`](../src/workshop/exercise_toggles.py):

| Exercise   | Toggle Variable            |
| ---------- | -------------------------- |
| Prompting  | `USE_PROMPTING_SOLUTION`   |
| Similarity | `USE_SIMILARITY_SOLUTION`  |
| Re-ranking | `USE_RERANKING_SOLUTION`   |
| Segmenting | `USE_SEGMENTING_SOLUTION`  |

All toggles are centralized in `exercise_toggles.py` to avoid circular imports.

---

## Next Steps

After the workshop, try:

1. **Your Own Data:** Export and analyze your personal WhatsApp chats
2. **Advanced Chunking:** Work through the optional segmenting exercise
3. **Better Re-ranking:** Implement BM25 or cross-encoder re-ranking
4. **Evaluation:** Add RAGAS metrics to measure retrieval quality
5. **Production:** Deploy to cloud with managed vector database

**Resources:**
- [Chunking Strategies Survey](https://arxiv.org/abs/2312.06648)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Qdrant Cloud](https://qdrant.tech/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
- [Cohere Rerank API](https://cohere.com/rerank)

---

*Happy RAG-ing!*
