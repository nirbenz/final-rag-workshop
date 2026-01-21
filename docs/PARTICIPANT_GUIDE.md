# RAG Workshop - Participant Hands-On Guide

> **Duration:** ~4 hours
> **Prerequisites:** Python 3.12+, basic Python knowledge, familiarity with NumPy

Welcome to the RAG Workshop! In this hands-on session, you will build a complete Retrieval-Augmented Generation system from scratch, using WhatsApp chat exports as your data source. By the end, you will understand how chunking strategies, embedding-based retrieval, and prompt engineering work together to create intelligent Q&A systems over your own conversations.

---

## Table of Contents

1. [Phase 0: Setup](#phase-0-setup-15-minutes)
2. [Phase 1: Baseline Exploration](#phase-1-baseline-exploration-35-minutes)
3. [Phase 2: Embedding-Based Retrieval](#phase-2-embedding-based-retrieval-55-minutes)
4. [Phase 3: Vector Database, Re-ranking & Prompting](#phase-3-vector-database-re-ranking--prompting-65-minutes)
5. [Phase 4: Advanced Chunking Strategies](#phase-4-advanced-chunking-strategies-45-minutes)
6. [Appendix: Troubleshooting](#appendix-troubleshooting)

---

## Phase 0: Setup (15 minutes)

> **Lecture Notes:** [00_introduction.md](lectures/00_introduction.md)

### 0.1 Environment Setup

First, clone the repository and install dependencies:

```bash
git clone <repository-url>
cd rag-workshop

# Install dependencies using uv (fast Python package manager)
uv sync

# (OR using pip, which can potentiall be buggy)
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

1. Open WhatsApp and navigate to the chat you want to export (preferrably a large group chat)
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

Copy the exported `.txt` file to the `chats/` directory:

```bash
cp ~/Downloads/WhatsApp\ Chat\ with\ MyGroup.txt chats/
```

> **Note:** Sample chats are provided in `chats/` if you do not have your own export.

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

Open your browser to `http://localhost:8080`. You should see the workshop interface.

---

## Phase 1: Baseline Exploration (35 minutes)

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
2. Select your WhatsApp export file from `chats/`
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

**Key Observation:** Notice that ALL chunks are sent to the LLM regardless of your query. This is the problem we will solve in Phase 2!

### 1.5 Reflection Questions

Before moving on, consider:

1. What chunk size feels "natural" for a conversation?
2. When chunks are too small, what information gets lost?
3. When chunks are too large, what problems emerge?
4. Why is sending ALL chunks to the LLM problematic?

---

## Phase 2: Embedding-Based Retrieval (55 minutes)

> **Lecture Notes:** [02_embeddings_similarity.md](lectures/02_embeddings_similarity.md)
>
> **Goal:** Implement semantic search using vector embeddings.

Now we will make retrieval intelligent! Instead of returning all chunks, we will return only the most relevant ones based on semantic similarity.

### 2.1 Concept Review

**Embeddings** convert text into dense vectors where similar meanings map to nearby points in vector space.

**Cosine Similarity** measures the angle between two vectors:

```
cos(theta) = (A . B) / (||A|| * ||B||)
```

- Returns values in `[-1, 1]`
- `1.0` = identical direction (most similar)
- `0.0` = perpendicular (unrelated)
- `-1.0` = opposite direction (most dissimilar)

### 2.2 Exercise: Implement Similarity Functions

Open `src/workshop/rag/exercises/similarity.py`:

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

### 2.3 Test Your Implementation

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

### 2.4 Enable Your Implementation

Once tests pass, switch to the similarity engine:

1. Open `src/nicegui_app/workshop_config.py`
2. Change the configuration:

```python
from workshop.rag.chunkers import MessageCountChunker
from workshop.rag.engines import SimilarityContextEngine

CHUNKER_CLASS = MessageCountChunker
ENGINE_CLASS = SimilarityContextEngine

ENGINE_KWARGS = {
    "similarity_threshold": 0.3,  # Adjust based on results
    "top_k": 10,
}
```

3. Restart the UI and test

### 2.5 Experiment: Compare Engines

Ask the same question with different engines:

1. Set `ENGINE_CLASS = NaiveContextEngine` - observe ALL chunks used
2. Set `ENGINE_CLASS = SimilarityContextEngine` - observe RELEVANT chunks used

**Discussion:**
- What similarity threshold works well for your data?
- What happens with very short or vague queries?
- Why is this still O(n) complexity? (Hint: we compare against ALL chunks)

### 2.6 Stuck? Use the Solution

If you cannot complete the exercise, enable the reference implementation:

```python
# In src/workshop/rag/engines/similarity.py
USE_SOLUTIONS = True  # Change from False to True
```

---

## Phase 3: Vector Database, Re-ranking & Prompting (65 minutes)

> **Lecture Notes:** [03_vector_databases.md](lectures/03_vector_databases.md) | [04_prompt_engineering.md](lectures/04_prompt_engineering.md)
>
> **Goal:** Scale retrieval with ANN search and optimize the generation step.

### Part A: Scaling with Vector Databases (25 minutes)

The similarity engine compares against ALL chunks - O(n) per query. Vector databases use Approximate Nearest Neighbor (ANN) algorithms to achieve sub-linear search.

#### 3A.1 Concept Review

**Two-Stage Retrieval Pattern:**

```
Query -> [ANN Search: 50-100 candidates] -> [Re-ranker: top 10] -> LLM
         Fast but approximate              Slow but accurate
```

**Why Two Stages?**
- Stage 1 (ANN): Trade accuracy for speed - retrieve rough candidates
- Stage 2 (Re-rank): Apply expensive scoring to small candidate set

#### 3A.2 Exercise: Implement Re-ranking

Open `src/workshop/rag/exercises/reranking.py`:

```python
from typing import List
from workshop.rag.engines.types import ChunkObject


def rerank(
    query: str,
    chunks: List[ChunkObject],
    top_k: int,
) -> List[ChunkObject]:
    """
    Re-rank retrieved chunks by relevance to the query.

    Args:
        query: The user's question
        chunks: Candidate chunks from ANN search
        top_k: Number of chunks to return

    Returns:
        Top-k chunks after re-ranking, most relevant first

    Implementation Options (choose one):
        1. Naive: Return first k chunks (baseline)
        2. Keyword overlap: Score by query term frequency in chunk
        3. Recency: Prefer more recent chunks
        4. Cross-encoder: Use sentence-transformers (advanced)
        5. LLM scoring: Ask model to rate relevance 1-10 (expensive)
    """
    # YOUR CODE HERE
    # Simplest implementation - return first k:
    return chunks[:top_k]
```

**Challenge Implementations:**

```python
# Option 2: Keyword overlap scoring
def rerank(query: str, chunks: List[ChunkObject], top_k: int) -> List[ChunkObject]:
    query_terms = set(query.lower().split())

    def score(chunk: ChunkObject) -> float:
        chunk_terms = set(chunk.text.lower().split())
        overlap = len(query_terms & chunk_terms)
        return overlap / len(query_terms) if query_terms else 0

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:top_k]
```

#### 3A.3 Enable Vector Database Engine

Update `src/nicegui_app/workshop_config.py`:

```python
from workshop.rag.chunkers import MessageCountChunker
from workshop.rag.engines import RAGContextEngine

CHUNKER_CLASS = MessageCountChunker
ENGINE_CLASS = RAGContextEngine

ENGINE_KWARGS = {
    "db_path": ".qdrant",
    "collection_name": "workshop_chunks",
    "top_k": 10,
    "rerank_candidates": 50,  # Retrieve 50, re-rank to 10
}
```

---

### Part B: Prompt Engineering (35 minutes)

Great retrieval means nothing with bad prompts. Now we optimize the "G" in RAG.

#### 3B.1 Concept Review

**Prompt Components for RAG:**

1. **System Prompt**: Sets the assistant's role and behavior
2. **Context Formatting**: How retrieved chunks are presented
3. **Output Instructions**: Constraints on the response format

#### 3B.2 Exercise: Design Your Prompts

Open `src/workshop/rag/exercises/prompting.py`:

```python
from typing import List


def get_system_prompt() -> str:
    """
    Design a system prompt for the RAG assistant.

    Include:
        - Role assignment ("You are...")
        - Task description ("Answer questions based on...")
        - Constraints ("Only use information from...")
        - Fallback behavior ("If the context doesn't help...")

    Returns:
        The system prompt string
    """
    # YOUR CODE HERE
    return """You are a helpful assistant that answers questions about conversations.

Use ONLY the provided context to answer. If the context does not contain
relevant information, say "I don't have enough information to answer that."

Be concise and cite specific parts of the conversation when possible."""


def format_context(chunks_text: List[str]) -> str:
    """
    Format retrieved chunks for inclusion in the prompt.

    Args:
        chunks_text: List of chunk text strings

    Returns:
        Formatted context string

    Options to try:
        - Numbered list (enables citations)
        - XML tags for clear boundaries
        - With/without metadata (timestamps, speakers)
    """
    # YOUR CODE HERE
    # Simple numbered format:
    formatted_chunks = []
    for i, text in enumerate(chunks_text, 1):
        formatted_chunks.append(f"[{i}] {text}")

    return "\n---\n".join(formatted_chunks)


def get_output_instructions() -> str:
    """
    Define how the LLM should structure its response.

    Options:
        - Length constraints ("2-3 sentences")
        - Citation requirements ("Quote relevant parts")
        - Confidence indicators ("High/Medium/Low confidence")
    """
    # YOUR CODE HERE
    return "Respond in 2-3 sentences. If quoting, use quotation marks."


def build_full_prompt(query: str, retrieved_chunks: List[str]) -> str:
    """
    Combine all components into the final prompt.

    Args:
        query: User's question
        retrieved_chunks: List of relevant chunk texts

    Returns:
        Complete prompt ready to send to LLM
    """
    system = get_system_prompt()
    context = format_context(retrieved_chunks)
    instructions = get_output_instructions()

    return f"""{system}

<context>
{context}
</context>

{instructions}

Question: {query}"""
```

#### 3B.3 Experiment with Prompts

Try different prompt variations and observe how responses change:

| Variation       | Try This                                                     |
| --------------- | ------------------------------------------------------------ |
| **Persona**     | "You are a friend who knows this person well..."             |
| **Constraints** | "Never mention exact dates"                                  |
| **Format**      | Use XML tags vs numbered list                                |
| **Length**      | "Answer in one sentence" vs "Provide a detailed explanation" |

**Key Insight:** Same retrieval + different prompts = very different answers!

#### 3B.4 Enable Your Prompts

```python
# In src/workshop/llm.py
USE_PROMPT_SOLUTIONS = False  # Use your prompts
```

Test with a fresh chat history (clear previous messages).

### 3B.5 Stretch Goals: Advanced Prompting Techniques

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

The workshop uses `RetrievalCoT` for structured responses. Try creating your own:

```python
# In src/workshop/structured_types.py
from pydantic import BaseModel, Field
from typing import List, Literal


class MyCustomResponse(BaseModel):
    """Custom structured response for RAG."""

    # Required: 'output' field is used for display
    output: str = Field(description="The answer to display")

    # Add your own fields:
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident are you in this answer?"
    )
    sources: List[int] = Field(
        description="Which chunk numbers support this answer?"
    )
    reasoning: str = Field(
        description="Brief explanation of your reasoning"
    )
```

Then use it in `workshop_config.py` or modify `main.py`:

```python
# The extract_llm_response function in state.py handles any BaseModel
# with an 'output' field - everything else becomes raw_output for debugging
```

**Key Insight:** Structured outputs make LLM responses predictable and parseable. The `output` field becomes the displayed answer; other fields are available in `raw_output` for debugging.

---

## Phase 4: Advanced Chunking Strategies (45 minutes)

> **Lecture Notes:** [05_chunking_strategies.md](lectures/05_chunking_strategies.md)
>
> **Goal:** Implement data-aware chunking that respects conversation boundaries.

### 4.1 The Problem with Fixed Windows

Fixed-window chunking has fundamental problems:

- Cuts conversations mid-topic
- No respect for natural boundaries (time gaps, speaker changes)
- Overlap is a hack, not a solution

**Better Approach:** Segment first, then chunk within segments.

### 4.2 Exercise: Time-Based Segmentation

Open `src/workshop/rag/exercises/segmenting.py`:

```python
from typing import List, Tuple
from datetime import timedelta
from workshop.chat import WhatsappMessage
from workshop.rag.engines.types import ChunkObject


def segment_by_time_gaps(
    messages: List[WhatsappMessage],
    time_gap_hours: float,
) -> List[List[WhatsappMessage]]:
    """
    Split messages into segments based on time gaps.

    A new segment starts when the gap between consecutive messages
    exceeds `time_gap_hours`.

    Args:
        messages: Chronologically sorted messages
        time_gap_hours: Minimum gap (in hours) to start new segment

    Returns:
        List of message segments (each segment is a list of messages)

    Hints:
        1. Iterate through messages, tracking current segment
        2. Compare timestamps: if gap > threshold, start new segment
        3. Don't forget the last segment!
    """
    if not messages:
        return []

    # YOUR CODE HERE
    segments: List[List[WhatsappMessage]] = []
    current_segment: List[WhatsappMessage] = [messages[0]]
    threshold = timedelta(hours=time_gap_hours)

    for i in range(1, len(messages)):
        gap = messages[i].timestamp - messages[i - 1].timestamp
        if gap > threshold:
            # Start new segment
            segments.append(current_segment)
            current_segment = []
        current_segment.append(messages[i])

    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)

    return segments


def chunk_segments(
    messages: List[WhatsappMessage],
    segments: List[Tuple[int, int]],
    chunk_length: int,
    chunk_overlap: int,
) -> List[ChunkObject]:
    """
    Apply sliding-window chunking within each segment.

    Args:
        messages: All messages (for text access)
        segments: List of (start_idx, end_idx) tuples
        chunk_length: Messages per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of ChunkObjects with segment_id in metadata

    Important:
        - Chunks should NOT cross segment boundaries
        - Track global message indices for message_ids
        - Add segment_id to metadata for debugging
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement chunk_segments")
```

### 4.3 Test Your Implementation

```bash
uv run pytest tests/test_segmenting_chunker.py -v
```

### 4.4 Enable Segmenting Chunker

Update `src/nicegui_app/workshop_config.py`:

```python
from workshop.rag.chunkers import SegmentingChunker
from workshop.rag.engines import RAGContextEngine

CHUNKER_CLASS = SegmentingChunker
ENGINE_CLASS = RAGContextEngine

CHUNKER_DEFAULTS = {
    "time_gap_hours": 6.0,  # 6-hour gaps start new segments
    "chunk_length": 6,
    "chunk_overlap": 2,
}
```

### 4.5 Compare Chunking Strategies

Ask the same question with different chunkers:

1. `MessageCountChunker` - Fixed windows, may cut mid-conversation
2. `SegmentingChunker` - Respects time boundaries

**Observe:** Which retrieves better context? Why?

### 4.6 Extension: Sentence Boundary Chunker

For an additional challenge, implement `src/workshop/rag/chunkers/sentence_boundary.py`:

- Chunk by token count instead of message count
- Respect sentence boundaries (no mid-sentence cuts)
- Use `tiktoken` for accurate token counting

---

## Wrap-Up: Key Takeaways

### The RAG Pipeline

```
Chunking -> Embedding -> Indexing -> Retrieval -> Re-ranking -> Prompting -> Generation
    |           |           |            |            |             |            |
    v           v           v            v            v             v            v
 Affects     Creates     Enables      Finds        Refines      Formats      Produces
what can    semantic     fast ANN    candidates    accuracy     context       answer
be found    vectors      search
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
- Evaluation metrics (RAGAS, faithfulness scores)
- Hybrid retrieval (BM25 + semantic)
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

Each exercise has a reference solution. Enable it by setting `USE_SOLUTIONS = True` in the relevant file:

| Exercise   | File                                      | Toggle                 |
| ---------- | ----------------------------------------- | ---------------------- |
| Similarity | `src/workshop/rag/engines/similarity.py`  | `USE_SOLUTIONS`        |
| Re-ranking | `src/workshop/rag/engines/qdrant.py`      | `USE_SOLUTIONS`        |
| Prompting  | `src/workshop/llm.py`                     | `USE_PROMPT_SOLUTIONS` |
| Segmenting | `src/workshop/rag/chunkers/segmenting.py` | `USE_SOLUTIONS`        |

---

## Next Steps

After the workshop, try:

1. **Your Own Data:** Export and analyze your personal WhatsApp chats
2. **Better Re-ranking:** Implement cross-encoder re-ranking
3. **Evaluation:** Add RAGAS metrics to measure retrieval quality
4. **Production:** Deploy to cloud with managed vector database

**Resources:**
- [Chunking Strategies Survey](https://arxiv.org/abs/2312.06648)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Qdrant Cloud](https://qdrant.tech/)
- [Cohere Rerank API](https://cohere.com/rerank)

---

*Happy RAG-ing!*
