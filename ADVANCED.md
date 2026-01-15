# Advanced Topics

This document covers architecture details, production usage patterns, and scaling considerations for the RAG Workshop.

## Architecture Overview

This RAG implementation supports a 4-hour workshop where participants learn chunking and retrieval strategies through hands-on implementation.

### Core Principles

1. **Participant Focus**: Only code in `src/workshop/rag/` needs attention
2. **Robust Boilerplate**: Remaining code is error-free, understandable, expandable
3. **Generic & Reusable**: Works in real projects, not just workshops
4. **Vector DB Flexibility**: Easy to swap Qdrant for pgvector, Pinecone, etc.

### Data Flow

```
Messages (WhatsApp)
    |
ChatContext (loading, token counting, windowing)
    |
Chunker.chunk_messages()
    |
ChunkObjects (with message_ids, metadata)
    |
Engine.add_context()
    |
Storage (in-memory, Qdrant, etc.)
    |
Engine.get_relevant_context(query)
    |
Retrieved ChunkObjects
    |
LLM (with context)
```

---

## Architecture Components

### 1. Chunker Layer (`src/workshop/rag/chunkers/`)

**Purpose**: Transform raw messages into ChunkObjects with metadata

**Interface**:
```python
class MessageChunkerProtocol(Protocol):
    params: BaseModel  # Pydantic model for auto-rendering in GUI

    def chunk_messages(messages: Sequence[WhatsappMessage]) -> Sequence[ChunkObject]
    def get_chunk_boundaries(num_messages: int) -> List[Tuple[int, int]]
```

**Implementations**:
| Chunker                   | Description                         | Status   |
| ------------------------- | ----------------------------------- | -------- |
| `MessageCountChunker`     | Fixed-size sliding window           | Complete |
| `SentenceBoundaryChunker` | Respects sentence boundaries        | Stub     |
| `SemanticChunker`         | Topic-aware with embeddings         | Stub     |
| `SegmentingChunker`       | Time-gap segmentation               | Stub     |
| `ContextualChunker`       | Conversation summary for embeddings | Stub     |

### 2. Engine Layer (`src/workshop/rag/engines/`)

**Purpose**: Store and retrieve chunks efficiently

**Interface**:
```python
class ContextEngineProtocol(Protocol):
    def add_context(context: Sequence[ChunkObject], embeddings: Optional[...] = None)
    def get_relevant_context(query: str, top_k: int = 10) -> Sequence[ChunkObject]

    @property
    def context(self) -> Sequence[ChunkObject]
```

**Implementations**:
| Engine                    | Description                   | Status   |
| ------------------------- | ----------------------------- | -------- |
| `NaiveContextEngine`      | Returns all chunks (baseline) | Complete |
| `SimilarityContextEngine` | Cosine similarity with NumPy  | Complete |
| `RAGContextEngine`        | Qdrant with ANN search        | Complete |

### 3. Configuration (`src/nicegui_app/workshop_config.py`)

Single file participants edit to switch components:

```python
CHUNKER_CLASS = MessageCountChunker
ENGINE_CLASS = NaiveContextEngine

CHUNKER_DEFAULTS = {
    "max_tokens": 25_000,
    "max_days": 25,
    "chunk_length": 6,
    "chunk_overlap": 4,
}

ENGINE_KWARGS = {}
```

---

## Production Usage

### RAGContextEngine with Embeddings

The RAGContextEngine uses Qdrant for vector storage with Pydantic-AI for embedding generation.

**Architecture**:
```
Pydantic-AI Embedder (from Hydra config)
    |
embed_fn callable (sync wrapper)
    |
RAGContextEngine (Qdrant local mode)
```

### Configuration

The embedding model is configured via Hydra (`configs/models/embedding_llm`). The engine receives an `embed_fn` callable:

```python
from workshop.rag.engines import RAGContextEngine

# embed_fn is injected by main.py from Hydra config
engine = RAGContextEngine(
    embed_fn=embed_fn,  # Callable[[list[str]], list[list[float]]]
    db_path=".qdrant",
    collection_name="production_chunks"
)
```

### Supported Embedding Models

RAGContextEngine supports any model Pydantic-AI supports:

**Gemini (Vertex AI)**:
```python
{"model_name": "google-vertex:gemini-embedding-001", "kwargs": {"task_type": "RETRIEVAL_QUERY"}}
```

**OpenAI**:
```python
{"model_name": "openai:text-embedding-3-small", "kwargs": {"dimensions": 1536}}
```

### Multi-Tenant Support

```python
def get_user_engine(user_id: str, embed_fn):
    return RAGContextEngine(
        embed_fn=embed_fn,
        db_path=f".qdrant/user_{user_id}",
        collection_name=f"user_{user_id}_chunks"
    )
```

---

## Key Design Decisions

### Stateless Chunkers
Chunkers don't hold state. They transform: `messages -> chunks`
- Simpler mental model
- Explicit data flow
- Easier debugging

### Dual Method Pattern
- `chunk_messages()` - Creates full ChunkObjects (for storage)
- `get_chunk_boundaries()` - Returns lightweight tuples (for preview)

### Message Traceability
ChunkObjects contain `message_ids: List[int]` mapping back to original messages
- Enables highlighting in GUI
- Supports analysis and debugging

### Metadata Conventions
Standard keys in `ChunkObject.metadata`:
- `start_idx`, `end_idx` - Message range
- `start_time`, `end_time` - Datetime range (auto-serialized)
- `speakers` - List of unique speakers
- `segment_id` - Optional conversation segment

---

## Project Structure

```
src/
  workshop/                 # RAG workshop code
    __init__.py
    chat.py                 # WhatsappMessage + load_whatsapp_chat()
    types.py                # LLMConfig TypedDict
    llm.py                  # Pydantic-AI agent factory
    structured_types.py     # RetrievalCoT, RetrievalResult
    utils.py                # Utility functions
    rag/                    # Chunkers and context engines
      chunkers/             # Chunker implementations
        chunker_types.py    # Protocols and base params
        message_count.py
        sentence_boundary.py
        semantic.py
        segmenting.py
        contextual.py
      engines/              # Context engine implementations
        context_types.py    # ChunkObject and engine protocol
        naive.py
        similarity.py
        qdrant.py

  nicegui_app/              # Workshop UI
    main.py                 # Page layout and rendering
    state.py                # Application state management
    workshop_config.py      # Participant configuration file
    vectordb_view.py        # VectorDB visualization
    llm_ui_utils.py         # Pydantic model rendering

configs/                    # Hydra configuration
  models/                   # LLM model configs
  paths/                    # Path configurations

tests/                      # Test suite
chats/                      # Sample chat data
```

---

## Scaling Considerations

| Vector DB       | Use Case                            |
| --------------- | ----------------------------------- |
| Qdrant (local)  | Single-machine, millions of vectors |
| pgvector        | PostgreSQL extension, < 1M vectors  |
| Qdrant/Pinecone | Cloud-native, billions of vectors   |

To swap vector DBs:
1. Create new engine class implementing `ContextEngineProtocol`
2. Implement `add_context()` and `get_relevant_context()`
3. Update `workshop_config.py` to use new engine

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_message_count_chunker.py -v

# Type checking
uv run pyright src/
```
