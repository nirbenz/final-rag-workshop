# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG Workshop is a 4-hour hands-on educational project for learning chunking and retrieval strategies in RAG (Retrieval-Augmented Generation) systems. Built with WhatsApp chat exports as sample data, it teaches participants to implement custom chunking strategies, build context retrieval engines, and integrate retrieval with LLM agents using Pydantic-AI.

The workshop is progressive: participants start with a working baseline and incrementally implement more sophisticated components across 4 phases.

## Common Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the workshop UI
uv run python -m nicegui_app.main

# Testing
uv run pytest tests/ -v                       # All tests
uv run pytest tests/test_message_count_chunker.py -v  # Specific file

# Type checking
uv run pyright src/

# Code formatting and linting
ruff check src/
ruff format src/
```

## Architecture

### Directory Layout
```
src/
  workshop/              # Core RAG workshop module
    rag/                 # Chunkers and context engines
      message_count_chunker.py      # Phase 1 baseline chunker
      sentence_boundary_chunker.py  # Phase 2 stub for implementation
      naive_context_engine.py       # Phase 1 baseline engine
      sim_context_engine.py         # Phase 2 NumPy similarity
      rag_context_engine.py         # Phase 3 Qdrant ANN
    chat.py              # WhatsApp message loading
    llm.py               # Pydantic-AI agent factory
    types.py             # LLMConfig TypedDict
  nicegui_app/           # Interactive UI using NiceGUI
    main.py              # Page layout, event handlers
    state.py             # AppState, ChatMessage models
    workshop_config.py   # PARTICIPANT CONFIG FILE (edit this!)
tests/                   # pytest test suite
configs/                 # Hydra configuration files
chats/                   # Sample WhatsApp chat data
```

### Key Design Patterns

**Protocol-Based Architecture**: Uses Python `typing.Protocol` for structural typing instead of inheritance:
- `MessageChunkerProtocol` - Chunkers implement `chunk_messages()` and `get_chunk_boundaries()`
- `ContextEngineProtocol` - Engines implement `add_context()` and `get_relevant_context()`

**Dual-Method Pattern in Chunkers**:
- `chunk_messages()` - Creates full ChunkObjects (expensive, for storage)
- `get_chunk_boundaries()` - Returns lightweight `(start, end)` tuples (fast, for UI preview)

**Data Flow**:
```
Messages -> ChatContext (windowing) -> Chunker -> ChunkObjects ->
Engine (storage/indexing) -> get_relevant_context() -> Retrieved chunks -> LLM
```

### Key Entry Points

| File                                 | Purpose                                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| `src/nicegui_app/main.py`            | UI application entry point                                     |
| `src/nicegui_app/workshop_config.py` | **Participant edit point** - controls chunker/engine selection |
| `src/workshop/chat.py`               | WhatsApp chat loading (`load_whatsapp_chat()`)                 |
| `src/workshop/llm.py`                | LLM agent creation (`get_pydantic_agent()`)                    |

## Code Conventions

### ChunkObject Metadata Standards
Always include these keys in `ChunkObject.metadata`:
- `start_idx` (int) - First message index
- `end_idx` (int) - Last message index (exclusive)
- `start_time` (datetime) - Timestamp of first message
- `end_time` (datetime) - Timestamp of last message
- `speakers` (List[str]) - Unique speakers in chunk
- `message_ids` - **Required** for UI highlighting/traceability

### LLMConfig Format
```python
LLMConfig = {
    "model_name": "provider:model",  # e.g., "google-vertex:gemini-embedding-001"
    "kwargs": {...},                  # Provider-specific arguments
    "structured_output_type": None,   # Optional Pydantic model
}
```

### Supported Embedding Models (via Pydantic-AI)
```python
# Google Gemini (Vertex AI)
{"model_name": "google-vertex:gemini-embedding-001", "kwargs": {"task_type": "RETRIEVAL_DOCUMENT"}}

# OpenAI
{"model_name": "openai:text-embedding-3-small", "kwargs": {"dimensions": 1536}}
```

### Code Quality Rules (ruff.toml)
- Target: Python 3.12+
- Line length: 120 characters
- Type checking mode: basic (not strict)

## Key Dependencies

| Component       | Technology           | Purpose                                    |
| --------------- | -------------------- | ------------------------------------------ |
| Package Manager | `uv` (>=0.9.0)       | Fast Python package management             |
| Web UI          | NiceGUI (>=3.3.1)    | Interactive workshop interface             |
| LLM Integration | Pydantic-AI (1.37.0) | Multi-provider LLM abstraction             |
| Vector DB       | Qdrant               | Local-mode vector database with ANN search |
| Text Processing | tiktoken, whatstk    | Token counting, WhatsApp chat parsing      |
| Config          | Hydra (1.3.2)        | Structured configuration system            |

## Workshop Context

This is an educational project. When helping:

1. **Prioritize clarity over optimization** - This teaches RAG concepts progressively
2. **Single config file** - Participants only edit `workshop_config.py` to switch implementations
3. **Stateless chunkers** - No state between calls; pure functions for simplicity
4. **Protocol compliance** - Implement protocols as structural typing, not inheritance
5. **Test fixtures** - Use `test_messages` and `small_test_messages` from `conftest.py`

### Workshop Phases
| Phase      | Chunker                                               | Engine                                      |
| ---------- | ----------------------------------------------------- | ------------------------------------------- |
| 1          | MessageCountChunker                                   | NaiveContextEngine                          |
| 2          | SentenceBoundaryChunker (implement)                   | NaiveContextEngine                          |
| 3          | MessageCountChunker                                   | SimilarityContextEngine or RAGContextEngine |
| Extensions | SemanticChunker, SegmentingChunker, ContextualChunker | Various                                     |

## Environment Variables

API keys configured in `.env`:
- `OPENAI_API_KEY` - OpenAI API access
- `ANTHROPIC_API_KEY` - Anthropic API access
- `GOOGLE_API_KEY` - Google Gemini API access
- `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` - Vertex AI access

## Testing

Test suite in `tests/` covers:
- Chunker implementations and parameter validation
- Context engine storage/retrieval
- End-to-end RAG pipeline

Run with: `uv run pytest tests/ -v`
