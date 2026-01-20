# RAG Workshop

A hands-on workshop for learning Retrieval-Augmented Generation (RAG) through implementing chunking and retrieval strategies. Build a production-ready RAG system in 4 hours.

## Quick Start

### MacOS

```bash
# Install uv (package manager)
brew install uv

# Install dependencies
uv sync

# Run the workshop application
uv run python -m nicegui_app.main
```

### Linux/WSL

```bash
# Install uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart shell or source the env
source $HOME/.local/bin/env

# Install dependencies
uv sync

# Run the workshop application
uv run python -m nicegui_app.main
```

### Windows (Non-WSL)

See [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

```powershell
# Install uv (package manager) using PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart terminal, then install dependencies
uv sync

# Run the workshop application
uv run python -m nicegui_app.main
```

## Environment Variables

Set these in a `.env` file or export them:

```bash
export GOOGLE_API_KEY="your-key-here"           # For Google AI Studio
export GOOGLE_CLOUD_PROJECT="your-project"      # For Vertex AI
export GOOGLE_CLOUD_LOCATION="us-central1"      # For Vertex AI
export OPENAI_API_KEY="your-key-here"           # For OpenAI
export ANTHROPIC_API_KEY="your-key-here"        # For Anthropic
```

## Exporting WhatsApp Chat

Follow instructions [for your device](https://whatstk.readthedocs.io/en/latest/source/getting_started/export_chat.html).
Place the exported text file at `chats/default_chat.txt` or use the included `chats/example_chat.txt`.

---

## Workshop Overview (4 hours)

A progressive, hands-on workshop where participants implement core RAG components:

| Phase       | Duration | What You Build                              |
| ----------- | -------- | ------------------------------------------- |
| **Phase 1** | 35 min   | Explore baseline system (no coding)         |
| **Phase 2** | 55 min   | Cosine similarity & top-k retrieval         |
| **Phase 3** | 60 min   | ANN search, re-ranking & prompt engineering |
| **Phase 4** | 45 min   | Time-gap based conversation segmentation    |

### Exercise Files

All participant code goes in `src/workshop/rag/exercises/`:

```
exercises/
  similarity.py    # Phase 2: cosine_similarity(), get_top_k()
  reranking.py     # Phase 3: rerank()
  prompting.py     # Phase 3: system prompts, context formatting
  segmenting.py    # Phase 4: segment_by_time_gaps(), chunk_segments()
```

Solutions in `src/workshop/rag/solutions/` - swap via `USE_SOLUTIONS = True` after each exercise.

### Phase Configurations

Edit `src/nicegui_app/workshop_config.py` to switch between phases.

See [WORKSHOP.md](WORKSHOP.md) for the detailed instructor guide with timing, discussion points, and common issues.

---

## License

MIT
