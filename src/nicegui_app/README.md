# NiceGUI Chat Application

A chat interface for interacting with conversation data using RAG (Retrieval-Augmented Generation). Replaces the Streamlit-based `st_chat_vis.py` with cleaner state management.

## Features

- **Persistent State**: Uses NiceGUI's `app.storage.user` for state persistence across sessions
- **LLM Integration**: Supports multiple LLM providers via `pydantic-ai` (Anthropic, OpenAI, Bedrock, Vertex AI)
- **Streaming Responses**: Optional token-by-token streaming with typing indicator
- **Context Highlighting**: Visualize which parts of the context the LLM used in its response
- **RTL Support**: Auto-detection and manual toggle for Hebrew/Arabic text
- **Usage Tracking**: Token usage and cost estimation per model

## Architecture

```
nicegui_app/
├── __init__.py        # Public exports
├── main.py            # Application entry point, page routes, core logic
├── state.py           # Pydantic models for state (AppState, ContextParams, etc.)
├── llm_ui_utils.py    # Reusable UI components (model selector, highlighting, auto-UI)
└── README.md
```

## Auto-UI from Pydantic Models

The killer feature: **automatically generate UI controls from any Pydantic model**.

### Usage

```python
from nicegui_app import render_model_controls

# Auto-generate sliders, toggles, dropdowns from model fields
render_model_controls(
    model=my_pydantic_model,
    title="Optional Title",
    on_change=lambda: handle_update(),
)
```

### How It Works

The renderer inspects Pydantic field metadata to determine the appropriate control:

| Field Type          | Constraints          | UI Control (default)    |
| ------------------- | -------------------- | ----------------------- |
| `int`               | `ge`, `le`           | Number input            |
| `float`             | `ge`, `le`           | Slider                  |
| `int` / `float`     | + `ui_type="number"` | Number input (override) |
| `int` / `float`     | + `ui_type="slider"` | Slider (override)       |
| `bool`              | -                    | Toggle switch           |
| `str`               | -                    | Text input              |
| `Literal["a", "b"]` | -                    | Dropdown select         |

### Field Metadata

Use `Field()` to provide UI hints:

```python
from pydantic import BaseModel, Field

class LLMKwargs(BaseModel):
    temperature: float = Field(
        default=0.7,
        ge=0.0,        # Input min
        le=2.0,        # Input max
        json_schema_extra={
            "step": 0.05,          # Increment step
            "label": "Temperature",  # Display name
            "tooltip": "Higher = more creative",  # Help text
            "ui_type": "slider",   # Use slider instead of number input
        }
    )

    internal_field: str = Field(
        default="hidden",
        json_schema_extra={"hidden": True}  # Skip in UI
    )
```

### Disabling Controls

Use the `disabled` parameter to gray out controls:

```python
render_model_controls(
    model=state.context_params,
    disabled=state.has_custom_chunker,  # Gray out when custom chunker active
)
```

### Example: Complete Model

```python
class ContextParams(BaseModel):
    num_tokens: int = Field(
        default=25_000,
        ge=10_000,
        le=200_000,
        json_schema_extra={"step": 10_000, "label": "Max Tokens"},
    )
    num_days: int = Field(
        default=10,
        ge=1,
        le=360,
        json_schema_extra={"step": 1, "label": "Max Days"},
    )
    # Nested models are automatically hidden
    llm_kwargs: LLMKwargs = Field(
        default_factory=LLMKwargs,
        json_schema_extra={"hidden": True}
    )
```

## Running the App

```bash
cd /path/to/project
python -m nicegui_app.main
```

Or via Hydra:

```bash
python -m nicegui_app.main --config-path=../../configs --config-name=main_app_cfg
```

## State Management

### AppState (Persisted)

Stored in `app.storage.user["app_state"]`, survives page refreshes:

- `chat_path`: Path to loaded chat file
- `context_params`: Hyperparameters (tokens, days, chunking, LLM kwargs)
- `messages`: Chat history
- `usage_maps`: Token usage per model
- `streaming`: Streaming configuration
- `display`: Display preferences (RTL mode, context view mode)

### PageState (Runtime)

Page-scoped, not persisted:

- `llm_wrapper`: The pydantic-ai Agent instance
- `refresh_*`: UI refresh callbacks

## Custom Chunking

The app supports custom chunking strategies via `ChunkProvider` (class) or `ChunkProviderFn` (callable):

```python
from nicegui_app import ChunkCoordinates

# Option 1: Class with get_chunks method
class SemanticChunker:
    """Custom chunker that uses semantic boundaries."""

    def get_chunks(self, text: str) -> ChunkCoordinates:
        # Return list of (start, end) indices
        return [(0, 100), (80, 200), (180, 300)]

state.chunk_provider = SemanticChunker()

# Option 2: Simple callable
def overlap_chunker(text: str) -> ChunkCoordinates:
    return [(i, i + 100) for i in range(0, len(text), 50)]

state.chunk_provider = overlap_chunker
```

When a custom chunker is active:
- UI chunking controls are grayed out (disabled)
- `state.compute_chunk_indices(text)` delegates to the provider
- Visualization still works, using custom chunk boundaries

Future: This pattern extends to vectordb/reranker integration in `chunking_strategies.py`.

## Key Components

### `render_model_controls()`
Auto-generates UI from Pydantic models. The killer feature.

### `create_model_selector()`
Dropdown for switching between configured LLM models.

### `create_usage_stats()`
Displays token usage and estimated costs.

### `render_with_highlights()`
Highlights context chunks used by the LLM in responses.

### `render_chunk_preview()`
Shows document chunks with boundaries.

## Configuration

Models are loaded from Hydra config files in `configs/models/`:

```yaml
# configs/models/anthropic_claude_sonnet_45.yaml
model_name: anthropic:claude-sonnet-4-20250514
model_provider: anthropic
```

## Dependencies

**See `pyproject.toml`.**
