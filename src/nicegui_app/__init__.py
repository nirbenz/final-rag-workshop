# NiceGUI Chat Application
# Replaces Streamlit-based chat visualization with cleaner state management

from nicegui_app.llm_ui_utils import (
    create_model_selector,
    create_usage_stats,
    render_chunk_preview,
    render_model_controls,
    render_with_coordinate_highlights,
    render_with_highlights,
)
from nicegui_app.state import (
    AppState,
    ChunkCoordinates,
    ChunkProvider,
    ChunkProviderFn,
    DisplayConfig,
    LLMKwargs,
    StreamingConfig,
    get_or_create_state,
    update_chunker_param,
    update_context_param,  # Deprecated
    update_llm_param,
)

__all__ = [
    "AppState",
    "ChunkCoordinates",
    "ChunkProvider",
    "ChunkProviderFn",
    "DisplayConfig",
    "LLMKwargs",
    "StreamingConfig",
    "get_or_create_state",
    "update_chunker_param",
    "update_context_param",  # Deprecated
    "update_llm_param",
    "create_model_selector",
    "create_usage_stats",
    "render_model_controls",
    "render_with_highlights",
    "render_with_coordinate_highlights",
    "render_chunk_preview",
]
