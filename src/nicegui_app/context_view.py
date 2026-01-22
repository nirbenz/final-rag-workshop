# Created by Nir Ben-Zvi
# Context view components for NiceGUI applications
#
# Modular context display with multiple view modes for RAG applications.
# Supports default, chunks, highlights, and vectordb view modes.

from typing import Any, Callable, List, Protocol, Sequence, Tuple

from loguru import logger
from nicegui import ui

from nicegui_app.llm_ui_utils import (
    get_direction_classes,
    get_text_direction,
    render_chunk_preview_with_boundaries,
    render_messages_with_collapsible_highlights,
    render_with_highlights,
)


class ChatContextProtocol(Protocol):
    """Protocol for chat context."""

    context: Sequence[Any]
    text_context: str


class ChunkerProtocol(Protocol):
    """Protocol for chunker with params."""

    params: Any


class DisplaySettingsProtocol(Protocol):
    """Protocol for display settings."""

    context_view_mode: str
    rtl_mode: str


class AppStateProtocol(Protocol):
    """Protocol for application state used by context view."""

    context: Any  # ChatContext or None
    chunker: Any  # Chunker or None
    display: Any  # DisplayConfig
    messages: List[Any]

    def get_chunk_boundaries(self, num_messages: int) -> List[Tuple[int, int]]: ...


class RefreshablePageState(Protocol):
    """Protocol for page state with refresh callbacks."""

    refresh_context: Callable[[], object]


def render_default_context(
    context: ChatContextProtocol,
    rtl_mode: str,
) -> None:
    """
    Render all messages in default view mode.

    Args:
        context: Chat context with messages
        rtl_mode: RTL display mode
    """
    full_text = "\n".join(getattr(msg, "timed_form", lambda: str(msg))() for msg in context.context)
    direction = get_text_direction(full_text, rtl_mode)
    dir_classes = get_direction_classes(direction)

    with ui.element("div").classes(f"{dir_classes} break-all").props(f'dir="{direction}"'):
        for msg in context.context:
            msg_text = getattr(msg, "timed_form", lambda: str(msg))()
            ui.markdown(msg_text).classes("break-all")


def render_chunks_view(
    context: ChatContextProtocol,
    chunker: ChunkerProtocol,
    get_boundaries: Callable[[int], List[Tuple[int, int]]],
    rtl_mode: str,
) -> None:
    """
    Render chunk preview mode showing chunk boundaries.

    Args:
        context: Chat context with messages
        chunker: Chunker with params for chunk_length and overlap
        get_boundaries: Function to get chunk boundaries
        rtl_mode: RTL display mode
    """
    messages_text = [getattr(msg, "timed_form", lambda: str(msg))() for msg in context.context]
    num_messages = len(messages_text)

    boundaries = get_boundaries(num_messages)

    chunk_length = chunker.params.chunk_length
    overlap = chunker.params.chunk_overlap

    render_chunk_preview_with_boundaries(
        messages=messages_text,
        boundaries=boundaries,
        chunk_length=chunk_length,
        overlap=overlap,
        rtl_mode=rtl_mode,
    )


def render_highlights_view(
    context: ChatContextProtocol,
    messages: List[Any],
    rtl_mode: str,
    highlight_source: str = "retrieval",
) -> None:
    """
    Render highlights mode showing retrieved/used context.

    Args:
        context: Chat context
        messages: Chat message history
        rtl_mode: RTL display mode
        highlight_source: "retrieval" for message_ids, "model" for context_used
    """
    if not messages:
        ui.label("No highlighted context available").classes("text-gray-400 italic")
        ui.label("Send a message to see retrieved context").classes("text-gray-500 text-sm")
        return

    last_msg = messages[-1]
    if last_msg.role != "assistant":
        ui.label("No highlighted context available").classes("text-gray-400 italic")
        ui.label("Send a message to see retrieved context").classes("text-gray-500 text-sm")
        return

    try:
        if highlight_source == "model":
            # Use context_used from model's structured output (now a List[str])
            if hasattr(last_msg, "context_used") and last_msg.context_used:
                render_with_highlights(
                    context.text_context,
                    last_msg.context_used,  # Already a list
                    rtl_mode=rtl_mode,
                )
                return
            else:
                ui.label("Model did not report context_used").classes("text-amber-400 italic")
                ui.label("The model's structured output has no context_used field").classes("text-gray-500 text-sm")
                return

        # Default: use retrieved_message_ids from retrieval engine
        if hasattr(last_msg, "retrieved_message_ids") and last_msg.retrieved_message_ids:
            render_messages_with_collapsible_highlights(
                context.context,
                last_msg.retrieved_message_ids,
                rtl_mode=rtl_mode,
            )
            return

        # Fallback if retrieval returned empty
        ui.label("No context was retrieved for this query").classes("text-amber-400 italic")
        ui.label("The retrieval engine returned no matching chunks").classes("text-gray-500 text-sm")
        return

    except Exception as e:
        logger.warning(f"Highlighting failed: {e}")
        ui.label(f"Highlighting failed: {e}").classes("text-red-400 text-sm")


def render_vectordb_view(state: Any, on_page_change: Callable[[], Any]) -> None:
    """
    Render vectordb mode showing stored chunks in the engine.

    Args:
        state: Application state with engine
        on_page_change: Callback to refresh the view when page changes
    """
    # Import lazily to avoid circular imports
    from nicegui_app.vectordb_view import render_vectordb_content, render_vectordb_header

    has_chunks = render_vectordb_header(state, on_page_change)
    if has_chunks:
        render_vectordb_content(state)


def create_context_view(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
) -> None:
    """
    Create the context display with view mode selector.

    View modes:
    - default: Plain text display
    - chunks: Show chunk boundaries (for tuning chunking params)
    - highlights: Show LLM-used context highlighted (if available)
    - vectordb: Show stored chunks in the engine

    Args:
        state: Application state
        page_state: Page state with refresh callbacks
    """
    with ui.card().classes("w-full h-full flex flex-col min-w-0 overflow-hidden"):
        with ui.row().classes("items-center justify-between mb-2 flex-none gap-2"):
            ui.label("Context").classes("text-xl font-bold")

            with ui.row().classes("items-center gap-2"):

                def on_mode_change(e):
                    state.display.context_view_mode = e.value
                    highlight_source_select.set_visibility(e.value == "highlights")
                    context_display.refresh()

                ui.select(
                    options=["default", "chunks", "highlights", "vectordb"],
                    value=state.display.context_view_mode,
                    on_change=on_mode_change,
                ).props("dense outlined").classes("w-32")

                def on_highlight_source_change(e):
                    state.display.highlight_source = e.value
                    context_display.refresh()

                highlight_source_select = (
                    ui.select(
                        options=["retrieval", "model"],
                        value=state.display.highlight_source,
                        on_change=on_highlight_source_change,
                    )
                    .props("dense outlined")
                    .classes("w-28")
                )
                highlight_source_select.set_visibility(state.display.context_view_mode == "highlights")

        with ui.scroll_area().classes("flex-grow w-full overflow-x-hidden"):

            @ui.refreshable
            def context_display():
                if state.context is None:
                    ui.label("No context loaded").classes("text-gray-400 italic")
                    ui.label("Load a chat file to see context here").classes("text-gray-500 text-sm")
                    return

                mode = state.display.context_view_mode

                if mode == "chunks":
                    if state.chunker:
                        render_chunks_view(
                            context=state.context,
                            chunker=state.chunker,
                            get_boundaries=state.get_chunk_boundaries,
                            rtl_mode=state.display.rtl_mode,
                        )
                    else:
                        ui.label("No chunker configured").classes("text-gray-400 italic")
                    return

                if mode == "highlights":
                    render_highlights_view(
                        context=state.context,
                        messages=state.messages,
                        rtl_mode=state.display.rtl_mode,
                        highlight_source=state.display.highlight_source,
                    )
                    return

                if mode == "vectordb":
                    render_vectordb_view(state, on_page_change=context_display.refresh)
                    return

                # Default mode
                render_default_context(
                    context=state.context,
                    rtl_mode=state.display.rtl_mode,
                )

            context_display()
            page_state.refresh_context = context_display.refresh
