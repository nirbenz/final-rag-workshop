# Created by Nir Ben-Zvi
# VectorDB View Component
# me@nirbnzvi.com

"""
VectorDB visualization component for RAG workshop.

Displays chunks stored in the context engine with pagination for performance.
Helps participants understand what's actually stored in the vector database.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

from nicegui import ui


def get_vectordb_page_info(state: Any) -> Optional[Tuple[int, int, int, int, Sequence[Any]]]:
    """
    Get pagination info and current page of chunks.

    Returns None if engine is not available or has no chunks.
    Otherwise returns (total_chunks, page, total_pages, offset, stored_chunks).
    """
    if state.engine is None:
        return None

    try:
        if hasattr(state.engine, "context_count"):
            total_chunks = state.engine.context_count
        else:
            total_chunks = len(state.engine.context)
    except AttributeError:
        return None

    if total_chunks == 0:
        return None

    page = state.display.vectordb_page
    page_size = state.display.vectordb_page_size
    total_pages = (total_chunks + page_size - 1) // page_size

    # Clamp page to valid range
    if page >= total_pages:
        state.display.vectordb_page = max(0, total_pages - 1)
        page = state.display.vectordb_page

    offset = page * page_size
    if hasattr(state.engine, "get_context_page"):
        stored_chunks = state.engine.get_context_page(offset=offset, limit=page_size)
    else:
        stored_chunks = state.engine.context[offset : offset + page_size]

    return (total_chunks, page, total_pages, offset, stored_chunks)


def render_vectordb_header(state: Any, on_page_change: Callable[[], Any]) -> bool:
    """
    Render the VectorDB header with stats and pagination controls.

    This should be rendered outside the scroll area to stay fixed at top.

    Args:
        state: AppState with engine attribute and display.vectordb_page/vectordb_page_size
        on_page_change: Callback to refresh the view when page changes

    Returns:
        True if there are chunks to display, False otherwise
    """
    if state.engine is None:
        ui.label("No engine loaded").classes("text-gray-500 italic")
        ui.label("Upload a file to initialize the engine").classes("text-xs text-gray-400 mt-1")
        return False

    try:
        if hasattr(state.engine, "context_count"):
            total_chunks = state.engine.context_count
        else:
            total_chunks = len(state.engine.context)
    except AttributeError:
        ui.label("Engine does not support context inspection").classes("text-amber-500 italic")
        return False

    if total_chunks == 0:
        ui.label("No chunks stored yet").classes("text-gray-500 italic")
        ui.label("Upload a file to chunk and store messages").classes("text-xs text-gray-400 mt-1")
        return False

    # Pagination state
    page = state.display.vectordb_page
    page_size = state.display.vectordb_page_size
    total_pages = (total_chunks + page_size - 1) // page_size

    # Clamp page to valid range
    if page >= total_pages:
        state.display.vectordb_page = max(0, total_pages - 1)
        page = state.display.vectordb_page

    offset = page * page_size

    # Header with stats and pagination controls
    with ui.row().classes("w-full items-center justify-between"):
        with ui.column().classes("gap-0"):
            ui.label(f"Stored Chunks: {total_chunks}").classes("text-lg font-bold")
            ui.label(f"Engine: {type(state.engine).__name__}").classes("text-xs text-blue-500")

        # Pagination controls
        with ui.row().classes("items-center gap-1"):
            ui.button(
                icon="first_page",
                on_click=lambda: _go_to_page(state, 0, on_page_change),
            ).props("flat dense").classes("text-gray-600").bind_enabled_from(
                state.display, "vectordb_page", backward=lambda p: p > 0
            )
            ui.button(
                icon="chevron_left",
                on_click=lambda: _go_to_page(state, page - 1, on_page_change),
            ).props("flat dense").classes("text-gray-600").bind_enabled_from(
                state.display, "vectordb_page", backward=lambda p: p > 0
            )

            ui.label(f"{page + 1} / {total_pages}").classes("text-sm text-gray-600 mx-2")

            ui.button(
                icon="chevron_right",
                on_click=lambda: _go_to_page(state, page + 1, on_page_change),
            ).props("flat dense").classes("text-gray-600").bind_enabled_from(
                state.display, "vectordb_page", backward=lambda p: p < total_pages - 1
            )
            ui.button(
                icon="last_page",
                on_click=lambda: _go_to_page(state, total_pages - 1, on_page_change),
            ).props("flat dense").classes("text-gray-600").bind_enabled_from(
                state.display, "vectordb_page", backward=lambda p: p < total_pages - 1
            )

    # Showing range info
    start_idx = offset + 1
    end_idx = min(offset + page_size, total_chunks)
    ui.label(f"Showing {start_idx}-{end_idx} of {total_chunks}").classes("text-xs text-gray-500")

    return True


def render_vectordb_content(state: Any) -> None:
    """
    Render the VectorDB chunk cards (scrollable content).

    This should be rendered inside the scroll area.

    Args:
        state: AppState with engine attribute
    """
    page_info = get_vectordb_page_info(state)
    if page_info is None:
        return

    _total_chunks, _page, _total_pages, offset, stored_chunks = page_info

    # Render chunks for current page
    for i, chunk in enumerate(stored_chunks):
        chunk_num = offset + i + 1
        with ui.card().classes("w-full mb-3"):
            # Chunk header
            with ui.row().classes("w-full items-center gap-2 mb-2"):
                ui.badge(f"#{chunk_num}", color="blue").classes("text-xs")
                ui.label(f"ID: {chunk.id}").classes("text-xs font-mono text-gray-600")

                # Show message range
                if hasattr(chunk, "message_ids") and chunk.message_ids:
                    msg_range = f"Messages {min(chunk.message_ids)}-{max(chunk.message_ids)}"
                    ui.badge(msg_range, color="gray").classes("text-xs")

            # Chunk text preview
            preview_text = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            ui.label(preview_text).classes("text-sm whitespace-pre-wrap break-words text-gray-700")

            # Metadata section (expandable)
            if chunk.metadata:
                with ui.expansion("Metadata", icon="info").classes("w-full mt-2"):
                    # Format metadata nicely
                    metadata_formatted = {}
                    for key, value in chunk.metadata.items():
                        if key == "embedding_context":
                            # Don't show embedding context (too verbose)
                            metadata_formatted[key] = f"<{len(str(value))} chars>"
                        else:
                            metadata_formatted[key] = value

                    # Display as formatted JSON
                    with ui.card().classes("w-full bg-gray-50"):
                        ui.json_editor({"content": {"json": metadata_formatted}}).classes("w-full").props("mode=view")

            # Full text expansion
            if len(chunk.text) > 200:
                with ui.expansion("Full Text", icon="article").classes("w-full mt-2"):
                    ui.label(chunk.text).classes("text-sm whitespace-pre-wrap break-words text-gray-600")


def _go_to_page(state: Any, page: int, on_page_change: Callable[[], Any]) -> None:
    """Navigate to a specific page and refresh the view."""
    state.display.vectordb_page = max(0, page)
    on_page_change()
