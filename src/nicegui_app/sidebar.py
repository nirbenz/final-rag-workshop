# Created by Nir Ben-Zvi
# Sidebar components for NiceGUI applications
#
# Modular sidebar with control panels for chat/RAG applications.
# Can be reused across different NiceGUI apps that need similar controls.

from pathlib import Path
import tempfile
from typing import Any, Callable, Optional, Protocol

from loguru import logger
from nicegui import ui

from nicegui_app.llm_ui_utils import (
    Config,
    create_model_selector,
    create_usage_stats,
    render_model_controls,
)


class RefreshablePageState(Protocol):
    """Protocol for page state with refresh callbacks."""

    refresh_context: Callable[[], object]
    refresh_context_engine: Callable[[], object]
    refresh_chat: Callable[[], object]
    refresh_usage: Callable[[], object]
    refresh_context_stats: Callable[[], object]
    llm_wrapper: Optional[Any]


class ChunkerProtocol(Protocol):
    """Protocol for chunker interface."""

    params: Any

    def chunk_messages(self, messages: Any) -> Any: ...


class EngineProtocol(Protocol):
    """Protocol for context engine interface."""

    def add_context(self, chunks: Any) -> None: ...

    @property
    def context(self) -> Any: ...


class ChatContextProtocol(Protocol):
    """Protocol for chat context."""

    start_time: Any
    end_time: Any
    num_messages: int
    num_tokens: int
    context: Any
    text_context: str


class AppStateProtocol(Protocol):
    """Protocol for application state used by sidebar."""

    config: Any  # Config dict, may be None initially
    chat_path: str
    context: Any  # ChatContext or None
    messages: list
    chunker: Any  # Chunker or None
    engine: Any  # Engine or None
    has_custom_chunker: Any  # Property returning bool
    llm_kwargs: Any
    display: Any
    streaming: Any
    selected_model_idx: int
    current_model_name: str
    usage_maps: Any


def create_chat_loader_card(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
    token_counter: Callable[[str], int],
    context_class: type,
    apply_context_params: Callable[[AppStateProtocol], None],
) -> None:
    """
    Create the chat file loader card.

    Args:
        state: Application state
        page_state: Page state with refresh callbacks
        token_counter: Function to count tokens in text
        context_class: Class to instantiate for chat context (e.g., ChatContext)
        apply_context_params: Function to apply context parameters after loading
    """
    with ui.card().classes("w-full"):
        ui.label("Chat Loader").classes("font-semibold mb-2")

        async def handle_file_upload(e):
            if not e.file:
                return

            try:
                temp_dir = Path(tempfile.gettempdir()) / "nicegui_chat"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / f"upload_{e.file.name}"

                await e.file.save(temp_path)

                state.chat_path = str(temp_path)
                state.context = context_class(
                    chat_path=str(temp_path),
                    token_counter=token_counter,
                )
                state.messages = []
                apply_context_params(state)

                page_state.refresh_context_stats()
                page_state.refresh_context()
                page_state.refresh_chat()
                file_label.set_text(f"Loaded: {e.file.name}")
                ui.notify(f"Loaded {e.file.name}", type="positive")
                logger.info(f"Loaded chat from upload: {e.file.name}")

            except Exception as ex:
                ui.notify(f"Failed to load: {ex}", type="negative")
                logger.error(f"Failed to load chat: {ex}")

        ui.upload(
            label="Choose chat file",
            on_upload=handle_file_upload,
            auto_upload=True,
        ).classes("w-full").props('accept=".txt,.json"')

        file_label = ui.label("No file loaded").classes("text-xs text-gray-500 mt-1")


def create_context_params_card(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
    apply_context_params: Callable[[AppStateProtocol], None],
) -> None:
    """
    Create the context parameters card with chunker controls.

    Args:
        state: Application state
        page_state: Page state with refresh callbacks
        apply_context_params: Function to apply context parameters
    """
    with ui.card().classes("w-full mt-4"):
        ui.label("Context Parameters").classes("font-semibold mb-2")

        @ui.refreshable
        def context_stats():
            if state.context:
                try:
                    ui.markdown(f"""
```json
{{
  "start": "{state.context.start_time.strftime('%y-%m-%d %H:%M')}",
  "end": "{state.context.end_time.strftime('%y-%m-%d %H:%M')}",
  "messages": {state.context.num_messages},
  "tokens": {state.context.num_tokens}
}}
```
                    """)
                except Exception:
                    ui.label("Context stats unavailable").classes("text-gray-500 text-xs")
            else:
                ui.label("No context loaded").classes("text-gray-500 text-xs")

        page_state.refresh_context_stats = context_stats.refresh

        def on_context_param_change() -> None:
            apply_context_params(state)
            page_state.refresh_context_stats()
            page_state.refresh_context()

        # Chunking controls - show chunker params if available
        if state.chunker is not None:
            ui.label(f"Chunker: {type(state.chunker).__name__}").classes("text-xs text-blue-500 italic mb-2")
            render_model_controls(
                model=state.chunker.params,
                on_change=on_context_param_change,
            )
        elif state.has_custom_chunker:
            ui.label("Custom chunker active").classes("text-xs text-amber-500 italic mb-2")

        ui.separator().classes("my-2")

        context_stats()


def create_vectordb_card(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
) -> None:
    """
    Create the vector database / engine control card.

    Args:
        state: Application state
        page_state: Page state with refresh callbacks
    """
    with ui.card().classes("w-full mt-4"):
        ui.label("Vector Database").classes("font-semibold mb-2")

        if state.engine is not None:
            engine_name = type(state.engine).__name__
            ui.label(f"Engine: {engine_name}").classes("text-xs text-green-500 italic mb-2")

            @ui.refreshable
            def engine_stats_display():
                try:
                    stored_chunks = len(state.engine.context) if state.engine else 0
                    ui.label(f"Stored chunks: {stored_chunks}").classes("text-sm text-gray-300 mb-2")

                    if hasattr(state.engine, "_embedding_function") and state.engine._embedding_function:
                        model_name = "Unknown"
                        if hasattr(state.engine._embedding_function, "name"):
                            model_name = state.engine._embedding_function.name
                        ui.label(f"Model: {model_name}").classes("text-xs text-gray-400")

                        if hasattr(state.engine, "_embed_dim"):
                            ui.label(f"Dimensions: {state.engine._embed_dim}").classes("text-xs text-gray-400")
                except Exception as e:
                    logger.debug(f"Could not display engine stats: {e}")

            engine_stats_display()
            page_state.refresh_context_engine = engine_stats_display.refresh

            ui.separator().classes("my-2")

            async def update_engine_db():
                """Reload chunks into the engine."""
                if state.context and state.chunker and state.engine:
                    update_btn.props("loading")
                    status_label.set_text("Chunking messages...")

                    try:
                        messages = state.context.context
                        status_label.set_text(f"Chunking {len(messages)} messages...")

                        # Run chunking in thread pool to avoid blocking
                        import asyncio
                        chunks = await asyncio.to_thread(
                            state.chunker.chunk_messages, messages
                        )

                        status_label.set_text(f"Embedding {len(chunks)} chunks...")

                        # Clear existing chunks
                        def clear_and_add():
                            if hasattr(state.engine, "_context"):
                                state.engine._context = []
                            elif hasattr(state.engine, "_chunks"):
                                state.engine._chunks = []
                                state.engine._embeddings = []
                            state.engine.add_context(chunks)

                        # Run embedding in thread pool to avoid blocking
                        await asyncio.to_thread(clear_and_add)

                        # Reset pagination to first page
                        state.display.vectordb_page = 0

                        page_state.refresh_context_engine()
                        page_state.refresh_context()
                        status_label.set_text("")
                        ui.notify(f"Updated {len(chunks)} chunks in engine", type="positive")
                        logger.info(f"Engine updated with {len(chunks)} chunks")
                    except Exception as ex:
                        status_label.set_text("")
                        ui.notify(f"Failed to update engine: {ex}", type="negative")
                        logger.error(f"Engine update failed: {ex}")
                    finally:
                        update_btn.props(remove="loading")

            update_btn = ui.button("Update Database", on_click=update_engine_db, icon="sync").props("color=green").classes("w-full")
            status_label = ui.label("").classes("text-xs text-gray-400 mt-1")

            ui.separator().classes("my-2")

            def clear_engine_db():
                """Clear all stored data from the engine."""
                if state.engine:
                    state.engine.clear()
                    state.display.vectordb_page = 0
                    page_state.refresh_context_engine()
                    page_state.refresh_context()
                    ui.notify("Database cleared", type="info")
                    logger.info("Engine database cleared")

            with ui.dialog() as confirm_dialog, ui.card():
                ui.label("Clear Database?").classes("text-lg font-semibold")
                ui.label("This will remove all stored chunks and embeddings.").classes("text-gray-400")
                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("Cancel", on_click=confirm_dialog.close).props("flat")
                    ui.button("Clear", on_click=lambda: (clear_engine_db(), confirm_dialog.close())).props("color=red")

            ui.button("Clear Database", on_click=confirm_dialog.open, icon="delete").props("color=red outline").classes("w-full")
        else:
            ui.label("No engine loaded").classes("text-gray-500 italic text-sm")


def create_llm_controls_card(state: AppStateProtocol) -> None:
    """
    Create the LLM generation controls card.

    Args:
        state: Application state with llm_kwargs
    """
    with ui.card().classes("w-full mt-4"):
        ui.label(f"Model: {state.current_model_name}").classes("text-xs text-purple-500 italic mb-2")
        render_model_controls(
            model=state.llm_kwargs,
            title="LLM Generation",
        )


def create_display_controls_card(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
) -> None:
    """
    Create the display settings card.

    Args:
        state: Application state with display settings
        page_state: Page state with refresh callbacks
    """
    with ui.card().classes("w-full mt-4"):

        def on_display_change() -> None:
            page_state.refresh_chat()
            page_state.refresh_context()

        render_model_controls(
            model=state.display,
            title="Display",
            on_change=on_display_change,
        )


def create_history_controls_card(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
) -> None:
    """
    Create the chat history controls card.

    Args:
        state: Application state with messages
        page_state: Page state with refresh callbacks
    """
    with ui.card().classes("w-full mt-4"):
        with ui.row().classes("w-full justify-between items-center"):
            history_label = ui.label()
            history_label.bind_text_from(
                state,
                "messages",
                backward=lambda msgs: f"History: {len(msgs)}",
            )

            def clear_history():
                state.messages = []
                page_state.refresh_chat()
                ui.notify("History cleared", type="info")

            ui.button("Clear", on_click=clear_history).props("flat dense")


def create_model_selector_card(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
    get_llm_wrapper: Callable[[Config], Optional[Any]],
) -> None:
    """
    Create the model selector and streaming controls card.

    Args:
        state: Application state
        page_state: Page state with refresh callbacks
        get_llm_wrapper: Function to create LLM wrapper from config
    """
    with ui.card().classes("w-full mt-4"):

        @ui.refreshable
        def model_selector_container():
            def on_model_select(idx: int, name: str, config: Config):
                state.selected_model_idx = idx
                state.current_model_name = name

                loading_notification = ui.notification(f"Loading {name}...", spinner=True, timeout=None)
                page_state.llm_wrapper = get_llm_wrapper(config)
                loading_notification.dismiss()

                if page_state.llm_wrapper:
                    ui.notify(f"Model {name} ready", type="positive")
                else:
                    ui.notify(f"Failed to load {name}", type="negative")

            create_model_selector(
                config=state.config,
                selected_idx=state.selected_model_idx,
                current_model_name=state.current_model_name,
                on_select=on_model_select,
            )

        model_selector_container()

        ui.separator().classes("my-2")

        render_model_controls(model=state.streaming)
        ui.label("Streaming disables reasoning/CoT").classes("text-xs text-gray-500 italic")

        ui.separator().classes("my-2")

        @ui.refreshable
        def usage_container():
            create_usage_stats(state.usage_maps)

        usage_container()
        page_state.refresh_usage = usage_container.refresh


def create_about_expansion() -> None:
    """Create the about/help expansion panel."""
    with ui.expansion("About", icon="info").classes("w-full mt-4"):
        ui.markdown("""
**Chat with your data using RAG**

1. Load a chat export file
2. Adjust context parameters
3. Toggle chunk preview to visualize chunking
4. Ask questions about the conversation
5. See highlighted context used for answers

**Streaming**: Enable for real-time responses (when supported by model)
        """)


def create_sidebar(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
    token_counter: Callable[[str], int],
    context_class: type,
    apply_context_params: Callable[[AppStateProtocol], None],
    get_llm_wrapper: Callable[[Config], Optional[Any]],
) -> None:
    """
    Create the complete sidebar with all control panels.

    This is the main entry point for creating a sidebar. It composes
    all the individual cards into a cohesive control panel.

    Args:
        state: Application state
        page_state: Page state with refresh callbacks
        token_counter: Function to count tokens
        context_class: Class for chat context
        apply_context_params: Function to apply context parameters
        get_llm_wrapper: Function to create LLM wrapper
    """
    ui.label("Control Panel").classes("text-xl font-bold mb-4")

    create_chat_loader_card(state, page_state, token_counter, context_class, apply_context_params)
    create_context_params_card(state, page_state, apply_context_params)
    create_vectordb_card(state, page_state)
    create_llm_controls_card(state)
    create_display_controls_card(state, page_state)
    create_history_controls_card(state, page_state)
    create_model_selector_card(state, page_state, get_llm_wrapper)
    create_about_expansion()
