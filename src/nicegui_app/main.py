# Created by Nir Ben-Zvi
# NiceGUI Chat Application
#
# A chat interface for interacting with conversation data using RAG.
# Main entry point that composes modular components.

# pyright: reportOptionalMemberAccess=false

# Load environment variables first
from dotenv import load_dotenv

load_dotenv(override=True)

# Apply SSL bypass early, before any network-related imports
from workshop.ssl_utils import apply_ssl_bypass

apply_ssl_bypass()

from typing import Any, Callable, Dict, Optional

import hydra
from loguru import logger
from nicegui import app, ui
from omegaconf import DictConfig, OmegaConf

from nicegui_app.chat_view import create_chat_view
from nicegui_app.context_view import create_context_view
from nicegui_app.llm_ui_utils import Config, get_embedder, get_llm_wrapper
from nicegui_app.sidebar import create_sidebar
from nicegui_app.state import (
    AppState,
    ChatMessage,
    extract_llm_response,
    get_or_create_state,
)
from workshop.chat import ChatContext, naive_token_counter
from workshop.structured_types import RAGResponse


class PageState:
    """
    Holds page-scoped state: refresh callbacks and runtime objects.

    NiceGUI's @ui.refreshable creates functions that need to be called
    to re-render. This class lets us wire them up after creation.

    Also holds runtime objects (like LLM wrapper) that shouldn't be persisted.
    """

    def __init__(self):
        self.refresh_context: Callable[[], object] = lambda: None
        self.refresh_context_engine: Callable[[], object] = lambda: None
        self.refresh_chat: Callable[[], object] = lambda: None
        self.refresh_usage: Callable[[], object] = lambda: None
        self.refresh_context_stats: Callable[[], object] = lambda: None
        self.llm_wrapper: Optional[Any] = None


def apply_context_params(state) -> None:  # type: ignore[no-untyped-def]
    """
    Apply current context parameters to the loaded context.

    Uses chunker.params for max_tokens and max_days settings.

    Args:
        state: Application state with context and chunker
    """
    if state.context is None or state.chunker is None:
        return

    params = state.chunker.params

    if params.max_tokens > 0 and params.max_days <= 0:
        state.context.get_context(token_limit=params.max_tokens, num_days=-1)
    elif params.max_days > 0:
        state.context.get_context(token_limit=-1, num_days=params.max_days)


def _create_llm_wrapper(model_config: Config) -> Optional[object]:
    """Create LLM wrapper with structured output type."""
    return get_llm_wrapper(model_config, RAGResponse)


def _get_embedder_kwargs(cls: type, state: AppState) -> Dict[str, Any]:
    """
    Get embedder kwarg if class requires it (declarative dependency injection).

    Classes declare their dependency via `requires_embedder = True` class attribute.
    This enables Open/Closed principle: new components declare needs without
    modifying initialization code.

    Args:
        cls: Component class to check for embedder requirement
        state: Application state with config for embedder creation

    Returns:
        Dict with 'embedder' key if required, empty dict otherwise

    Raises:
        ValueError: If embedder is required but creation fails
    """
    if not getattr(cls, "requires_embedder", False):
        return {}

    embedding_llm_config = state.config.get("models", {}).get("embedding_llm", {})
    embedder = get_embedder(embedding_llm_config)

    if embedder is None:
        raise ValueError(f"{cls.__name__} requires embedder but failed to create one")

    return {"embedder": embedder}


def initialize_workshop_components(state: AppState) -> None:
    """
    Initialize chunker and engine from workshop_config.

    Participants edit workshop_config.py to select which chunker and engine
    implementations to use. This function instantiates them and attaches to state.

    Uses declarative dependency injection: components declare `requires_embedder = True`
    class attribute if they need an embedder, eliminating string-based type checking.

    Args:
        state: Application state to populate with chunker and engine
    """
    try:
        from nicegui_app import workshop_config

        # Initialize chunker if not already set
        if state.chunker is None:
            chunker_cls = workshop_config.CHUNKER_CLASS
            chunker_kwargs = getattr(workshop_config, "CHUNKER_KWARGS", {})

            embedder_kwargs = _get_embedder_kwargs(chunker_cls, state)
            state.chunker = chunker_cls(**embedder_kwargs, **chunker_kwargs)

        # Initialize engine if not already set
        if state.engine is None:
            engine_cls = workshop_config.ENGINE_CLASS
            engine_kwargs = getattr(workshop_config, "ENGINE_KWARGS", {})

            embedder_kwargs = _get_embedder_kwargs(engine_cls, state)
            state.engine = engine_cls(**embedder_kwargs, **engine_kwargs)

        logger.info(f"Workshop components initialized: {type(state.chunker).__name__}, {type(state.engine).__name__}")

    except ImportError as e:
        logger.warning(f"Failed to import workshop_config: {e}")
    except ValueError as e:
        logger.error(f"Failed to initialize workshop components: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize workshop components: {e}")


def create_chat_page(single_client: bool = False) -> None:
    """Register the main chat page route."""
    connected_clients: set[str] = set()

    @ui.page("/")
    async def chat_page():
        client = ui.context.client

        # Single-client mode: reject additional connections
        if single_client:
            if connected_clients:
                ui.label("App is already open in another tab.").classes("text-xl text-red-400 m-8")
                return
            connected_clients.add(client.id)
            client.on_disconnect(lambda: connected_clients.discard(client.id))

        config: Config = app.state.config
        state = get_or_create_state(app.storage.user, config)
        page_state = PageState()

        initialize_workshop_components(state)

        # Reconstruct context from persisted chat_path or use initial_context
        if state.context is None:
            if state.chat_path:
                try:
                    state.context = ChatContext(
                        chat_path=state.chat_path,
                        token_counter=naive_token_counter,
                    )
                    apply_context_params(state)
                except Exception as e:
                    logger.warning(f"Failed to load context from {state.chat_path}: {e}")
                    state.chat_path = ""
            elif hasattr(app.state, "initial_context"):
                state.context = app.state.initial_context
                apply_context_params(state)

        # Header
        with ui.header().classes("bg-gray-900 items-center justify-between px-4"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("chat").classes("text-2xl text-blue-400")
                ui.label("Retrieval Playground").classes("text-xl font-bold")
            ui.label("Chat With Your Data").classes("text-gray-400")

        # Sidebar
        with ui.left_drawer(value=True).classes("bg-gray-800 p-4") as drawer:
            create_sidebar(
                state=state,
                page_state=page_state,
                token_counter=naive_token_counter,
                context_class=ChatContext,
                apply_context_params=apply_context_params,
                get_llm_wrapper=_create_llm_wrapper,
            )

        # Menu toggle
        with ui.page_sticky(position="top-left").classes("m-2 ml-4"):
            ui.button(icon="menu", on_click=drawer.toggle).props("flat round color=white")

        # Main content
        with ui.row().classes("w-full h-[calc(100vh-8rem)] p-4 gap-4"):
            with ui.column().classes("flex-1 h-full min-w-0 overflow-hidden"):
                create_chat_view(
                    state=state,
                    page_state=page_state,
                    chat_message_class=ChatMessage,
                    extract_llm_response=extract_llm_response,
                )

            with ui.column().classes("flex-1 h-full min-w-0 overflow-hidden"):
                create_context_view(state, page_state)

        # Footer
        with ui.footer().classes("bg-gray-900 text-gray-500 text-xs py-2"):
            ui.label("Built with NiceGUI + Hydra")


def _patch_json_serializer() -> None:
    """Patch NiceGUI's JSON serializer to support Pydantic models."""
    from nicegui.json import orjson_wrapper
    from pydantic import BaseModel

    original = orjson_wrapper._orjson_converter

    def patched_converter(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        return original(obj)

    orjson_wrapper._orjson_converter = patched_converter


@hydra.main(version_base=None, config_path="../../configs", config_name="main_app_cfg")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the NiceGUI chat application.

    Uses standard Hydra configuration.

    Run with:
        python -m nicegui_app.main

    Override configs with:
        python -m nicegui_app.main paths.chat_export_path=/path/to/chat.txt
    """
    from genai_prices import wait_prices_updated_sync

    if not wait_prices_updated_sync(timeout=10):
        logger.error("Failed to update genai-prices data snapshot")

    _patch_json_serializer()

    config: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    logger.info("=" * 50)
    logger.info("Starting LLM NiceGUI app")
    logger.info("=" * 50)
    app.state.config = config

    import os

    paths = config.get("paths", {})
    chat_path = os.getenv("CHAT_FILE_PATH") or paths.get("chat_export_path")

    if chat_path:
        logger.info(f"Pre-loading chat context from: {chat_path}")
        app.state.initial_context = ChatContext(
            chat_path=chat_path,
            token_counter=naive_token_counter,
        )

    single_client = config.get("app", {}).get("single_client", False)
    create_chat_page(single_client=single_client)

    def cleanup():
        logger.info("Shutting down gracefully...")

    app.on_shutdown(cleanup)

    try:
        ui.run(
            title="Retrieval Playground - Chat With Your Data",
            storage_secret="retrieval-playground-nicegui-secret",
            dark=True,
            reload=False,
            show=False,  # Don't auto-open browser; participants click the link
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, exiting...")


if __name__ == "__main__":
    main()
