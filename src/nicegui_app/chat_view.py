# Created by Nir Ben-Zvi
# Chat view components for NiceGUI applications
#
# Modular chat interface with message rendering, streaming support,
# and LLM response handling. Can be reused across different NiceGUI apps.

# pyright: reportOptionalMemberAccess=false

import asyncio
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from loguru import logger
from nicegui import ui

from nicegui_app.llm_ui_utils import get_direction_classes, get_text_direction


class ChatMessage(Protocol):
    """Protocol for chat message objects."""

    role: str
    content: str
    is_streaming: bool
    retrieved_message_ids: Optional[List[int]]

    def get_raw_output_display(self) -> Optional[str]: ...
    def to_dict(self) -> Dict[str, str]: ...

    @classmethod
    def user(cls, content: str) -> "ChatMessage": ...

    @classmethod
    def assistant(
        cls,
        content: str,
        raw_output: Any = None,
        retrieved_message_ids: Optional[List[int]] = None,
    ) -> "ChatMessage": ...

    @classmethod
    def streaming_placeholder(cls) -> "ChatMessage": ...


class RefreshablePageState(Protocol):
    """Protocol for page state with refresh callbacks."""

    refresh_context: Callable[[], object]
    refresh_chat: Callable[[], object]
    refresh_usage: Callable[[], object]
    llm_wrapper: Optional[Any]


class AppStateProtocol(Protocol):
    """Protocol for application state used by chat view."""

    messages: List[Any]
    current_query: str
    context: Any
    streaming: Any
    llm_kwargs: Any
    current_model_name: str
    usage_maps: Any
    display: Any
    engine: Any


def render_typing_indicator() -> None:
    """Render animated typing indicator (three bouncing dots)."""
    ui.add_css("""
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 4px 0;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #60a5fa;
            border-radius: 50%;
            animation: typing-bounce 1.4s infinite ease-in-out both;
        }
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0s; }
        @keyframes typing-bounce {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
    """)
    with ui.element("div").classes("typing-indicator"):
        ui.element("span")
        ui.element("span")
        ui.element("span")


def render_message(msg: Any, rtl_mode: str) -> None:
    """
    Render a single chat message with proper RTL support and reasoning display.

    Args:
        msg: ChatMessage object with role, content, is_streaming, and get_reasoning()
        rtl_mode: RTL display mode ("auto", "rtl", "ltr")
    """
    is_user = msg.role == "user"
    direction = get_text_direction(msg.content, rtl_mode)
    dir_classes = get_direction_classes(direction)

    with ui.chat_message(sent=is_user, name="You" if is_user else "Assistant"):
        with (
            ui.element("div")
            .classes(f"{dir_classes} break-words whitespace-pre-wrap max-w-full")
            .props(f'dir="{direction}"')
        ):
            if msg.is_streaming:
                if msg.content:
                    ui.markdown(msg.content + " ▌")
                else:
                    render_typing_indicator()
            else:
                ui.markdown(msg.content)

        # Show full structured output for assistant messages (when available)
        if not is_user and (raw_output_md := msg.get_raw_output_display()):
            with ui.expansion("Full Response", icon="data_object").classes("text-xs mt-1"):
                ui.markdown(raw_output_md).classes("break-words whitespace-pre-wrap")


def get_message_history(messages: Sequence[Any]) -> List[Dict[str, str]]:
    """
    Convert ChatMessages to minimal dict format for LLM history.

    Args:
        messages: Sequence of ChatMessage objects

    Returns:
        List of dicts with role and content keys
    """
    return [msg.to_dict() for msg in messages]


def retrieve_context(engine: Any, query: str, top_k: int = 10) -> tuple[str, List[int]]:
    """
    Retrieve relevant context for the query using the context engine.

    Args:
        engine: Context engine with get_relevant_context method
        query: User query text
        top_k: Maximum number of chunks to retrieve

    Returns:
        Tuple of (context_text, message_ids):
        - context_text: The text to pass to the LLM as context
        - message_ids: List of message indices for exact highlighting

    Raises:
        RuntimeError: If retrieval fails
    """
    try:
        chunks = engine.get_relevant_context(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        raise RuntimeError(f"Failed to retrieve context: {e}") from e

    if not chunks:
        logger.warning("No chunks retrieved for query")
        return "", []

    context_text = "\n\n---\n\n".join(chunk.text for chunk in chunks)

    all_message_ids: List[int] = []
    for chunk in chunks:
        # Handle chunks that may not have message_ids attribute
        chunk_ids = getattr(chunk, "message_ids", None)
        if chunk_ids:
            all_message_ids.extend(chunk_ids)

    unique_message_ids = sorted(set(all_message_ids))

    logger.info(f"Retrieved {len(chunks)} chunks covering {len(unique_message_ids)} messages")
    return context_text, unique_message_ids


def _handle_llm_error(error: Exception, state: AppStateProtocol, message_list: Any) -> None:
    """
    Handle LLM errors with user-friendly notifications.

    Args:
        error: The exception that occurred
        state: Application state (to remove placeholder message)
        message_list: Message list UI element to refresh
    """
    state.messages = state.messages[:-1]
    message_list.refresh()

    error_msg = str(error)
    if isinstance(error, RuntimeError):
        ui.notify(error_msg, type="negative")
    if "rate limit" in error_msg.lower() or ("tokens" in error_msg.lower() and "exceed" in error_msg.lower()):
        # Show detailed rate limit guidance
        with ui.dialog() as rate_limit_dialog, ui.card().classes("w-[500px]"):
            ui.label("Rate Limit Reached").classes("text-lg font-semibold text-red-500")
            ui.markdown("""
**The LLM API rate limit was exceeded.**

This usually happens when:
- Message history is too long (many conversation turns)
- Retrieved context is very large
- Multiple requests in quick succession

**Recommendations:**
1. **Clear chat history** using the "Clear" button in the History section
2. Wait a moment before trying again
3. If using a large chat export, consider using a smaller file
4. Reduce the number of retrieved chunks (top_k setting)
            """).classes("text-sm")
            ui.button("OK", on_click=rate_limit_dialog.close).props("color=primary")
        rate_limit_dialog.open()
    elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
        ui.notify("API authentication error. Check your API keys.", type="negative")
    else:
        ui.notify(f"LLM error: {error_msg[:100]}", type="negative")


async def handle_llm_response(
    state: AppStateProtocol,
    query: str,
    message_list: Any,
    scroll: Any,
    page_state: RefreshablePageState,
    chat_message_class: type,
    extract_llm_response: Callable[[Any], str],
    use_streaming: bool = False,
) -> None:
    """
    Handle LLM response (streaming or batch mode).

    Unified handler that supports both streaming and batch responses.
    Streaming mode uses output_type=str for token-by-token display.
    Batch mode uses structured output for reasoning/CoT.

    Args:
        state: Application state
        query: User query text
        message_list: Refreshable message list UI element
        scroll: Scroll area UI element
        page_state: Page state with LLM wrapper
        chat_message_class: Class to create chat messages
        extract_llm_response: Function to extract content from structured output
        use_streaming: Whether to use streaming mode
    """
    placeholder = chat_message_class.streaming_placeholder()
    state.messages = state.messages + [placeholder]
    message_list.refresh()
    scroll.scroll_to(percent=1.0)

    try:
        history = get_message_history(state.messages[:-1])
        model_settings = {k: v for k, v in state.llm_kwargs.model_dump(mode="json").items() if v is not None}
        context_text, retrieved_message_ids = retrieve_context(state.engine, query)

        if use_streaming:
            async with page_state.llm_wrapper.run_stream(
                query,
                output_type=str,
                deps={"context": context_text},
                message_history=history,
                model_settings=model_settings,
            ) as stream:
                async for cumulative_text in stream.stream_text(delta=False):
                    state.messages[-1].content = cumulative_text
                    message_list.refresh()
                    scroll.scroll_to(percent=1.0)

                    if state.streaming.chunk_delay_ms > 0:
                        await asyncio.sleep(state.streaming.chunk_delay_ms / 1000)

                final_content = await stream.get_output()
                usage = stream.usage()

            msg = chat_message_class.assistant(
                content=final_content,
                raw_output=None,
                retrieved_message_ids=retrieved_message_ids,
            )
        else:
            result = await page_state.llm_wrapper.run(
                query,
                deps={"context": context_text},
                message_history=history,
                model_settings=model_settings,
            )
            content = extract_llm_response(result.output)
            usage = result.usage()

            msg = chat_message_class.assistant(
                content=content,
                raw_output=result.output,
                retrieved_message_ids=retrieved_message_ids,
            )

        state.messages[-1] = msg

        if usage:
            state.usage_maps[state.current_model_name].append(usage)
            page_state.refresh_usage()

        message_list.refresh()
        page_state.refresh_context()
        scroll.scroll_to(percent=1.0)

    except Exception as e:
        _handle_llm_error(e, state, message_list)
        logger.exception("LLM response error")
        raise


def create_chat_view(
    state: AppStateProtocol,
    page_state: RefreshablePageState,
    chat_message_class: type,
    extract_llm_response: Callable[[Any], str],
) -> None:
    """
    Create the chat interface with message history and input.

    Args:
        state: Application state
        page_state: Page state with refresh callbacks
        chat_message_class: Class for creating chat messages
        extract_llm_response: Function to extract content from LLM output
    """
    with ui.card().classes("w-full h-full flex flex-col"):
        ui.label("Chat").classes("text-xl font-bold mb-2 flex-none")

        with ui.scroll_area().classes("flex-grow w-full overflow-x-hidden") as scroll:

            @ui.refreshable
            def message_list():
                for msg in state.messages:
                    render_message(msg, state.display.rtl_mode)

            message_list()
            page_state.refresh_chat = message_list.refresh

        ui.separator().classes("flex-none")

        async def handle_send():
            query = state.current_query
            if not query.strip():
                return

            state.current_query = ""
            state.messages = state.messages + [chat_message_class.user(query)]
            message_list.refresh()
            scroll.scroll_to(percent=1.0)

            if not state.context:
                ui.notify("No context loaded", type="warning")
                return

            if not page_state.llm_wrapper:
                ui.notify("Please select a model first", type="warning")
                return

            if not state.engine:
                ui.notify("No retrieval engine configured", type="warning")
                return

            try:
                await handle_llm_response(
                    state=state,
                    query=query,
                    message_list=message_list,
                    scroll=scroll,
                    page_state=page_state,
                    chat_message_class=chat_message_class,
                    extract_llm_response=extract_llm_response,
                    use_streaming=state.streaming.enabled,
                )

            except Exception:
                # Error already logged and notified in handler functions
                pass

        with ui.row().classes("w-full items-center gap-2 flex-none"):
            query_input = ui.input(
                placeholder="Ask something about this chat...",
            ).classes("flex-grow")
            query_input.bind_value(state, "current_query")
            query_input.on("keydown.enter", handle_send)

            ui.button(icon="send", on_click=handle_send).props("flat round")
