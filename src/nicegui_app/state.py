# Created by Nir Ben-Zvi
# NiceGUI Chat Application State
#
# State is organized into logical groups:
# - ContextParams: Parameters for context window and chunking
# - DisplayConfig: UI display preferences (RTL, view modes)
# - StreamingConfig: LLM streaming behavior
# - ChatMessage: Individual chat messages with structured response support
# - AppState: Full application state including UI and runtime objects

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import RunUsage

from workshop.structured_types import RetrievalCoT

# Config type - plain dict after OmegaConf.to_container()
Config = Dict[str, Any]

# Type alias for chunk coordinates: list of (start, end) character indices
ChunkCoordinates = List[Tuple[int, int]]

# Type alias for chunk provider callable
# Takes full text, returns list of (start, end) index tuples
ChunkProviderFn = Callable[[str], ChunkCoordinates]


class ChunkProvider(Protocol):
    """
    Protocol for custom chunk providers (for static typing).

    Implement this to provide custom chunking logic that bypasses UI controls.
    The visualization will use these coordinates regardless of UI settings.

    Note: For Pydantic field compatibility, the actual field type is Any.
    This protocol is for type checking only.
    """

    def get_chunks(self, text: str) -> ChunkCoordinates:
        """
        Compute chunk boundaries for the given text.

        Args:
            text: The full document text

        Returns:
            List of (start, end) tuples representing chunk boundaries
        """
        ...


class LLMKwargs(BaseModel):
    """LLM generation parameters."""

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        json_schema_extra={"step": 0.05, "label": "Temperature", "tooltip": "Higher = more creative"},
    )
    max_output_tokens: int = Field(
        default=4096,
        ge=512,
        le=65536,
        json_schema_extra={"step": 256, "label": "Max Tokens", "tooltip": "Response length limit"},
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        json_schema_extra={"step": 0.05, "label": "Top P", "tooltip": "Nucleus sampling threshold"},
    )


# DEPRECATED: ContextParams removed - conversation windowing moved to chunker.params
# LLM parameters moved to top-level AppState.llm_kwargs


class StreamingConfig(BaseModel):
    """
    Configuration for LLM streaming behavior.

    Attributes:
        enabled: Whether to use streaming responses when available
        chunk_delay_ms: Artificial delay between chunks for smoother display (0 = no delay)
    """

    enabled: bool = Field(
        default=False,
        json_schema_extra={"label": "Enable Streaming", "tooltip": "Stream tokens as they arrive"},
    )
    chunk_delay_ms: int = Field(
        default=0,
        ge=0,
        le=500,
        json_schema_extra={
            "label": "Chunk Delay (ms)",
            "tooltip": "Artificial delay for smoother display",
            "hidden": True,
        },
    )


class DisplayConfig(BaseModel):
    """
    UI display preferences.

    Attributes:
        rtl_mode: Text direction mode (auto-detect, force RTL, force LTR)
        context_view_mode: How to display context (default, chunks, highlights, vectordb)
            Note: This field is excluded from sidebar rendering since it has a dedicated
            dropdown selector at the top of the context panel.
        vectordb_page: Current page in vectordb view (0-indexed)
        vectordb_page_size: Number of chunks per page in vectordb view
    """

    rtl_mode: Literal["auto", "rtl", "ltr"] = Field(
        default="auto",
        json_schema_extra={"label": "Text Direction"},
    )
    context_view_mode: Literal["default", "chunks", "highlights", "vectordb"] = Field(
        default="default",
        json_schema_extra={"label": "Context View", "exclude_from_form": True},
    )
    vectordb_page: int = Field(
        default=0,
        ge=0,
        json_schema_extra={"exclude_from_form": True},
    )
    vectordb_page_size: int = Field(
        default=20,
        ge=5,
        le=100,
        json_schema_extra={"exclude_from_form": True},
    )


class ChatMessage(BaseModel):
    """
    A chat message with display content and optional structured data.

    This provides a clean interface for both user and assistant messages,
    with support for structured LLM responses.

    Attributes:
        role: "user" or "assistant"
        content: The display content (final answer for assistant)
        raw_output: The full structured output from LLM (for inspection/debugging)
        context_used: Context chunks used (for highlighting via text matching - fallback)
        retrieved_message_ids: Message indices from retrieval (for exact highlighting - preferred)
        is_streaming: Whether this message is currently being streamed
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: str  # "user" or "assistant"
    content: str
    raw_output: Optional[RetrievalCoT] = None
    context_used: Optional[str] = None
    retrieved_message_ids: Optional[List[int]] = None  # Exact indices from retrieval
    is_streaming: bool = Field(default=False, exclude=True)  # Runtime only

    @field_validator("raw_output", mode="before")
    @classmethod
    def validate_raw_output(cls, v: Any) -> Optional[RetrievalCoT]:
        """
        Reconstruct RetrievalCoT from dict after JSON serialization.

        When ChatMessage is serialized to JSON (via NiceGUI storage or transmission),
        raw_output becomes a plain dict. This validator reconstructs the Pydantic model.
        """
        if v is None:
            return None
        if isinstance(v, RetrievalCoT):
            return v
        if isinstance(v, dict):
            return RetrievalCoT.model_validate(v)
        return v

    def get_reasoning(self) -> Optional[str]:
        """Get the full reasoning/chain-of-thought if available."""
        if self.raw_output is None:
            return None
        if hasattr(self.raw_output, "to_message"):
            return self.raw_output.to_message()
        return str(self.raw_output)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for message history (minimal format for LLM)."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls,
        content: str,
        raw_output: Optional[RetrievalCoT] = None,
        context_used: Optional[str] = None,
        retrieved_message_ids: Optional[List[int]] = None,
    ) -> ChatMessage:
        """Create an assistant message from LLM response."""
        return cls(
            role="assistant",
            content=content,
            raw_output=raw_output,
            context_used=context_used,
            retrieved_message_ids=retrieved_message_ids,
        )

    @classmethod
    def streaming_placeholder(cls) -> ChatMessage:
        """Create a placeholder for streaming response."""
        return cls(role="assistant", content="", is_streaming=True)


def extract_llm_response(response_output: Any) -> tuple[str, Optional[str]]:
    """
    Extract display content and context_used from any LLM response type.

    Handles (in order of priority):
    - Plain strings
    - BaseModel with .output attribute (OneStageAnswer, etc.)
    - Nested structures with .output.output (RetrievalResult inside ComplexAnswer)
    - context_used field for highlighting (from RetrievalResult or RAG)

    Fallback chain:
    1. to_message() method if available
    2. JSON serialization for Pydantic models
    3. str() as last resort

    Returns:
        Tuple of (display_content, context_used or None)
    """
    context_used = None

    # Extract context_used from response or nested output (for highlighting)
    # Check both field names for compatibility
    if hasattr(response_output, "context_used"):
        context_used = response_output.context_used

    # Extract the final answer
    if isinstance(response_output, str):
        return response_output, context_used

    # Handle nested .output (ComplexAnswer style)
    if hasattr(response_output, "output"):
        inner = response_output.output

        # Check nested output for context_used (RetrievalResult inside ComplexAnswer)
        if context_used is None:
            if hasattr(inner, "context_used"):
                context_used = inner.context_used

        # Extract the actual answer
        if hasattr(inner, "output"):
            # RetrievalResult.output
            return inner.output, context_used
        elif hasattr(inner, "answer"):
            return inner.answer, context_used
        elif isinstance(inner, str):
            return inner, context_used
        else:
            # Inner object doesn't have output/answer - try JSON then str
            return _format_unknown_response(inner), context_used

    # Fallback chain: to_message() -> JSON -> str()
    if hasattr(response_output, "to_message"):
        return response_output.to_message(), context_used

    return _format_unknown_response(response_output), context_used


def _format_unknown_response(response: Any) -> str:
    """
    Format an unknown response type for display.

    Tries JSON serialization first (for Pydantic models and dicts),
    falls back to str() representation.

    Args:
        response: Unknown response object

    Returns:
        Formatted string representation
    """
    import json

    # Try Pydantic model serialization first
    if hasattr(response, "model_dump_json"):
        try:
            return response.model_dump_json(indent=2)
        except Exception:
            pass

    # Try Pydantic v1 style
    if hasattr(response, "json"):
        try:
            return response.json(indent=2)
        except Exception:
            pass

    # Try dict-like objects
    if hasattr(response, "model_dump"):
        try:
            return json.dumps(response.model_dump(), indent=2, default=str)
        except Exception:
            pass

    # Try regular dict
    if isinstance(response, dict):
        try:
            return json.dumps(response, indent=2, default=str)
        except Exception:
            pass

    # Last resort: str()
    return str(response)


class AppState(BaseModel):
    """
    Application state for NiceGUI chat app.

    This replaces ApplicationCache entirely. NiceGUI's app.storage.user
    persists this across page refreshes automatically.

    Pydantic handles serialization/deserialization via model_validate().
    Runtime-only fields are marked with exclude=True.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Runtime only (not persisted)
    config: Optional[Config] = Field(default=None, exclude=True)
    context: Optional[Any] = Field(default=None, exclude=True)  # ChatContext - reconstructed from chat_path
    chunk_provider: Optional[Any] = Field(default=None, exclude=True)  # ChunkProvider | ChunkProviderFn

    # Workshop components (runtime only)
    chunker: Optional[Any] = Field(default=None, exclude=True)  # Instance of CHUNKER_CLASS from workshop_config
    engine: Optional[Any] = Field(default=None, exclude=True)  # Instance of ENGINE_CLASS from workshop_config

    # Persisted
    chat_path: str = ""
    llm_kwargs: LLMKwargs = Field(default_factory=LLMKwargs)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)

    messages: List[ChatMessage] = Field(default_factory=list)
    usage_maps: Dict[str, List[RunUsage]] = Field(default_factory=lambda: defaultdict(list))

    selected_model_idx: int = 0
    model_configs: List[Config] = Field(default_factory=list)
    current_model_name: str = "None"

    current_query: str = ""

    @property
    def has_custom_chunker(self) -> bool:
        """
        Whether a custom chunk provider is active (disables UI chunking controls).

        Deprecated: Use state.chunker instead. This is kept for backward compatibility.
        """
        return self.chunk_provider is not None or self.chunker is not None

    def get_chunk_boundaries(self, num_messages: int) -> ChunkCoordinates:
        """
        Get chunk boundaries for preview visualization.

        Delegates to the active chunker if available. This method is used by
        the GUI for fast chunk preview without creating full ChunkObjects.

        Args:
            num_messages: Total number of messages in conversation

        Returns:
            List of (start_idx, end_idx) tuples for each chunk
        """
        if self.chunker is not None:
            return self.chunker.get_chunk_boundaries(num_messages)

        if self.chunk_provider is not None:
            provider = self.chunk_provider
            if hasattr(provider, "get_chunk_boundaries"):
                return provider.get_chunk_boundaries(num_messages)
            elif hasattr(provider, "get_chunks"):
                return cast(ChunkCoordinates, provider.get_chunks(""))

        # Fallback: use default MessageCountParams values if no chunker is set
        # This should rarely happen - chunker should always be initialized
        chunk_length = max(1, 6)  # Default from MessageCountParams
        overlap = max(0, min(2, chunk_length - 1))  # Default from MessageCountParams
        stride = chunk_length - overlap

        chunks: ChunkCoordinates = []
        i = 0
        while i < num_messages:
            end = min(i + chunk_length, num_messages)
            chunks.append((i, end))
            i += stride
            if i >= num_messages:
                break

        return chunks


def get_or_create_state(storage: Dict[str, Any], config: Config) -> AppState:
    """
    Get existing state from storage or create new one.

    Pydantic's model_validate() handles deserialization from dict automatically.

    Args:
        storage: NiceGUI storage dict (app.storage.user)
        config: Hydra configuration

    Returns:
        AppState instance
    """
    if "app_state" not in storage:
        state = AppState()
        storage["app_state"] = state
    else:
        stored = storage["app_state"]
        if isinstance(stored, AppState):
            state = stored
        else:
            # Deserialize from dict - Pydantic handles this automatically
            state = AppState.model_validate(stored)
            storage["app_state"] = state

    # Inject runtime objects (excluded from persistence)
    state.config = config
    state.context = None  # Will be reconstructed from chat_path in chat_page()

    # Ensure usage_maps is a defaultdict (lost during JSON serialization)
    if not isinstance(state.usage_maps, defaultdict):
        state.usage_maps = defaultdict(list, state.usage_maps)

    return state


def update_chunker_param(
    state: AppState,
    key: str,
    value: Any,
    on_change: Optional[Callable[[], None]] = None,
) -> None:
    """
    Update a chunker parameter and trigger refresh if provided.

    Args:
        state: Application state
        key: Parameter key to update
        value: New value
        on_change: Optional callback after update
    """
    if state.chunker is not None and hasattr(state.chunker.params, key):
        setattr(state.chunker.params, key, value)
        if on_change:
            on_change()


def update_llm_param(
    state: AppState,
    key: str,
    value: Any,
    on_change: Optional[Callable[[], None]] = None,
) -> None:
    """
    Update an LLM parameter and trigger refresh if provided.

    Args:
        state: Application state
        key: Parameter key to update
        value: New value
        on_change: Optional callback after update
    """
    if hasattr(state.llm_kwargs, key):
        setattr(state.llm_kwargs, key, value)
        if on_change:
            on_change()


# DEPRECATED: Use update_chunker_param or update_llm_param instead
def update_context_param(
    state: AppState,
    key: str,
    value: Any,
    on_change: Optional[Callable[[], None]] = None,
) -> None:
    """
    DEPRECATED: This function is kept for backward compatibility.
    Use update_chunker_param() or update_llm_param() instead.
    """
    # Try to update chunker params first
    if state.chunker is not None and hasattr(state.chunker.params, key):
        update_chunker_param(state, key, value, on_change)
    # Fall back to LLM params
    elif hasattr(state.llm_kwargs, key):
        update_llm_param(state, key, value, on_change)
