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

from workshop.structured_types import RAGResponse

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
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "step": 0.05,
            "label": "Top P",
            "tooltip": "Nucleus sampling threshold (leave empty if using temperature)",
        },
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
        highlight_source: Source for highlights - retrieval (message_ids) or model (context_used)
        vectordb_page: Current page in vectordb view (0-indexed)
        vectordb_page_size: Number of chunks per page in vectordb view
        clear_db_before_update: Clear database before updating with new chunks
    """

    rtl_mode: Literal["auto", "rtl", "ltr"] = Field(
        default="auto",
        json_schema_extra={"label": "Text Direction"},
    )
    context_view_mode: Literal["default", "chunks", "highlights", "vectordb"] = Field(
        default="default",
        json_schema_extra={"label": "Context View", "exclude_from_form": True},
    )
    highlight_source: Literal["retrieval", "model"] = Field(
        default="retrieval",
        json_schema_extra={"label": "Highlight Source", "exclude_from_form": True},
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
    clear_db_before_update: bool = Field(
        default=True,
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
        retrieved_message_ids: Message indices from retrieval (for exact highlighting)
        is_streaming: Whether this message is currently being streamed
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: str  # "user" or "assistant"
    content: str
    raw_output: Optional[Any] = None  # RAGResponse or str (non-structured)
    retrieved_message_ids: Optional[List[int]] = None  # Exact indices from retrieval
    is_streaming: bool = Field(default=False, exclude=True)  # Runtime only

    @field_validator("raw_output", mode="before")
    @classmethod
    def validate_raw_output(cls, v: Any) -> Optional[Any]:
        """
        Reconstruct RAGResponse from dict after JSON serialization.

        Handles:
        - None: Returns None
        - RAGResponse: Returns as-is
        - dict: Attempts to reconstruct as RAGResponse
        - str: Returns as-is (non-structured output)
        - Other: Returns as-is
        """
        if v is None:
            return None
        if isinstance(v, RAGResponse):
            return v
        if isinstance(v, dict):
            try:
                return RAGResponse.model_validate(v)
            except Exception:
                return v  # Return dict as-is if validation fails
        return v

    def get_raw_output_display(self) -> Optional[str]:
        """
        Get the full structured output as formatted markdown for display.

        Returns None if raw_output is None (e.g., streaming mode).
        Uses str() which leverages RAGResponse.__str__() for nice formatting.
        For non-RAGResponse outputs, falls back to string representation.
        """
        if self.raw_output is None:
            return None

        # For RAGResponse, str() uses its __str__ method which formats nicely
        # For other types, str() gives a reasonable fallback
        result = str(self.raw_output)

        # If it's a RAGResponse, remove the Output section since it's shown as message content
        if isinstance(self.raw_output, RAGResponse):
            # Remove the "## Output" section from the display
            lines = result.split("\n")
            filtered_lines = []
            skip_until_next_header = False
            for line in lines:
                if line.startswith("## Output"):
                    skip_until_next_header = True
                    continue
                if skip_until_next_header and line.startswith("## "):
                    skip_until_next_header = False
                if not skip_until_next_header:
                    filtered_lines.append(line)
            result = "\n".join(filtered_lines).strip()

        return result if result else None

    @property
    def context_used(self) -> Optional[List[str]]:
        """Get context_used from raw_output (for UI highlighting)."""
        if self.raw_output is None:
            return None
        if isinstance(self.raw_output, RAGResponse):
            return self.raw_output.context_used
        if hasattr(self.raw_output, "context_used"):
            ctx = self.raw_output.context_used
            # Handle both list and string formats
            if isinstance(ctx, list):
                return ctx
            if isinstance(ctx, str):
                return [c.strip() for c in ctx.split("\n") if c.strip()]
        return None

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
        raw_output: Optional[RAGResponse] = None,
        retrieved_message_ids: Optional[List[int]] = None,
    ) -> ChatMessage:
        """Create an assistant message from LLM response."""
        return cls(
            role="assistant",
            content=content,
            raw_output=raw_output,
            retrieved_message_ids=retrieved_message_ids,
        )

    @classmethod
    def streaming_placeholder(cls) -> ChatMessage:
        """Create a placeholder for streaming response."""
        return cls(role="assistant", content="", is_streaming=True)


def extract_llm_response(response_output: Any) -> str:
    """
    Extract display content from LLM response.

    Simple extraction logic:
    1. If string, return as-is
    2. If has .output attribute, return that
    3. Fall back to JSON or str() representation

    Args:
        response_output: Raw output from LLM (string, RAGResponse, or other)

    Returns:
        Display content string
    """
    # Plain string - return as-is
    if isinstance(response_output, str):
        return response_output

    # Has .output attribute (RAGResponse or similar)
    if hasattr(response_output, "output"):
        output = response_output.output
        if isinstance(output, str):
            return output
        # Nested .output (shouldn't happen with simplified types, but handle it)
        if hasattr(output, "output"):
            return str(output.output)

    # Fallback: format as JSON or string
    return _format_as_json(response_output)


def _format_as_json(response: Any) -> str:
    """
    Format response as JSON for display.

    Tries Pydantic serialization first, then regular JSON, then str().

    Args:
        response: Response object to format

    Returns:
        JSON string or str() representation
    """
    import json

    # Pydantic v2
    if hasattr(response, "model_dump_json"):
        try:
            return response.model_dump_json(indent=2)
        except Exception:
            pass

    # Pydantic v1
    if hasattr(response, "json"):
        try:
            return response.json(indent=2)
        except Exception:
            pass

    # Dict-like with model_dump
    if hasattr(response, "model_dump"):
        try:
            return json.dumps(response.model_dump(), indent=2, default=str)
        except Exception:
            pass

    # Plain dict
    if isinstance(response, dict):
        try:
            return json.dumps(response, indent=2, default=str)
        except Exception:
            pass

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

    # Fix: Reset top_p to None if it has a value (to avoid conflicts with temperature)
    # This handles cached browser state from before the fix
    if state.llm_kwargs.top_p is not None:
        state.llm_kwargs.top_p = None

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
