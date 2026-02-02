# Created by Nir Ben-Zvi
# Reusable LLM UI Components for NiceGUI
#
# These components are generic and can be reused across different NiceGUI apps
# that need LLM model selection, usage tracking, and RAG-style highlighting.

import re
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, get_args, get_origin

from loguru import logger
from nicegui import ui
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_ai import Embedder, RunUsage

# Config type - plain dict after OmegaConf.to_container()
Config = Dict[str, Any]


def get_llm_wrapper(model_config: Config, structured_output_type: Optional[type] = None) -> Optional[object]:
    """
    Initialize and return an LLM wrapper for the given model config.

    Args:
        model_config: Model configuration dict with model_name and kwargs
        structured_output_type: Optional Pydantic model for structured output

    Returns:
        Initialized LLM wrapper (pydantic-ai Agent) or None if initialization fails
    """
    try:
        from workshop.llm import get_pydantic_agent
        from workshop.types import LLMConfig

        return get_pydantic_agent(LLMConfig(**model_config), structured_output_type)
    except ImportError:
        logger.warning("workshop.llm not available, LLM wrapper not initialized")
        return None
    except Exception:
        logger.exception("Failed to initialize LLM")
        return None


def get_embedder(model_config: Config) -> Optional[Embedder]:
    """
    Initialize and return a pydantic-ai Embedder for the given model config.

    Args:
        model_config: Embedding model configuration with model_name and kwargs

    Returns:
        Pydantic-AI Embedder instance, or None if init fails
    """
    try:
        from workshop.llm import get_embedding_model
        from workshop.types import LLMConfig

        return get_embedding_model(LLMConfig(**model_config))
    except ImportError as e:
        logger.warning(f"Failed to import embedding model: {e}")
        return None
    except Exception:
        logger.exception("Failed to initialize embedding model")
        return None


def render_model_controls(
    model: BaseModel,
    on_change: Optional[Callable[[], None]] = None,
    title: Optional[str] = None,
    disabled: bool = False,
) -> None:
    """
    Auto-generate UI controls from a Pydantic BaseModel.

    Uses field metadata to determine control type and constraints:
    - float/int: Number input (default) or slider if ui_type="slider"
    - bool: Switch
    - str: Input
    - Literal[...]: Select dropdown

    Field.json_schema_extra can contain:
    - label: Display label (defaults to field name)
    - tooltip: Help text
    - step: Step size for numeric inputs
    - hidden: Skip this field (deprecated, use exclude_from_form)
    - exclude_from_form: Skip this field from rendering
    - ui_type: "number" (default), "slider", "input"

    Args:
        model: Pydantic model instance to create controls for
        on_change: Optional callback when any value changes
        title: Optional section title
        disabled: If True, all controls are grayed out and non-interactive
    """
    container_classes = "w-full"
    if disabled:
        container_classes += " opacity-50 pointer-events-none"

    with ui.column().classes(container_classes):
        if title:
            ui.label(title).classes("font-semibold text-gray-300 mb-2")

        for field_name, field_info in model.model_fields.items():
            _render_field_control(model, field_name, field_info, on_change, disabled)


def _get_numeric_constraints(field_info: FieldInfo) -> Tuple[Optional[float], Optional[float]]:
    """Extract ge (min) and le (max) constraints from Pydantic v2 field metadata."""
    ge_val: Optional[float] = None
    le_val: Optional[float] = None

    for constraint in field_info.metadata:
        # Pydantic v2 uses constraint objects in metadata
        if hasattr(constraint, "ge"):
            ge_val = constraint.ge
        if hasattr(constraint, "le"):
            le_val = constraint.le

    return ge_val, le_val


def _render_field_control(
    model: BaseModel,
    field_name: str,
    field_info: FieldInfo,
    on_change: Optional[Callable[[], None]] = None,
    disabled: bool = False,
) -> None:
    """Render a single field control based on its type and metadata."""
    # json_schema_extra can be dict or callable; handle both
    raw_extra = field_info.json_schema_extra
    if isinstance(raw_extra, dict):
        extra: Dict[str, Any] = dict(raw_extra)
    else:
        extra = {}

    # Skip hidden or excluded fields
    if extra.get("hidden") or extra.get("exclude_from_form"):
        return

    # Get display info
    label: str = str(extra.get("label", field_name.replace("_", " ").title()))
    tooltip: str = str(extra.get("tooltip", ""))
    current_value = getattr(model, field_name)
    annotation = field_info.annotation

    # Determine control type based on annotation
    origin = get_origin(annotation)

    def update_value(e: Any) -> None:
        setattr(model, field_name, e.value)
        if on_change:
            on_change()

    # Handle Literal (dropdown)
    if origin is Literal:
        options = list(get_args(annotation))
        with ui.column().classes("w-full gap-1"):
            ui.label(label).classes("text-sm text-gray-400")
            select = ui.select(options=options, value=current_value, on_change=update_value).classes("w-full")
            if disabled:
                select.disable()
        return

    # Handle bool (switch)
    if annotation is bool:
        switch = ui.switch(label, value=current_value, on_change=update_value).classes("w-full")
        if disabled:
            switch.disable()
        return

    # Handle numeric types (float/int)
    ge_val, le_val = _get_numeric_constraints(field_info)
    if annotation in (float, int):
        step: float = float(extra.get("step", 1 if annotation is int else 0.1))

        # Default ui_type: float->slider (if bounds), int->number
        default_ui_type = "slider" if annotation is float and ge_val is not None and le_val is not None else "number"
        ui_type: str = str(extra.get("ui_type", default_ui_type))

        # Use slider if requested AND bounds available
        if ui_type == "slider" and ge_val is not None and le_val is not None:
            with ui.column().classes("w-full gap-1"):
                with ui.row().classes("w-full justify-between items-center"):
                    ui.label(label).classes("text-sm text-gray-400")
                    if tooltip:
                        ui.icon("help_outline").classes("text-gray-500 text-sm").tooltip(tooltip)
                    value_label = ui.label().classes("text-sm text-blue-400 font-mono")

                    # Format based on type
                    if annotation is int:
                        value_label.bind_text_from(model, field_name, backward=lambda v: f"{int(v):,}")
                    else:
                        value_label.bind_text_from(model, field_name, backward=lambda v: f"{v:.2f}")

                def make_slider_update(fn: Optional[Callable[[Any], Any]]) -> Callable[[Any], None]:
                    def handler(e: Any) -> None:
                        if e.value is None:
                            return
                        val = fn(e.value) if fn else e.value
                        setattr(model, field_name, val)
                        if on_change:
                            on_change()

                    return handler

                slider = ui.slider(
                    min=ge_val,
                    max=le_val,
                    step=step,
                    value=current_value,
                    on_change=make_slider_update(int if annotation is int else None),
                ).classes("w-full")
                if disabled:
                    slider.disable()
            return

        # Default: number input
        def make_number_update() -> Callable[[Any], None]:
            def handler(e: Any) -> None:
                if e.value is None or e.value == "":
                    return
                val = int(e.value) if annotation is int else float(e.value)
                setattr(model, field_name, val)
                if on_change:
                    on_change()

            return handler

        number_input = ui.number(
            label=label,
            value=current_value,
            step=step,
            min=ge_val,
            max=le_val,
            on_change=make_number_update(),
        ).classes("w-full")
        if tooltip:
            number_input.tooltip(tooltip)
        if disabled:
            number_input.disable()
        return

    # Handle string (input)
    if annotation is str:
        with ui.column().classes("w-full gap-1"):
            text_input = ui.input(label=label, value=current_value, on_change=update_value).classes("w-full")
            if disabled:
                text_input.disable()
        return

    # Handle Optional[datetime] (date picker)
    from datetime import datetime

    # Check if this is Optional[datetime]
    if origin and datetime in get_args(annotation):
        with ui.column().classes("w-full gap-1"):
            ui.label(label).classes("text-sm text-gray-400")

            # Format current value for display
            display_value = None
            if current_value is not None:
                if isinstance(current_value, datetime):
                    display_value = current_value.strftime("%Y-%m-%d")
                else:
                    display_value = str(current_value)

            def on_date_change(e: Any) -> None:
                """Handle date picker change."""
                if e.value:
                    # Parse date string to datetime (start of day)
                    try:
                        from datetime import datetime

                        dt = datetime.strptime(e.value, "%Y-%m-%d")
                        setattr(model, field_name, dt)
                        if on_change:
                            on_change()
                    except Exception as ex:
                        logger.error(f"Failed to parse date: {ex}")
                else:
                    # Clear the date (set to None)
                    setattr(model, field_name, None)
                    if on_change:
                        on_change()

            date_input = ui.date(value=display_value, on_change=on_date_change).classes("w-full")
            if tooltip:
                date_input.tooltip(tooltip)
            if disabled:
                date_input.disable()

            # Add clear button
            if current_value is not None:
                with ui.row().classes("w-full items-center gap-2 mt-1"):
                    ui.button(
                        "Clear",
                        on_click=lambda: (
                            setattr(model, field_name, None),
                            date_input.set_value(None),
                            on_change() if on_change else None,
                        ),
                    ).props("flat dense size=sm color=grey")
        return

    # Fallback: just show the value
    ui.label(f"{label}: {current_value}").classes("text-sm text-gray-400")


# Hebrew Unicode range
HEBREW_PATTERN = re.compile(r"[\u0590-\u05FF]")


def contains_hebrew(text: str) -> bool:
    """Check if text contains Hebrew characters."""
    return bool(HEBREW_PATTERN.search(text))


def get_text_direction(text: str, mode: str = "auto") -> str:
    """
    Get text direction based on content and mode.

    Args:
        text: The text to analyze
        mode: "auto" (detect), "rtl", or "ltr"

    Returns:
        "rtl" or "ltr"
    """
    if mode == "rtl":
        return "rtl"
    elif mode == "ltr":
        return "ltr"
    else:  # auto
        return "rtl" if contains_hebrew(text) else "ltr"


def get_direction_classes(direction: str) -> str:
    """Get Tailwind classes for text direction."""
    if direction == "rtl":
        return "text-right"
    return "text-left"


def parse_model_configs(config: Config, llm_key: str) -> List[Config]:
    """
    Parse model configs from Hydra config, handling various formats.

    Hydra can return models as dict, list, or single item depending on
    how the config is structured. This normalizes all cases to a list.

    Args:
        config: Root configuration (plain dict)
        llm_key: Key in config.models to look up

    Returns:
        List of model configurations

    Raises:
        ValueError: If model config not found for the given key
    """
    models = config.get("models", {})
    model_config = models.get(llm_key)

    if model_config is None:
        raise ValueError(f"Model config not found for key: {llm_key}")

    if "model_name" in model_config:
        return [model_config]
    elif isinstance(model_config, Mapping):
        return list(model_config.values())
    elif isinstance(model_config, Sequence):
        return list(model_config)
    else:
        raise ValueError(f"Model config is not a valid format: {model_config}")


def create_model_selector(
    config: Optional[Config],
    selected_idx: int,
    current_model_name: str,
    on_select: Callable[[int, str, Config], None],
    llm_key: str = "conversation_llm",
) -> None:
    """
    Create a model selector dropdown, or just display the model if only one is configured.

    This is a pure UI component that doesn't manage state internally.
    The caller provides the current selection and a callback for changes.

    When only a single model is configured, no dropdown is shown - just the model name.

    Args:
        config: Configuration dict containing models
        selected_idx: Currently selected model index
        current_model_name: Display name of current model (for badge)
        on_select: Callback(idx, model_name, model_config) when selection changes
        llm_key: Key in config.models to look up
    """
    if config is None:
        ui.label("No config loaded").classes("text-gray-500 text-xs italic")
        return

    model_configs = parse_model_configs(config, llm_key)
    initial_value = model_configs[selected_idx]["model_name"]

    # Always init on render - page_state.llm_wrapper is fresh on each page load
    on_select(selected_idx, initial_value, model_configs[selected_idx])

    with ui.column().classes("w-full gap-2"):
        ui.label("Model").classes("font-semibold text-gray-300")

        # Only show dropdown if multiple models are configured
        if len(model_configs) > 1:
            options = {cfg["model_name"]: idx for idx, cfg in enumerate(model_configs)}

            def handle_change(e):
                idx = options[e.value]
                logger.info(f"Selected model: {e.value}")
                on_select(idx, e.value, model_configs[idx])

            ui.select(
                options=list(options.keys()),
                value=initial_value,
                on_change=handle_change,
            ).classes("w-full").props("outlined dense")

        # Show current model badge
        with ui.row().classes("items-center gap-2 bg-gray-700 rounded px-2 py-1"):
            ui.icon("smart_toy").classes("text-blue-400 text-sm")
            ui.label(current_model_name).classes("text-blue-300 font-mono text-xs")


def sum_usages(usages: List[RunUsage]) -> RunUsage:
    """
    Sum multiple RunUsage objects into a total.

    Args:
        usages: List of usage objects to sum

    Returns:
        Combined usage totals
    """
    if not usages:
        return RunUsage()

    total = usages[0]
    for usage in usages[1:]:
        total.incr(usage)
    return total


def create_usage_stats(usage_maps: Dict[str, List[RunUsage]]) -> None:
    """
    Create usage statistics display for LLM calls.

    Shows total and last usage cost for each model that has been used.

    Args:
        usage_maps: Dict mapping model_name -> list of RunUsage objects
    """
    if not usage_maps or all(len(v) == 0 for v in usage_maps.values()):
        ui.label("No usage data yet").classes("text-gray-500 text-xs italic")
        return

    try:
        from genai_prices import calc_price
    except ImportError:
        ui.label("genai_prices not installed").classes("text-yellow-500 text-xs")
        return

    for model_name, usages in usage_maps.items():
        if not usages:
            continue

        total_usage = sum_usages(usages)
        last_usage = usages[-1]
        provider_id, model_ref = model_name.split(":", 1) if ":" in model_name else (None, model_name)

        # Try to calculate cost, fallback to tokens-only if model not in price database
        try:
            total_cost = calc_price(total_usage, model_ref=model_ref, provider_id=provider_id)
            last_cost = calc_price(last_usage, model_ref=model_ref, provider_id=provider_id)
            has_pricing = True
        except Exception as e:
            logger.warning(f"Could not calculate price for {model_name}: {e}")
            has_pricing = False

        with ui.column().classes("gap-1 w-full"):
            ui.label(model_name).classes("font-semibold text-xs text-gray-300 truncate")

            with ui.row().classes("gap-4 text-xs w-full"):
                with ui.column().classes("gap-0"):
                    ui.label("Total").classes("text-gray-500")
                    if has_pricing:
                        ui.label(f"${total_cost.total_price:.4f}").classes("text-green-400 font-mono")
                    else:
                        ui.label("N/A").classes("text-gray-500 font-mono")
                    ui.label(f"{total_usage.total_tokens} tokens").classes("text-gray-500 text-xs")

                with ui.column().classes("gap-0"):
                    ui.label("Last").classes("text-gray-500")
                    if has_pricing:
                        ui.label(f"${last_cost.total_price:.4f}").classes("text-green-400 font-mono")
                    else:
                        ui.label("N/A").classes("text-gray-500 font-mono")
                    ui.label(f"{last_usage.total_tokens} tokens").classes("text-gray-500 text-xs")


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def render_with_coordinate_highlights(
    full_text: str,
    coordinates: List[Tuple[int, int]],
    highlight_class: str = "bg-yellow-400 text-black",
    rtl_mode: str = "auto",
) -> None:
    """
    Core rendering function: render text with highlighted regions by coordinates.

    This is the single source of truth for highlight rendering.
    Higher-level functions (like render_with_highlights) should use this.

    Args:
        full_text: The complete document text
        coordinates: List of (start, end) tuples to highlight
        highlight_class: Tailwind classes for highlighting
        rtl_mode: "auto", "rtl", or "ltr"
    """
    direction = get_text_direction(full_text, rtl_mode)
    dir_classes = get_direction_classes(direction)

    with ui.element("div").classes(f"{dir_classes} break-words whitespace-pre-wrap").props(f'dir="{direction}"'):
        if not coordinates:
            ui.markdown(full_text)
            return

        sorted_coords = sorted(coordinates, key=lambda x: x[0])

        last_end = 0
        for start, end in sorted_coords:
            if start < last_end:  # Skip overlapping
                continue

            if last_end < start:
                ui.html(f"<span>{_escape_html(full_text[last_end:start])}</span>", sanitize=False)

            ui.html(
                f'<span class="{highlight_class} px-1 rounded">{_escape_html(full_text[start:end])}</span>',
                sanitize=False,
            )
            last_end = end

        if last_end < len(full_text):
            ui.html(f"<span>{_escape_html(full_text[last_end:])}</span>", sanitize=False)


def render_with_highlights(
    full_text: str,
    chunks: List[str],
    rtl_mode: str = "auto",
    highlight_class: str = "bg-yellow-400 text-black",
) -> None:
    """
    Render text with highlighted chunks for RAG-style context display.

    Finds chunk coordinates in the document, then delegates to
    render_with_coordinate_highlights for actual rendering.

    Args:
        full_text: The complete document text
        chunks: List of text chunks to highlight
        rtl_mode: "auto", "rtl", or "ltr"
        highlight_class: Tailwind classes for highlighting
    """
    coordinates: List[Tuple[int, int]] = []

    try:
        from workshop.utils import get_chunk_coordinates

        coordinates = get_chunk_coordinates(
            document_text=full_text,
            chunk_texts=chunks,
            allow_partial=True,  # Skip chunks that don't match exactly
        )
    except Exception as e:
        logger.warning(f"get_chunk_coordinates failed: {e}")

    if not coordinates:
        # Show message when no matches found
        ui.label("(No matching context found for highlighting)").classes("text-yellow-500 text-xs italic mb-2")

    render_with_coordinate_highlights(
        full_text=full_text,
        coordinates=coordinates,
        highlight_class=highlight_class,
        rtl_mode=rtl_mode,
    )


def render_messages_with_highlights(
    messages: Sequence[Any],
    highlighted_indices: List[int],
    rtl_mode: str = "auto",
    highlight_class: str = "bg-yellow-400 text-black",
) -> None:
    """
    Render messages with exact highlighting by message index.

    This is the preferred highlighting method when retrieval provides message_ids,
    as it avoids text matching and provides exact highlighting.

    Args:
        messages: Sequence of message objects with timed_form() method
        highlighted_indices: List of message indices to highlight
        rtl_mode: "auto", "rtl", or "ltr"
        highlight_class: Tailwind classes for highlighting
    """
    if not messages:
        ui.label("No messages to display").classes("text-gray-500 italic")
        return

    highlighted_set = set(highlighted_indices)
    full_text = "\n".join(getattr(msg, "timed_form", lambda: str(msg))() for msg in messages)
    direction = get_text_direction(full_text, rtl_mode)
    dir_classes = get_direction_classes(direction)

    with ui.element("div").classes(f"{dir_classes} break-words whitespace-pre-wrap").props(f'dir="{direction}"'):
        for idx, msg in enumerate(messages):
            msg_text = getattr(msg, "timed_form", lambda: str(msg))()
            if idx in highlighted_set:
                ui.html(
                    f'<div class="{highlight_class} px-1 rounded mb-1">{_escape_html(msg_text)}</div>',
                    sanitize=False,
                )
            else:
                ui.html(f'<div class="mb-1">{_escape_html(msg_text)}</div>', sanitize=False)


def group_indices_to_regions(
    highlighted_indices: List[int],
    total_messages: int,
    padding: int = 1,
) -> List[Tuple[int, int]]:
    """
    Group highlighted indices into contiguous regions with padding.

    Takes scattered message indices and groups consecutive ones together,
    then adds padding before/after each region. Overlapping regions are merged.

    Args:
        highlighted_indices: List of message indices that are highlighted
        total_messages: Total number of messages in the conversation
        padding: Number of messages to include before/after each region

    Returns:
        List of (start, end) tuples representing visible regions (end exclusive)
    """
    if not highlighted_indices:
        return []

    sorted_indices = sorted(set(highlighted_indices))

    # Group consecutive indices into regions
    regions: List[Tuple[int, int]] = []
    region_start = sorted_indices[0]
    region_end = sorted_indices[0] + 1

    for idx in sorted_indices[1:]:
        if idx <= region_end:
            # Extend current region
            region_end = idx + 1
        else:
            # Save current region, start new one
            regions.append((region_start, region_end))
            region_start = idx
            region_end = idx + 1

    regions.append((region_start, region_end))

    # Add padding and clamp to bounds
    padded_regions = [(max(0, start - padding), min(total_messages, end + padding)) for start, end in regions]

    # Merge overlapping regions
    if not padded_regions:
        return []

    merged: List[Tuple[int, int]] = [padded_regions[0]]
    for start, end in padded_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Overlaps - merge
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def compute_display_segments(
    regions: List[Tuple[int, int]],
    total_messages: int,
) -> List[Dict[str, Any]]:
    """
    Compute alternating visible regions and gaps for display.

    Takes a list of visible regions and computes the gaps between them,
    returning a sequence of segments that can be rendered in order.

    Args:
        regions: List of (start, end) tuples for visible regions
        total_messages: Total number of messages

    Returns:
        List of dicts with keys:
        - type: "region" or "gap"
        - start: Start index (inclusive)
        - end: End index (exclusive)
    """
    if not regions:
        # Everything is a gap
        if total_messages > 0:
            return [{"type": "gap", "start": 0, "end": total_messages}]
        return []

    segments: List[Dict[str, Any]] = []
    current_pos = 0

    for region_start, region_end in regions:
        # Add gap before this region if there is one
        if current_pos < region_start:
            segments.append({"type": "gap", "start": current_pos, "end": region_start})

        # Add the region
        segments.append({"type": "region", "start": region_start, "end": region_end})
        current_pos = region_end

    # Add trailing gap if any
    if current_pos < total_messages:
        segments.append({"type": "gap", "start": current_pos, "end": total_messages})

    return segments


def render_messages_with_collapsible_highlights(
    messages: Sequence[Any],
    highlighted_indices: List[int],
    rtl_mode: str = "auto",
    highlight_class: str = "bg-yellow-400 text-black",
) -> None:
    """
    Render messages with collapsible gaps between highlighted regions.

    Shows only highlighted messages plus 1 message of padding before/after.
    Gaps between regions are collapsed into clickable dividers that expand
    to reveal hidden messages.

    Args:
        messages: Sequence of message objects with timed_form() method
        highlighted_indices: List of message indices to highlight
        rtl_mode: "auto", "rtl", or "ltr"
        highlight_class: Tailwind classes for highlighting
    """
    if not messages:
        ui.label("No messages to display").classes("text-gray-500 italic")
        return

    total_messages = len(messages)
    highlighted_set = set(highlighted_indices)

    # Compute regions and segments
    regions = group_indices_to_regions(highlighted_indices, total_messages, padding=1)
    segments = compute_display_segments(regions, total_messages)

    # If no segments (no highlights), show all messages without collapse
    if not segments or not highlighted_indices:
        render_messages_with_highlights(messages, highlighted_indices, rtl_mode, highlight_class)
        return

    # Determine text direction from full content
    full_text = "\n".join(getattr(msg, "timed_form", lambda: str(msg))() for msg in messages)
    direction = get_text_direction(full_text, rtl_mode)
    dir_classes = get_direction_classes(direction)

    # Track expanded gaps locally
    expanded_gaps: set = set()

    @ui.refreshable
    def render_content():
        with ui.element("div").classes(f"{dir_classes} break-words whitespace-pre-wrap").props(f'dir="{direction}"'):
            for seg_idx, segment in enumerate(segments):
                seg_type = segment["type"]
                start = segment["start"]
                end = segment["end"]
                gap_count = end - start

                if seg_type == "region":
                    # Render messages in this region
                    for idx in range(start, end):
                        msg = messages[idx]
                        msg_text = getattr(msg, "timed_form", lambda: str(msg))()
                        if idx in highlighted_set:
                            ui.html(
                                f'<div class="{highlight_class} px-1 rounded mb-1">{_escape_html(msg_text)}</div>',
                                sanitize=False,
                            )
                        else:
                            # Padding messages - show but dimmed
                            ui.html(
                                f'<div class="mb-1 text-gray-400">{_escape_html(msg_text)}</div>',
                                sanitize=False,
                            )

                elif seg_type == "gap":
                    gap_key = (start, end)

                    if gap_key in expanded_gaps:
                        # Expanded: show messages with collapse button
                        def make_collapse_handler(key):
                            def handler():
                                expanded_gaps.discard(key)
                                render_content.refresh()

                            return handler

                        with ui.row().classes("w-full items-center gap-2 my-2"):
                            ui.button(
                                icon="unfold_less",
                                on_click=make_collapse_handler(gap_key),
                            ).props("flat dense size=sm color=grey").tooltip("Collapse")
                            ui.label(f"Messages {start + 1}-{end}").classes("text-xs text-gray-500")

                        # Show the gap messages (dimmed)
                        for idx in range(start, end):
                            msg = messages[idx]
                            msg_text = getattr(msg, "timed_form", lambda: str(msg))()
                            ui.html(
                                f'<div class="mb-1 text-gray-500">{_escape_html(msg_text)}</div>',
                                sanitize=False,
                            )
                    else:
                        # Collapsed: show expandable divider
                        def make_expand_handler(key):
                            def handler():
                                expanded_gaps.add(key)
                                render_content.refresh()

                            return handler

                        with (
                            ui.row()
                            .classes(
                                "w-full items-center justify-center gap-2 my-2 py-2 "
                                "border-y border-gray-600 cursor-pointer hover:bg-gray-800 rounded"
                            )
                            .on("click", make_expand_handler(gap_key))
                        ):
                            ui.icon("unfold_more").classes("text-gray-500")
                            ui.label(f"Show {gap_count} hidden messages").classes("text-xs text-gray-500")

    render_content()


CHUNK_COLORS = [
    "bg-blue-200 text-blue-900",
    "bg-green-200 text-green-900",
    "bg-purple-200 text-purple-900",
    "bg-orange-200 text-orange-900",
    "bg-pink-200 text-pink-900",
    "bg-teal-200 text-teal-900",
]


def render_chunk_preview_with_boundaries(
    messages: List[str],
    boundaries: List[Tuple[int, int]],
    chunk_length: int,
    overlap: int,
    max_to_show: int = 20,
    rtl_mode: str = "auto",
) -> None:
    """
    Render a preview of chunks using pre-computed boundaries.

    Shows each chunk with alternating colors so you can visualize the
    chunk length and overlap. This version accepts boundaries from a chunker
    instead of computing them inline.

    Args:
        messages: List of message strings
        boundaries: List of (start, end) tuples from chunker
        chunk_length: Number of messages per chunk (for display only)
        overlap: Number of messages overlapping (for display only)
        max_to_show: Maximum messages to show per chunk
        rtl_mode: "auto", "rtl", or "ltr"
    """
    if not messages:
        ui.label("No messages to preview").classes("text-gray-500 italic")
        return

    with ui.column().classes("w-full min-w-0 gap-1"):
        ui.label(f"Chunks: {len(boundaries)} | Length: {chunk_length} | Overlap: {overlap}").classes(
            "text-xs text-gray-400 mb-2"
        )

        for chunk_idx, (start, end) in enumerate(boundaries):
            color = CHUNK_COLORS[chunk_idx % len(CHUNK_COLORS)]

            with ui.card().classes(f"w-full min-w-0 overflow-hidden p-2 {color}"):
                ui.label(f"Chunk {chunk_idx + 1} (msgs {start + 1}-{end})").classes("text-xs font-semibold mb-1")

                chunk_messages = messages[start:end]
                preview_text = "\n".join(chunk_messages[:max_to_show])
                if len(chunk_messages) > max_to_show:
                    preview_text += f"\n... (+{len(chunk_messages) - max_to_show} more)"

                direction = get_text_direction(preview_text, rtl_mode)
                dir_classes = get_direction_classes(direction)

                ui.label(preview_text).classes(f"text-xs whitespace-pre-wrap break-all {dir_classes}").props(
                    f'dir="{direction}"'
                )


def render_chunk_preview(
    messages: List[str],
    chunk_length: int,
    overlap: int,
    max_to_show: int = 20,
    rtl_mode: str = "auto",
) -> None:
    """
    Render a preview of how messages would be chunked with given parameters.

    Deprecated: Use render_chunk_preview_with_boundaries() instead.
    This is kept for backward compatibility.

    Args:
        messages: List of message strings to chunk
        chunk_length: Number of messages per chunk
        overlap: Number of messages overlapping between consecutive chunks
        rtl_mode: "auto", "rtl", or "ltr"
    """
    if not messages:
        ui.label("No messages to preview").classes("text-gray-500 italic")
        return

    if chunk_length <= 0:
        chunk_length = 1
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_length:
        overlap = chunk_length - 1

    stride = chunk_length - overlap

    chunks = []
    i = 0
    while i < len(messages):
        end = min(i + chunk_length, len(messages))
        chunks.append((i, end))
        i += stride
        if i >= len(messages) and end < len(messages):
            break

    render_chunk_preview_with_boundaries(
        messages=messages,
        boundaries=chunks,
        chunk_length=chunk_length,
        overlap=overlap,
        max_to_show=max_to_show,
        rtl_mode=rtl_mode,
    )
