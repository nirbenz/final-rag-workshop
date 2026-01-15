"""Tests for chunker parameter classes."""



from workshop.rag.chunkers import MessageCountParams
from workshop.rag.types import BaseChunkerParams


def test_base_chunker_params_defaults():
    """Test BaseChunkerParams default values."""
    params = BaseChunkerParams()

    assert params.max_tokens == 25_000
    assert params.max_days == 25


def test_base_chunker_params_custom():
    """Test BaseChunkerParams with custom values."""
    params = BaseChunkerParams(
        max_tokens=50_000,
        max_days=30,
    )

    assert params.max_tokens == 50_000
    assert params.max_days == 30


def test_message_count_params_inheritance():
    """Test MessageCountParams inherits from BaseChunkerParams."""
    params = MessageCountParams()

    # Check inherited fields
    assert hasattr(params, "max_tokens")
    assert hasattr(params, "max_days")

    # Check own fields
    assert hasattr(params, "chunk_length")
    assert hasattr(params, "chunk_overlap")


def test_message_count_params_mixed_values():
    """Test MessageCountParams accepts both inherited and own fields."""
    params = MessageCountParams(
        max_tokens=100_000,
        max_days=60,
        chunk_length=10,
        chunk_overlap=3,
    )

    assert params.max_tokens == 100_000
    assert params.max_days == 60
    assert params.chunk_length == 10
    assert params.chunk_overlap == 3
