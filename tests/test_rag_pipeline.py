#!/usr/bin/env python3
# Test script for RAG workshop pipeline
# Tests: chunker -> engine pipeline with Phase 1 components

from datetime import datetime
from pathlib import Path
import sys
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers import MessageCountChunker, MessageCountParams
from workshop.rag.engines import NaiveContextEngine
from workshop.rag.types import BaseChunkerParams


def test_base_chunker_params():
    """Test BaseChunkerParams creation and defaults."""
    print("\n=== Testing BaseChunkerParams ===")

    params = BaseChunkerParams()
    assert params.max_tokens == 25_000, "Default max_tokens should be 25,000"
    assert params.max_days == 25, "Default max_days should be 25"

    print(" BaseChunkerParams defaults are correct")

    # Test with custom values
    custom_params = BaseChunkerParams(
        max_tokens=50_000,
        max_days=30,
    )
    assert custom_params.max_tokens == 50_000
    assert custom_params.max_days == 30

    print(" BaseChunkerParams accepts custom values")


def test_message_count_params_inheritance():
    """Test that MessageCountParams inherits from BaseChunkerParams."""
    print("\n=== Testing MessageCountParams Inheritance ===")

    params = MessageCountParams()

    # Check inherited fields
    assert hasattr(params, "max_tokens"), "Should inherit max_tokens"
    assert hasattr(params, "max_days"), "Should inherit max_days"

    # Check own fields
    assert hasattr(params, "chunk_length"), "Should have chunk_length"
    assert hasattr(params, "chunk_overlap"), "Should have chunk_overlap"

    print(" MessageCountParams correctly inherits from BaseChunkerParams")

    # Test with mixed values
    mixed_params = MessageCountParams(
        max_tokens=100_000,
        max_days=60,
        chunk_length=10,
        chunk_overlap=3,
    )
    assert mixed_params.max_tokens == 100_000
    assert mixed_params.max_days == 60
    assert mixed_params.chunk_length == 10
    assert mixed_params.chunk_overlap == 3

    print(" MessageCountParams accepts both inherited and own fields")


def create_test_messages(count: int = 20) -> List[WhatsappMessage]:
    """Create test messages for chunking."""
    messages = []
    for i in range(count):
        msg = WhatsappMessage(
            timestamp=datetime(2024, 1, 1, 10, i),
            user=f"User{i % 3}",  # 3 different users
            text=f"Test message number {i}",
        )
        messages.append(msg)
    return messages


def test_message_count_chunker():
    """Test MessageCountChunker chunking logic."""
    print("\n=== Testing MessageCountChunker ===")

    messages = create_test_messages(20)

    # Test with default params
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(messages)

    print(f" Created {len(chunks)} chunks from {len(messages)} messages")

    # Verify chunk structure
    first_chunk = chunks[0]
    assert hasattr(first_chunk, "id"), "Chunk should have id"
    assert hasattr(first_chunk, "text"), "Chunk should have text"
    assert hasattr(first_chunk, "message_ids"), "Chunk should have message_ids"
    assert hasattr(first_chunk, "metadata"), "Chunk should have metadata"

    print(" Chunks have correct structure")

    # Verify metadata
    assert "start_idx" in first_chunk.metadata
    assert "end_idx" in first_chunk.metadata
    assert "start_time" in first_chunk.metadata
    assert "end_time" in first_chunk.metadata
    assert "speakers" in first_chunk.metadata

    print(" Chunk metadata is complete")

    # Test with custom params (chunk_length=3, overlap=1)
    custom_chunker = MessageCountChunker(params=MessageCountParams(chunk_length=3, chunk_overlap=1))
    custom_chunks = custom_chunker.chunk_messages(messages[:10])

    # Expected: [0:3], [2:5], [4:7], [6:9], [8:10] = 5 chunks
    expected_count = 5
    assert len(custom_chunks) == expected_count, f"Expected {expected_count} chunks, got {len(custom_chunks)}"

    print(f" Custom chunking (length=3, overlap=1) creates correct number of chunks")

    # Verify overlap
    second_chunk = custom_chunks[1]
    assert 2 in second_chunk.message_ids, "Second chunk should include message 2 (overlap)"

    print(" Overlap works correctly")


def test_get_chunk_boundaries():
    """Test get_chunk_boundaries method for preview."""
    print("\n=== Testing get_chunk_boundaries ===")

    chunker = MessageCountChunker(params=MessageCountParams(chunk_length=5, chunk_overlap=2))

    boundaries = chunker.get_chunk_boundaries(12)

    # Expected with length=5, overlap=2, stride=3:
    # [0:5], [3:8], [6:11], [9:12]
    expected = [(0, 5), (3, 8), (6, 11), (9, 12)]
    assert boundaries == expected, f"Expected {expected}, got {boundaries}"

    print(f" Chunk boundaries: {boundaries}")
    print(" get_chunk_boundaries works correctly")


def test_naive_context_engine():
    """Test NaiveContextEngine storage and retrieval."""
    print("\n=== Testing NaiveContextEngine ===")

    # Create test chunks
    messages = create_test_messages(10)
    chunker = MessageCountChunker()
    chunks = chunker.chunk_messages(messages)

    # Create engine
    engine = NaiveContextEngine()

    # Add chunks
    engine.add_context(chunks)

    print(f" Added {len(chunks)} chunks to engine")

    # Verify storage
    stored_chunks = engine.context
    assert len(stored_chunks) == len(chunks), "All chunks should be stored"

    print(" Engine correctly stores chunks")

    # Test retrieval
    retrieved = engine.get_relevant_context("test query", top_k=5)
    assert len(retrieved) == len(chunks), "NaiveContextEngine should return all chunks"

    print(" get_relevant_context returns all chunks (naive behavior)")


def test_full_pipeline():
    """Test complete pipeline: messages -> chunker -> engine -> retrieval."""
    print("\n=== Testing Full Pipeline ===")

    # 1. Create messages
    messages = create_test_messages(30)
    print(f" Created {len(messages)} test messages")

    # 2. Initialize chunker
    chunker = MessageCountChunker(
        params=MessageCountParams(
            max_tokens=25_000,
            max_days=10,
            chunk_length=5,
            chunk_overlap=2,
        )
    )
    print(" Initialized MessageCountChunker")

    # 3. Chunk messages
    chunks = chunker.chunk_messages(messages)
    print(f" Chunked into {len(chunks)} chunks")

    # 4. Initialize engine
    engine = NaiveContextEngine()
    print(" Initialized NaiveContextEngine")

    # 5. Load chunks into engine
    engine.add_context(chunks)
    print(f" Loaded {len(chunks)} chunks into engine")

    # 6. Verify storage
    stored = engine.context
    assert len(stored) == len(chunks)
    print(f" Engine contains {len(stored)} chunks")

    # 7. Test retrieval
    query = "test query"
    retrieved = engine.get_relevant_context(query, top_k=10)
    assert len(retrieved) > 0
    print(f" Retrieved {len(retrieved)} chunks for query: '{query}'")

    # 8. Verify chunk traceability
    first_chunk = retrieved[0]
    reconstructed_messages = first_chunk.get_messages(messages)
    assert len(reconstructed_messages) > 0
    print(f" Can reconstruct {len(reconstructed_messages)} original messages from chunk")

    print("\n=== PIPELINE TEST PASSED ===")


def test_workshop_config_loading():
    """Test that workshop_config.py can be imported and used."""
    print("\n=== Testing Workshop Config ===")

    from nicegui_app import workshop_config

    assert hasattr(workshop_config, "CHUNKER_CLASS"), "Should have CHUNKER_CLASS"
    assert hasattr(workshop_config, "ENGINE_CLASS"), "Should have ENGINE_CLASS"
    assert hasattr(workshop_config, "CHUNKER_DEFAULTS"), "Should have CHUNKER_DEFAULTS"

    print(" workshop_config.py imports successfully")

    # Test instantiation with defaults
    # Need to create params object from CHUNKER_DEFAULTS
    from workshop.rag.chunkers import MessageCountParams

    chunker_params = MessageCountParams(**workshop_config.CHUNKER_DEFAULTS)
    chunker = workshop_config.CHUNKER_CLASS(params=chunker_params)

    # Some engines require embed_fn - provide a mock if needed
    engine_name = workshop_config.ENGINE_CLASS.__name__
    engine_kwargs = getattr(workshop_config, "ENGINE_KWARGS", {})

    if engine_name in ("RAGContextEngine", "SimilarityContextEngine"):
        # Mock embedder for testing
        from pydantic_ai import Embedder
        from pydantic_ai.embeddings import TestEmbeddingModel

        mock_embedder = Embedder(TestEmbeddingModel())
        engine = workshop_config.ENGINE_CLASS(embedder=mock_embedder, **engine_kwargs)
    else:
        engine = workshop_config.ENGINE_CLASS(**engine_kwargs)

    print(f" Created chunker: {type(chunker).__name__}")
    print(f" Created engine: {type(engine).__name__}")

    # Verify defaults match workshop_config.CHUNKER_DEFAULTS
    assert chunker.params.chunk_length == 6
    assert chunker.params.chunk_overlap == 4

    print(" CHUNKER_DEFAULTS correctly applied")


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG WORKSHOP PIPELINE TEST SUITE")
    print("=" * 60)

    try:
        test_base_chunker_params()
        test_message_count_params_inheritance()
        test_message_count_chunker()
        test_get_chunk_boundaries()
        test_naive_context_engine()
        test_full_pipeline()
        test_workshop_config_loading()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ")
        print("=" * 60)
        print("\nPhase 1 (MessageCountChunker + NaiveEngine) is ready!")
        print("\nNext steps:")
        print("1. Run the app: python src/nicegui_app/main.py")
        print("2. Upload a WhatsApp/Telegram chat file")
        print("3. Verify:")
        print("   - Chunker parameters appear in sidebar")
        print("   - 'chunks' mode shows colored preview")
        print("   - 'vectordb' mode shows stored chunks")
        print("   - Querying returns responses")

        return 0

    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
