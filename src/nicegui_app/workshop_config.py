# Created by Nir Ben-Zvi
# Workshop Configuration
# me@nirbnzvi.com

# noqa: I001

"""
Workshop Configuration File

Change PHASE below to advance through the workshop.
Edit exercise_toggles.py to switch between your code and reference solutions.
Restart the app after changes.

Workshop Progression:
- Phase 1: NaiveContextEngine (baseline exploration)
- Phase 2: Prompt engineering exercise (still NaiveContextEngine)
  Exercise: exercises/prompting.py - get_system_prompt()
- Phase 3: SimilarityContextEngine (NumPy in-memory retrieval)
  Exercise: exercises/similarity.py - cosine_similarity(), get_top_k()
- Phase 4: RAGContextEngine (Qdrant ANN + BM25 re-ranking)
  Exercise: exercises/reranking.py - rerank()
- Optional: SegmentingChunker, SentenceBoundaryChunker, SemanticChunker
"""

# ============================================================================
# Per-Exercise Solution Toggles
# ============================================================================
# Edit exercise_toggles.py to switch between exercises and solutions.
# Toggles are in a separate file to avoid circular imports.
#
# - USE_PROMPTING_SOLUTION: get_system_prompt() for RAG prompt design (Phase 2)
# - USE_SIMILARITY_SOLUTION: cosine_similarity() and get_top_k() (Phase 3)
# - USE_RERANKING_SOLUTION: rerank() for two-stage retrieval (Phase 4)
# - USE_SEGMENTING_SOLUTION: segment_by_time_gaps() + chunk_segments() (optional)

from workshop.exercise_toggles import (  # noqa: F401
    USE_PROMPTING_SOLUTION,
    USE_RERANKING_SOLUTION,
    USE_SEGMENTING_SOLUTION,
    USE_SIMILARITY_SOLUTION,
)

# ============================================================================
# Workshop Phase (change this number to advance)
# ============================================================================
PHASE = 1

# ============================================================================
# Engine initialization kwargs (optional) - default to empty
# ============================================================================

ENGINE_KWARGS = {}

CHUNKER_KWARGS = {}

# ============================================================================
# Phase 1-2: Baseline + Prompt Engineering (Naive engine)
# ============================================================================
if PHASE in (1, 2):
    # Phase 1: Explore baseline system, build intuition
    # Phase 2: Prompt engineering exercise (exercises/prompting.py)

    from workshop.rag.chunkers import MessageCountChunker
    from workshop.rag.engines import NaiveContextEngine

    CHUNKER_CLASS = MessageCountChunker
    ENGINE_CLASS = NaiveContextEngine

    CHUNKER_DEFAULTS = {
        "max_tokens": 25_000,
        "max_days": 25,
        "chunk_length": 6,
        "chunk_overlap": 4,
    }

# ============================================================================
# Phase 3: Similarity engine (NumPy in-memory cosine similarity)
# ============================================================================

if PHASE == 3:
    # Exercise: exercises/similarity.py - cosine_similarity(), get_top_k()
    # Note: embedder is injected automatically from Hydra config (models.embedding_llm)

    from workshop.rag.chunkers import MessageCountChunker
    from workshop.rag.engines import SimilarityContextEngine

    CHUNKER_CLASS = MessageCountChunker
    ENGINE_CLASS = SimilarityContextEngine

    ENGINE_KWARGS = {
        "similarity_threshold": 0.0,
        "top_k": 10,
    }

    CHUNKER_DEFAULTS = {
        "chunk_length": 6,
        "chunk_overlap": 4,
    }

# ============================================================================
# Phase 4: RAG engine (Qdrant ANN search + BM25 re-ranking)
# ============================================================================

if PHASE == 4:
    # Exercise: exercises/reranking.py - rerank()
    # Note: embedder is injected automatically from Hydra config (models.embedding_llm)

    from workshop.rag.chunkers import MessageCountChunker
    from workshop.rag.engines import RAGContextEngine

    CHUNKER_CLASS = MessageCountChunker
    ENGINE_CLASS = RAGContextEngine

    ENGINE_KWARGS = {
        "db_path": ".qdrant",
        "collection_name": "workshop_chunks",
        "top_k": 10,
        "rerank_candidates": 50,
    }

    CHUNKER_DEFAULTS = {
        "chunk_length": 6,
        "chunk_overlap": 4,
    }

if PHASE not in (1, 2, 3, 4):
    msg = f"Invalid PHASE={PHASE}. Must be 1, 2, 3, or 4."
    raise ValueError(msg)

# ============================================================================
# Optional / Take-Home: Advanced chunkers
# ============================================================================

# Segmenting chunker (time-gap segmentation)
# from workshop.rag.chunkers import SegmentingChunker, SegmentingChunkerParams
#
# CHUNKER_CLASS = SegmentingChunker
# ENGINE_CLASS = RAGContextEngine
#
# CHUNKER_DEFAULTS = {
#     "time_gap_hours": 6.0,
#     "chunk_length": 6,
#     "chunk_overlap": 4,
# }

# Sentence-boundary chunker (token-aware chunking)
# from workshop.rag.chunkers import SentenceBoundaryChunker
#
# CHUNKER_CLASS = SentenceBoundaryChunker
# ENGINE_CLASS = RAGContextEngine
#
# CHUNKER_DEFAULTS = {
#     "max_tokens": 25_000,
#     "max_days": 25,
#     "max_chunk_tokens": 500,
#     "overlap_sentences": 2,
# }

# Semantic chunker (topic-aware)
# from workshop.rag.chunkers import SemanticChunker, SemanticChunkerParams
#
# CHUNKER_CLASS = SemanticChunker
# ENGINE_CLASS = NaiveContextEngine
#
# CHUNKER_KWARGS = {}  # embed_fn injected by main.py
#
# CHUNKER_DEFAULTS = {
#     "similarity_threshold": 0.7,
#     "min_chunk_size": 3,
# }

# Contextual chunker (conversation summary for embeddings)
# from workshop.rag.chunkers import ContextualChunker, ContextualChunkerParams
#
# def summary_fn(messages):
#     """Generate conversation summary."""
#     raise NotImplementedError("Implement your summary function here")
#
# CHUNKER_CLASS = ContextualChunker
# ENGINE_CLASS = NaiveContextEngine
#
# CHUNKER_KWARGS = {
#     "summary_fn": summary_fn,
# }
#
# CHUNKER_DEFAULTS = {
#     "base_chunk_length": 5,
#     "include_summary": True,
# }
