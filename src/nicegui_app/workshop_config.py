# Created by Nir Ben-Zvi
# Workshop Configuration
# me@nirbnzvi.com

# noqa: I001

"""
Workshop Configuration File

Participants edit this file to select chunker and engine implementations.
The GUI automatically adapts based on these selections.

How to use:
1. Uncomment the chunker/engine you want to test
2. Comment out the others
3. Restart the app
4. GUI will render the appropriate hyperparameters automatically

Workshop Progression:
- Phase 1: MessageCountChunker + NaiveContextEngine (baseline + exploration)
- Phase 2: SimilarityContextEngine (embedding-based retrieval)
  Exercise: exercises/similarity.py - cosine_similarity(), get_top_k()
- Phase 3: RAGContextEngine + Prompt Engineering (ANN search + generation)
  Exercise: exercises/reranking.py - rerank()
  Exercise: exercises/prompting.py - system prompt, context formatting
- Phase 4: SegmentingChunker (time-gap segmentation)
  Exercise: exercises/segmenting.py - segment_by_time_gaps(), chunk_segments()
- Extensions: SemanticChunker, ContextualChunker
"""

# ============================================================================
# Engine initialization kwargs (optional) - default to empty (override by uncommenting)
# ============================================================================

ENGINE_KWARGS = {}

CHUNKER_KWARGS = {}

# ============================================================================
# Phase 1: Baseline (Message-count chunker + Naive engine)
# ============================================================================

from workshop.rag.chunkers import MessageCountChunker
from workshop.rag.engines import NaiveContextEngine

CHUNKER_CLASS = MessageCountChunker
ENGINE_CLASS = NaiveContextEngine

CHUNKER_DEFAULTS = {
    # Conversation windowing (inherited from BaseChunkerParams)
    "max_tokens": 25_000,
    "max_days": 25,
    # Chunking-specific
    "chunk_length": 6,
    "chunk_overlap": 4,
}

# ============================================================================
# Phase 2: Similarity engine (embedding-based retrieval with NumPy)
# ============================================================================
# Note: embedder is injected automatically from Hydra config (models.embedding_llm)

# from workshop.rag.engines import SimilarityContextEngine
#
# CHUNKER_CLASS = MessageCountChunker
# ENGINE_CLASS = SimilarityContextEngine
#
# ENGINE_KWARGS = {
#     "similarity_threshold": 0.0,
#     "top_k": 10,
# }
#
# CHUNKER_DEFAULTS = {
#     "chunk_length": 6,
#     "chunk_overlap": 4,
# }

# ============================================================================
# Phase 3: RAG engine (Qdrant with ANN search + re-ranking + prompting)
# ============================================================================
# Note: embedder is injected automatically from Hydra config (models.embedding_llm)
#
# Exercises for Phase 3:
# 1. exercises/reranking.py - Implement rerank() for two-stage retrieval
# 2. exercises/prompting.py - Design system prompts and context formatting
#
# Toggle solutions in:
# - src/workshop/rag/engines/qdrant.py: USE_SOLUTIONS = True/False
# - src/workshop/llm.py: USE_PROMPT_SOLUTIONS = True/False

# from workshop.rag.engines import RAGContextEngine
#
# CHUNKER_CLASS = MessageCountChunker
# ENGINE_CLASS = RAGContextEngine
#
# ENGINE_KWARGS = {
#     "db_path": ".qdrant",
#     "collection_name": "workshop_chunks",
#     "top_k": 10,
#     "rerank_candidates": 50,  # Retrieve 50 candidates, re-rank to top_k
# }
#
# CHUNKER_DEFAULTS = {
#     "chunk_length": 6,
#     "chunk_overlap": 4,
# }

# ============================================================================
# Phase 4: Sentence-boundary chunker (participants implement)
# ============================================================================

# from workshop.rag.chunkers import SentenceBoundaryChunker, SentenceBoundaryParams
#
# CHUNKER_CLASS = SentenceBoundaryChunker
# ENGINE_CLASS = NaiveContextEngine
#
# CHUNKER_DEFAULTS = {
#     # Conversation windowing
#     "max_tokens": 25_000,
#     "max_days": 25,
#     # Chunking-specific
#     "max_chunk_tokens": 500,  # Note: different from max_tokens (context window)
#     "overlap_sentences": 2,
# }


# ============================================================================
# Extensions: Advanced chunkers
# ============================================================================

# Semantic chunker (topic-aware)
# Note: embed_fn is injected automatically from Hydra config (models.embedding_llm)
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

# Segmenting chunker (time-gap segmentation)
# from workshop.rag.chunkers import SegmentingChunker, SegmentingChunkerParams
#
# CHUNKER_CLASS = SegmentingChunker
# ENGINE_CLASS = NaiveContextEngine
#
# CHUNKER_DEFAULTS = {
#     "time_gap_hours": 6.0,
#     "chunk_length": 6,
#     "chunk_overlap": 4,
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
