# Workshop RAG module
# Chunkers and context engines for the RAG workshop
#
# Module structure:
#   rag/
#     chunkers/   - Message chunking strategies and types (MessageCountChunker, BaseChunkerParams, etc.)
#     engines/    - Context engines and types (NaiveContextEngine, ChunkObject, etc.)

# Types (re-exported from chunkers and engines for backward compatibility)
# Chunkers
from workshop.rag.chunkers import (
    BaseChunkerParams,
    ContextualChunker,
    ContextualChunkerParams,
    MessageChunkerProtocol,
    MessageCountChunker,
    MessageCountParams,
    SegmentingChunker,
    SegmentingChunkerParams,
    SemanticChunker,
    SemanticChunkerParams,
    SentenceBoundaryChunker,
    SentenceBoundaryParams,
)

# Engines
from workshop.rag.engines import (
    ChunkEmbedding,
    ChunkObject,
    ContextEngineProtocol,
    NaiveContextEngine,
    RAGContextEngine,
    SimilarityContextEngine,
)

__all__ = [
    # Types
    "BaseChunkerParams",
    "MessageChunkerProtocol",
    "ChunkObject",
    "ChunkEmbedding",
    "ContextEngineProtocol",
    # Chunkers
    "MessageCountChunker",
    "MessageCountParams",
    "SentenceBoundaryChunker",
    "SentenceBoundaryParams",
    "SemanticChunker",
    "SemanticChunkerParams",
    "SegmentingChunker",
    "SegmentingChunkerParams",
    "ContextualChunker",
    "ContextualChunkerParams",
    # Engines
    "NaiveContextEngine",
    "SimilarityContextEngine",
    "RAGContextEngine",
]
