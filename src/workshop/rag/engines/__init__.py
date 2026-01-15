# RAG Engines submodule
# Context engines for storage and retrieval

# Types
# Engine implementations
from workshop.rag.engines.naive import NaiveContextEngine
from workshop.rag.engines.qdrant import RAGContextEngine
from workshop.rag.engines.similarity import SimilarityContextEngine
from workshop.rag.engines.types import ChunkEmbedding, ChunkObject, ContextEngineProtocol

__all__ = [
    # Types
    "ChunkObject",
    "ChunkEmbedding",
    "ContextEngineProtocol",
    # Engines
    "NaiveContextEngine",
    "SimilarityContextEngine",
    "RAGContextEngine",
]
