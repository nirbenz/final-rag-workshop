# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

# pyright: ignore
# ruff: ignore

"""
Re-ranking Exercise - Phase 3

This exercise teaches the concept of two-stage retrieval:
1. Fast initial retrieval (ANN search) returns many candidates
2. Re-ranking scores candidates more accurately and returns top results

After completing this exercise, you'll understand:
- Why re-ranking improves retrieval quality
- The speed/quality tradeoff in two-stage retrieval
- How cross-encoders differ from bi-encoders

Common re-ranking approaches:
- Cross-encoder models (e.g., ms-marco-MiniLM)
- LLM-based scoring
- Cohere Rerank API
- Reciprocal Rank Fusion (for hybrid search)
"""

from typing import List, Sequence

from workshop.rag.engines.types import ChunkObject


def rerank(
    query: str,
    chunks: Sequence[ChunkObject],
    top_k: int = 5,
) -> List[ChunkObject]:
    """
    Re-rank retrieved chunks by relevance to query.

    This function takes candidates from initial retrieval and re-scores them
    using a more accurate (but slower) method. The goal is to push the most
    relevant chunks to the top.

    Args:
        query: The user's search query
        chunks: Candidate chunks from initial retrieval (e.g., top 50 from ANN)
        top_k: Number of top results to return after re-ranking

    Returns:
        Top-k chunks sorted by relevance (most relevant first)

    Example workflow:
        1. ANN search returns 50 candidates (fast, approximate)
        2. Re-ranker scores all 50 with cross-encoder (slower, accurate)
        3. Return top 5 by re-ranked score

    Implementation options:

    Option A - Cross-encoder (recommended for production):
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)
        # Sort by scores, return top_k

    Option B - LLM-based scoring:
        For each chunk, ask LLM: "Rate relevance 1-10"
        Sort by scores, return top_k

    Option C - Cohere Rerank API:
        import cohere
        co = cohere.Client(api_key)
        results = co.rerank(query=query, documents=[c.text for c in chunks])
        # Return top_k by rerank score

    TODO: Implement a re-ranking strategy
    """
    raise NotImplementedError(
        "Implement rerank.\n"
        "Hint: Start simple - even sorting by chunk length or recency can demonstrate the concept.\n"
        "For production, use a cross-encoder like 'cross-encoder/ms-marco-MiniLM-L-6-v2'."
    )
