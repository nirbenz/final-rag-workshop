# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Re-ranking - Naive Baseline Solution

This is a simple baseline that just truncates to top_k.
Participants should improve this with actual re-ranking logic.

The naive solution demonstrates the interface without adding dependencies.
For production, replace with cross-encoder or LLM-based re-ranking.
"""

from typing import List, Sequence

from workshop.rag.engines.types import ChunkObject


def rerank(
    query: str,
    chunks: Sequence[ChunkObject],
    top_k: int = 5,
) -> List[ChunkObject]:
    """
    Naive re-ranker: simply returns the first top_k chunks.

    This baseline assumes the initial retrieval order is good enough.
    In practice, you'd want to re-score using:
    - Cross-encoder models
    - LLM relevance scoring
    - Cohere Rerank API

    Args:
        query: The user's search query (unused in naive version)
        chunks: Candidate chunks from initial retrieval
        top_k: Number of top results to return

    Returns:
        First top_k chunks (no actual re-ranking)
    """
    # Naive: just take first top_k results
    # The initial ANN search already returns by similarity,
    # so this is a reasonable baseline
    return list(chunks[:top_k])
