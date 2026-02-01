# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

# pyright: ignore
# ruff: ignore

"""
Re-ranking Exercise - Phase 4

Two-stage retrieval: ANN search returns many candidates (fast, approximate),
then a re-ranker scores them more carefully and returns the best results.

The baseline below just truncates to top_k. Your job: replace it with
something smarter. Pick any option below (or invent your own).

Implementation options (easiest to hardest):

1. Recency: sort chunks by end_time (newest first). For chat data this is
   surprisingly effective -- recent conversations are often most relevant.
   Access timestamps via chunk.metadata["end_time"].

2. Keyword overlap: score each chunk by how many query words appear in it.
   Split query and chunk text into word sets, count the intersection size.

3. Combined: multiply keyword score by a recency boost (e.g., 1/rank_by_time)
   to get the best of both signals.

4. BM25: term-frequency scoring with IDF weighting and length normalization.
   See the reference solution for a full implementation.

5. Cross-encoder (production): use sentence-transformers CrossEncoder to
   jointly encode (query, chunk) pairs and score relevance.

The reference solution uses option 4 (BM25). Flip USE_RERANKING_SOLUTION
in exercise_toggles.py to see it in action.
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

    Args:
        query: The user's search query
        chunks: Candidate chunks from ANN search (e.g., top 50)
        top_k: Number of top results to return after re-ranking

    Returns:
        Top-k chunks sorted by relevance (most relevant first)

    """
    raise NotImplementedError("Not implemented")
    if not chunks:
        return []

    # Baseline: just return the first top_k candidates unchanged.
    # Replace this with a scoring strategy from the options above.
    return list(chunks[:top_k])
