# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

# pyright: ignore
# ruff: ignore

"""
Similarity Functions Exercise - Phase 2

This exercise teaches the core concepts of embedding-based retrieval:
1. Cosine similarity computation
2. Top-k selection with threshold filtering

After completing this exercise, you'll understand:
- How cosine similarity measures semantic relatedness
- Why we normalize vectors for comparison
- How to efficiently select top-k results

Implementation tips:
- Use NumPy for vectorized operations
- Handle edge cases (zero vectors, empty arrays)
- Think about numerical stability (epsilon for division)
"""

from numpy.typing import NDArray


def cosine_similarity(query_vec: NDArray, chunk_vecs: NDArray) -> NDArray:
    """
    Compute cosine similarity between a query vector and multiple chunk vectors.

    Cosine similarity measures the angle between two vectors, ignoring magnitude.
    It's computed as: cos(theta) = (A . B) / (||A|| * ||B||)

    Args:
        query_vec: Query embedding vector with shape (dim,)
        chunk_vecs: Matrix of chunk embeddings with shape (num_chunks, dim)

    Returns:
        Array of similarity scores with shape (num_chunks,)
        Values range from -1 (opposite) to 1 (identical direction)

    Examples:
        >>> import numpy as np
        >>> query = np.array([1.0, 0.0, 0.0])
        >>> chunks = np.array([
        ...     [1.0, 0.0, 0.0],   # Identical to query
        ...     [0.0, 1.0, 0.0],   # Orthogonal to query
        ...     [-1.0, 0.0, 0.0],  # Opposite to query
        ... ])
        >>> sims = cosine_similarity(query, chunks)
        >>> # sims should be approximately [1.0, 0.0, -1.0]

    Implementation hints:
    1. Compute the L2 norm of query_vec
    2. Compute L2 norms of each row in chunk_vecs
    3. Compute dot products between query and all chunks
    4. Divide dot products by product of norms
    5. Handle zero-norm vectors (return 0 similarity)
    6. Add small epsilon (1e-10) to avoid division by zero

    TODO: Implement this function
    TODO: REMOVE THE EXCEPTION
    """
    raise NotImplementedError(
        "Implement cosine_similarity.\n" "Hint: Use np.linalg.norm() for norms and np.dot() for dot products."
    )


def get_top_k(
    similarities: NDArray,
    threshold: float,
    k: int,
) -> NDArray:
    """
    Get indices of top-k chunks above similarity threshold.

    This function filters chunks by similarity threshold, then returns
    the indices of the k highest-scoring chunks in descending order.

    Args:
        similarities: Array of similarity scores with shape (num_chunks,)
        threshold: Minimum similarity score (chunks below this are excluded)
        k: Maximum number of indices to return

    Returns:
        Array of indices sorted by similarity (highest first)
        Length is min(k, number of chunks above threshold)

    Examples:
        >>> import numpy as np
        >>> sims = np.array([0.9, 0.3, 0.7, 0.1, 0.8])
        >>> indices = get_top_k(sims, threshold=0.5, k=2)
        >>> # indices should be [0, 4] (scores 0.9 and 0.8)

        >>> indices = get_top_k(sims, threshold=0.5, k=10)
        >>> # indices should be [0, 4, 2] (all above 0.5, sorted)

    Implementation hints:
    1. Create a boolean mask for similarities >= threshold
    2. Get indices where mask is True (np.where)
    3. Get similarity values at those indices
    4. Sort indices by similarity in descending order (np.argsort with negation)
    5. Take first k indices

    TODO: Implement this function
    TODO: REMOVE THE EXCEPTION
    """
    raise NotImplementedError("Implement get_top_k.\n" "Hint: Use np.where() to filter, np.argsort() to sort indices.")
