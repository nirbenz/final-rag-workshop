# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Similarity Functions - Reference Solutions

These are the working implementations for the similarity exercise.
Compare your implementation with these solutions after completing the exercise.
"""

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(query_vec: NDArray, chunk_vecs: NDArray) -> NDArray:
    """
    Compute cosine similarity between a query vector and multiple chunk vectors.

    Args:
        query_vec: Query embedding vector with shape (dim,)
        chunk_vecs: Matrix of chunk embeddings with shape (num_chunks, dim)

    Returns:
        Array of similarity scores with shape (num_chunks,)
    """
    # Compute L2 norm of query vector
    query_norm = np.linalg.norm(query_vec)

    # Handle zero-norm query (return zeros)
    if query_norm == 0:
        return np.zeros(len(chunk_vecs))

    # Compute L2 norms of each chunk vector (along axis 1)
    chunk_norms = np.linalg.norm(chunk_vecs, axis=1)

    # Compute dot products between query and all chunks
    # chunk_vecs @ query_vec gives shape (num_chunks,)
    dot_products = np.dot(chunk_vecs, query_vec)

    # Divide by product of norms with epsilon for numerical stability
    similarities = dot_products / (query_norm * chunk_norms + 1e-10)

    return similarities


def get_top_k(
    similarities: NDArray,
    threshold: float,
    k: int,
) -> NDArray:
    """
    Get indices of top-k chunks above similarity threshold.

    Args:
        similarities: Array of similarity scores with shape (num_chunks,)
        threshold: Minimum similarity score
        k: Maximum number of indices to return

    Returns:
        Array of indices sorted by similarity (highest first)
    """
    # Create mask for similarities above threshold
    mask = similarities >= threshold

    # Get indices where mask is True
    filtered_indices = np.where(mask)[0]

    # If no chunks pass threshold, return empty array
    if len(filtered_indices) == 0:
        return np.array([], dtype=np.int64)

    # Get similarity values at filtered indices
    filtered_similarities = similarities[filtered_indices]

    # Sort by similarity (descending) using negative for reverse sort
    sorted_order = np.argsort(-filtered_similarities)

    # Apply sort order to get indices in descending similarity order
    sorted_indices = filtered_indices[sorted_order]

    # Take top k
    return sorted_indices[:k]
