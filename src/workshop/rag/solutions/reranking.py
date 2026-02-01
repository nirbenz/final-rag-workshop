# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
BM25 Re-ranking Solution.

A lightweight BM25 (Best Matching 25) re-ranker for two-stage retrieval.
BM25 is a sparse/lexical scoring function that ranks chunks by query term
frequency, penalized by document length. It complements dense (embedding)
retrieval by catching exact keyword matches that embeddings miss.

No external dependencies -- pure Python + math stdlib.
"""

from collections import Counter
import math
from typing import Dict, List, Sequence

from workshop.rag.engines.types import ChunkObject


def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenization with lowercasing.

    Production systems would use proper tokenization (stemming, stopword
    removal, etc.) but this is sufficient for workshop demonstration.

    Args:
        text: Raw text to tokenize

    Returns:
        List of lowercase tokens
    """
    return text.lower().split()


def _bm25_score(
    query_terms: List[str],
    doc_terms: List[str],
    doc_freqs: Dict[str, int],
    num_docs: int,
    avg_doc_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    Compute BM25 score for a single document against query terms.

    BM25 formula per term:
        IDF(q) * tf(q,D) * (k1 + 1) / (tf(q,D) + k1 * (1 - b + b * |D| / avgdl))

    Args:
        query_terms: Tokenized query
        doc_terms: Tokenized document
        doc_freqs: Number of documents containing each term
        num_docs: Total number of documents in corpus
        avg_doc_len: Average document length in tokens
        k1: Term frequency saturation parameter (default 1.2)
        b: Length normalization parameter (default 0.75)

    Returns:
        BM25 relevance score (higher = more relevant)
    """
    score = 0.0
    doc_len = len(doc_terms)
    tf_map = Counter(doc_terms)

    for term in query_terms:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        df = doc_freqs.get(term, 0)
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_norm

    return score


def rerank(
    query: str,
    chunks: Sequence[ChunkObject],
    top_k: int = 5,
) -> List[ChunkObject]:
    """
    BM25 re-ranker for two-stage retrieval.

    Scores each candidate chunk using BM25 (term frequency + inverse document
    frequency with length normalization) and returns the top-k by score.

    This complements the dense ANN retrieval in Stage 1:
    - Stage 1 (dense/ANN): Good at semantic similarity ("trip" ~ "vacation")
    - Stage 2 (sparse/BM25): Good at exact keyword matching ("Paris" = "Paris")

    Together they catch both semantic and lexical relevance.

    Args:
        query: The user's search query
        chunks: Candidate chunks from ANN search (Stage 1)
        top_k: Number of top results to return

    Returns:
        Top-k chunks sorted by BM25 score (most relevant first)
    """
    if not chunks:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return list(chunks[:top_k])

    doc_terms_list = [_tokenize(chunk.text) for chunk in chunks]

    # Document frequency: how many docs contain each term
    doc_freqs: Dict[str, int] = {}
    for doc_terms in doc_terms_list:
        for term in set(doc_terms):
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    num_docs = len(chunks)
    avg_doc_len = sum(len(d) for d in doc_terms_list) / num_docs

    scored = []
    for chunk, doc_terms in zip(chunks, doc_terms_list):
        score = _bm25_score(query_terms, doc_terms, doc_freqs, num_docs, avg_doc_len)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
