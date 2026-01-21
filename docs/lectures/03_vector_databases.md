# Lecture 03: Vector Databases and Two-Stage Retrieval

> **Duration:** 8 minutes
> **Phase:** Phase 3A - Scaling Retrieval

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand why O(n) similarity search doesn't scale
2. Know how ANN algorithms trade accuracy for speed
3. Understand the two-stage retrieval pattern
4. Know common re-ranking approaches

---

## Outline

### 1. The Scaling Problem (2 minutes)

**Phase 2 Reality Check:**

```python
# For every query:
for chunk in all_chunks:  # O(n)
    similarity = cosine_similarity(query, chunk)
```

**At Scale:**

| Chunks | Comparisons/Query | At 100 QPS |
|--------|-------------------|------------|
| 1,000 | 1,000 | 100,000/sec |
| 100,000 | 100,000 | 10M/sec |
| 10M | 10,000,000 | 1B/sec |

**This doesn't scale.**

---

### 2. Approximate Nearest Neighbor (ANN) Search (3 minutes)

**The Key Insight:** Trade some accuracy for massive speed gains.

**Common Algorithms:**

| Algorithm | How It Works | Typical Recall |
|-----------|--------------|----------------|
| **HNSW** | Graph-based navigation | ~95%+ |
| **IVF** | Cluster-based lookup | ~90-95% |
| **PQ** | Compressed vectors | ~85-90% |

**HNSW (Hierarchical Navigable Small Worlds):**

```
Query: "What did John say about the meeting?"

Instead of comparing to ALL 100k chunks:
1. Start at random entry point in graph
2. Navigate to nearby nodes (greedy search)
3. Explore local neighborhood
4. Return best matches found

Result: ~100-1000 comparisons instead of 100,000
```

**Vector Databases:** Qdrant, Pinecone, Weaviate, Chroma, Milvus

They handle:
- Index building and maintenance
- Fast ANN search
- Metadata filtering
- Persistence and scaling

---

### 3. Two-Stage Retrieval Pattern (2 minutes)

**The Problem:** ANN is fast but approximate. We might miss the best matches.

**The Solution:** Two stages.

```
┌─────────────────────────────────────────────────────────┐
│                    TWO-STAGE RETRIEVAL                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query ──> [Stage 1: ANN Search] ──> 50-100 candidates  │
│                     FAST                                │
│                     ~10ms                               │
│                     May miss some                       │
│                                                         │
│            ──> [Stage 2: Re-ranker] ──> Top 10 results  │
│                     SLOW                                │
│                     ~100ms                              │
│                     Very accurate                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Why This Works:**

- Stage 1: Cast a wide net quickly
- Stage 2: Carefully pick the best from candidates

---

### 4. Re-ranking Approaches (1 minute)

**From Simple to Sophisticated:**

| Approach | Speed | Accuracy | Cost |
|----------|-------|----------|------|
| No re-ranking | Fastest | Lowest | Free |
| Keyword overlap | Fast | Medium | Free |
| Cross-encoder | Slow | High | Free |
| LLM scoring | Slowest | Highest | $$$ |
| Cohere Rerank | Medium | High | $ |

**Cross-Encoder vs Bi-Encoder:**

```
Bi-Encoder (embedding):     Cross-Encoder (re-ranking):
  Query    Document           Query + Document
    |          |                    |
    v          v                    v
 Encoder    Encoder              Encoder
    |          |                    |
    v          v                    v
 [vec_q]    [vec_d]             [score]

 Independent encoding          Joint encoding
 Fast, pre-computable          Slow, query-time only
```

---

## Instructor Notes

- Emphasize the 95% recall of HNSW - "We might miss 5%, but we're 100x faster"
- Draw the two-stage funnel
- Explain that Qdrant runs locally for this workshop (no Docker needed)
- Mention production options: Qdrant Cloud, Pinecone, etc.

---

## Slides

> **TODO:** Create presentation slides for this lecture

---

## Code References

- Engine: `src/workshop/rag/engines/qdrant.py`
- Re-ranking exercise: `src/workshop/rag/exercises/reranking.py`
- Re-ranking solution: `src/workshop/rag/solutions/reranking.py`

---

## Further Reading

- [HNSW Algorithm Explained](https://www.pinecone.io/learn/hnsw/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Two-Stage Retrieval at Google](https://research.google/pubs/pub37043/)
