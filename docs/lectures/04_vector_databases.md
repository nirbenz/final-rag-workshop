# Lecture 04: Vector Databases and Re-ranking

> **Phase:** Phase 4 - Scaling Retrieval

---

## Where We Are

```
                                  *** NEW ***          *** NEW ***
┌──────────┐   ┌───────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐
│ Chunking │-->│ Embedding │-->│ ANN INDEX │-->│ RE-RANKING │-->│ Prompting │--> LLM
└──────────┘   └───────────┘   └───────────┘   └────────────┘   └───────────┘
  Phase 1        Phase 3         (Qdrant)        (BM25/custom)     Phase 2
```

*Phase 4 architecture: ANN index replaces brute-force search, re-ranking adds lexical precision.*

Phase 3's brute-force search doesn't scale. We add speed (ANN) and precision (re-ranking).

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand why O(n) similarity search doesn't scale
2. Know how ANN algorithms trade accuracy for speed
3. Understand why dense retrieval has blind spots
4. Know how TF-IDF and BM25 complement embedding search

---

## Outline

### 1. The Scaling Problem

**Phase 3 Reality Check:**

```python
# For every query:
for chunk in all_chunks:  # O(n)
    similarity = cosine_similarity(query, chunk)
```

**At Scale:**

| Chunks    | Comparisons/Query | At 100 QPS   |
|-----------|-------------------|--------------|
| 1,000     | 1,000             | 100,000/sec  |
| 100,000   | 100,000           | 10M/sec      |
| 10M       | 10,000,000        | 1B/sec       |

**This doesn't scale.** We need sub-linear search.

---

### 2. Approximate Nearest Neighbor (ANN) Search

**The Key Insight:** Trade a little accuracy for massive speed gains.

**HNSW (Hierarchical Navigable Small Worlds):**

```
Instead of comparing to ALL 100k chunks:

1. Build a graph where similar vectors are neighbors
2. Start at a random entry point
3. Navigate greedily toward the query
4. Explore the local neighborhood
5. Return best matches found

Result: ~100-1000 comparisons instead of 100,000
        ~95%+ recall (you find most of the true nearest neighbors)
```

**Vector Databases** handle this for you: Qdrant, Pinecone, Weaviate, Chroma, Milvus

They manage:

- Index building and maintenance
- Fast ANN search
- Metadata filtering
- Persistence and scaling

**In our workshop:** Qdrant runs locally, no Docker needed.

---

### 3. The Blind Spot of Dense Retrieval

Embeddings are great at semantic similarity but have a weakness:

```
Query: "What did Sarah say about Paris?"

Dense retrieval finds:
  [x] "Let's plan a European vacation!"     (semantically similar)
  [x] "I love traveling to France"           (semantically similar)
  [ ] "Sarah: Paris is too expensive"        (exact keyword match -- MISSED!)
```

The query says "Paris" and the best chunk also says "Paris." But dense retrieval might rank semantically-similar-but-wrong chunks higher, because it matches on **meaning** not **words**.

**Note:** This is a different problem from the entrance code in Lecture 03. That was a
**chunking** problem (answer in a different chunk from the question). This is a **retrieval**
problem -- the right chunk exists, but dense search ranks the wrong one higher.

**The solution:** Combine dense (semantic) with sparse (keyword) retrieval.

---

### 4. TF-IDF and BM25: Smart Keyword Search

> **Slide guidance:** TF-IDF intuition + bar chart is one slide; BM25 description is one slide; Dense vs Sparse comparison table is one slide.

**TF-IDF: An Advanced Bag of Words**

Most of you have seen TF-IDF before. The intuition:

```
TF-IDF asks two questions about each word:

1. How often does this word appear in THIS chunk?     (Term Frequency)
2. How rare is this word across ALL chunks?            (Inverse Document Frequency)

Common words ("the", "is", "and") --> low IDF  --> low score
Rare words ("Paris", "deadline", "4569#")  --> high IDF  --> high score
```

```
         TF-IDF Scoring

 "the"   |  ##                              |  common word = low score
 "Sarah" |  ##########                      |  less common = medium score
 "Paris" |  ##################              |  rare word = high score
 "4569#" |  ##########################      |  unique term = highest score
         +----------------------------------+
```

*Bar chart: common words ("the") score low; rare domain-specific terms ("4569#") score highest.*

**BM25: TF-IDF's Popular Sibling**

BM25 (Best Matching 25) builds on TF-IDF with two practical improvements:

1. **Diminishing returns on term frequency** -- a word appearing 10x is not 10x better than 1x
2. **Length normalization** -- long chunks don't get unfair advantage over short ones

BM25 is the default algorithm in Elasticsearch, Solr, and most search engines.
It is the industry standard for keyword-based retrieval.

**Dense vs Sparse: Complementary Strengths**

```
                    Dense              Sparse
                    (Embeddings)       (BM25)
                    ┌──────────┐       ┌──────────┐
 "vacation"="trip"  │    Yes   │       │    No    │
                    ├──────────┤       ├──────────┤
 "Paris"="Paris"    │ Sometimes│       │  Always  │
                    ├──────────┤       ├──────────┤
 "4569#"="4569#"    │  Rarely  │       │  Always  │
                    ├──────────┤       ├──────────┤
 Understands intent │    Yes   │       │    No    │
                    └──────────┘       └──────────┘
```

*Dense retrieval understands meaning; sparse retrieval matches exact terms. They complement each other.*

**This is why production RAG systems combine both.**

---

### 5. Two-Stage Retrieval Pattern

**The Architecture:**

```
┌────────────────────────────────────────────────────────────┐
│                    TWO-STAGE RETRIEVAL                      │
│                                                            │
│  Query ──> [Stage 1: ANN Search] ──> 50-100 candidates    │
│                 FAST (~10ms)                               │
│                 Semantic matching                          │
│                                                            │
│        ──> [Stage 2: Re-rank]    ──> Top 5-10             │
│                 FAST!                                      │
│                 Keyword precision                          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

*Two-stage retrieval funnel: ANN casts a wide semantic net (50-100), re-ranker narrows to the best 5-10.*

> **Slide guidance:** The architecture diagram is one slide; the workshop code + Qdrant production code are a second slide.

**Why This Works:**

- Stage 1 (dense/ANN): Casts a wide semantic net quickly
- Stage 2 (sparse/BM25): Re-scores the small candidate set by exact keyword relevance
- Together: semantic breadth + lexical precision

**The Workshop's Implementation:**

```python
# In RAGContextEngine:

# 1. ANN search returns 50 candidates (dense)
candidates = self.qdrant_client.search(query_embedding, limit=50)

# 2. Re-ranker scores and returns top 10 (sparse)
results = rerank(query_text, candidates, top_k=10)
```

**In Production:** Vector databases like Qdrant handle the entire pipeline internally --
dense retrieval, sparse retrieval, and fusion -- all on the DB's compute, no Python loops:

```python
# Qdrant hybrid search -- dense + sparse in one query
client.query_points(
    collection_name="chunks",
    prefetch=[
        models.Prefetch(query=dense_embedding, using="dense", limit=20),
        models.Prefetch(query=sparse_embedding, using="sparse", limit=20),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
    limit=10,
)
```

Two prefetch stages (dense + sparse) run in parallel, then RRF merges the results.
This is the hybrid method in practice -- semantic breadth and lexical precision in a single call.

---

## What Changed: Phase 3 -> Phase 4

| Before (Phase 3) | After (Phase 4) |
|---|---|
| Brute-force O(n) search | ANN sub-linear search (Qdrant) |
| Dense retrieval only | Dense + sparse (re-ranking) |
| NumPy in-memory | Persistent vector database |

**Next up:** Phase 4 hands-on -- implement a re-ranking strategy.

---

## Instructor Notes

- Emphasize: "95% recall with HNSW means we might miss 5%, but we're 100x faster"
- Draw the two-stage funnel on the board
- BM25 demo: show a query where dense retrieval misses an exact keyword match
- TF-IDF: most people know this; frame it as context rather than teaching from scratch
- Reference back to the entrance code problem from Lecture 03
- Qdrant runs locally for this workshop (no Docker needed)

---

## Code References

- Engine: `src/workshop/rag/engines/qdrant.py`
- Re-ranking exercise: `src/workshop/rag/exercises/reranking.py`
- BM25 solution: `src/workshop/rag/solutions/reranking.py`

---

## Further Reading

- [HNSW Algorithm Explained](https://www.pinecone.io/learn/hnsw/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [BM25 (Robertson & Zaragoza)](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)
- [Hybrid Search at Qdrant](https://qdrant.tech/articles/hybrid-search/)
