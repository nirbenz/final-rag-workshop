# Lecture 03: Vector Databases, Re-ranking, and Evaluation

> **Duration:** 15 minutes
> **Phase:** Phase 4 - Scaling and Evaluating Retrieval

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand why O(n) similarity search doesn't scale
2. Know how ANN algorithms trade accuracy for speed
3. Understand the two-stage retrieval pattern (dense + sparse)
4. Know why BM25 complements embedding-based retrieval
5. Be able to reason about retrieval quality (precision, recall)

---

## Outline

### 1. The Scaling Problem (2 minutes)

**Phase 3 Reality Check:**

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

### 3. Dense vs Sparse Retrieval (3 minutes)

**The Blind Spot of Dense Retrieval:**

Embeddings are great at semantic similarity but have a fundamental weakness:

```
Query: "What did Sarah say about Paris?"

Dense retrieval finds:
  [x] "Let's plan a European vacation!"     (semantically similar)
  [x] "I love traveling to France"           (semantically similar)
  [ ] "Sarah: Paris is too expensive"        (exact match -- MISSED)
```

The query says "Paris" but the best chunk also says "Paris." Dense retrieval
might rank the semantically-similar-but-wrong chunks higher.

**BM25: Smart Keyword Search**

BM25 (Best Matching 25) is a sparse retrieval function based on term frequency:

```
Score(query, document) = sum over query terms of:
    IDF(term) * TF(term, doc) * (k1 + 1)
    ─────────────────────────────────────────
    TF(term, doc) + k1 * (1 - b + b * |doc| / avg_doc_len)
```

In plain English:
- **TF (term frequency):** Words that appear more in a chunk score higher
- **IDF (inverse document frequency):** Rare words score higher than common ones
- **Length normalization:** Long chunks don't get unfair advantage

**Why BM25 Still Matters:**

| Method | Finds "trip" when query says "vacation" | Finds "Paris" when query says "Paris" |
|--------|----------------------------------------|---------------------------------------|
| Dense (embeddings) | Yes | Sometimes |
| Sparse (BM25) | No | Always |
| **Hybrid (both)** | **Yes** | **Always** |

Research consistently shows hybrid retrieval outperforms either method alone.
This is why production RAG systems almost always combine dense and sparse.

---

### 4. Two-Stage Retrieval Pattern (2 minutes)

**The Solution:** Combine speed with accuracy using two stages.

```
┌─────────────────────────────────────────────────────────┐
│                    TWO-STAGE RETRIEVAL                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Query ──> [Stage 1: ANN Search] ──> 50-100 candidates  │
│                     FAST (dense)                        │
│                     ~10ms                               │
│                     Semantic matching                   │
│                                                         │
│            ──> [Stage 2: BM25 Re-rank] ──> Top 10       │
│                     FAST (sparse)                       │
│                     ~5ms                                │
│                     Lexical precision                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Why This Works:**

- Stage 1 (dense/ANN): Casts a wide semantic net quickly
- Stage 2 (sparse/BM25): Re-scores by exact keyword relevance
- Together: semantic breadth + lexical precision

**The Workshop's Implementation:**

```python
# In qdrant.py (RAGContextEngine):
# 1. ANN search returns 50 candidates (dense)
candidates = self.qdrant_client.search(query_embedding, limit=50)

# 2. Re-ranker scores and returns top 10 (sparse)
results = rerank(query_text, candidates, top_k=10)
```

**In Production: DB-Level Hybrid Search**

Our workshop implements re-ranking in Python for learning purposes. Production
vector databases handle the entire pipeline internally -- no application-level
re-ranking needed:

```python
# Qdrant's built-in multi-stage retrieval pipeline:
from qdrant_client import models

results = client.query_points(
    "my-collection",
    prefetch=[
        # Stage 1a: Dense retrieval (ANN)
        models.Prefetch(query=dense_vector, using="dense", limit=50),
        # Stage 1b: Sparse retrieval (BM25-like)
        models.Prefetch(query=sparse_vector, using="sparse", limit=50),
    ],
    # Stage 2: Fuse results with Reciprocal Rank Fusion
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

This runs entirely on the DB's compute -- no round trips, no Python loops.
Qdrant also supports late-interaction reranking (ColBERT-style) as a final
stage in the same pipeline. This is the pattern you would use in production.

---

### 5. Beyond BM25: ColBERT and Late Interaction (2 minutes)

**The Spectrum of Retrieval Models:**

```
Sparse (BM25)         Bi-Encoder          Cross-Encoder       ColBERT
    |                     |                    |                  |
 Keyword matching    Sentence vectors     Joint encoding    Token-level
 Very fast           Fast, pre-compute    Slow, accurate    Best of both
 No semantics        Good semantics       Best semantics    Great semantics
                                                            + lexical
```

**ColBERT (Contextualized Late Interaction over BERT):**

Instead of compressing each document into a single vector, ColBERT keeps
per-token embeddings and computes similarity at the token level:

```
Query:    "Paris trip"     ->  [vec_Paris, vec_trip]
Document: "vacation in Paris" -> [vec_vacation, vec_in, vec_Paris]

Score = sum of max similarities:
  vec_Paris  <-> max(vec_vacation, vec_in, vec_Paris) = vec_Paris  (1.0)
  vec_trip   <-> max(vec_vacation, vec_in, vec_Paris) = vec_vacation (0.8)
  Total = 1.8
```

This gives ColBERT both semantic understanding AND lexical precision.

**RAGatouille** is a Python library that makes ColBERT easy to use:

```python
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
results = RAG.search(query="Paris trip", k=10)
```

ColBERT is worth exploring post-workshop for production RAG systems where
retrieval quality is critical.

---

### 6. Evaluating Retrieval Quality (3 minutes)

**The Fundamental Question:**

After switching from NaiveContextEngine to SimilarityContextEngine to
RAGContextEngine -- how do you know it's actually better? "It looks better
in the UI" is not an answer you'd accept in production.

**Discussion: How Would You Measure This?**

Think about it: you run a query, you get back 5 chunks. Some are relevant,
some aren't. How do you quantify "good retrieval"?

**Precision and Recall:**

```
All chunks in corpus: [A] [B] [C] [D] [E] [F] [G] [H] [I] [J]

Actually relevant:     [A]     [C]         [F]
Retrieved (top-5):     [A] [B] [C] [D]         [G]

Precision@5 = relevant in retrieved / retrieved = 2/5 = 0.40
Recall@5    = relevant in retrieved / all relevant = 2/3 = 0.67
```

**The Tradeoff:**

| Setting | Effect on Precision | Effect on Recall |
|---------|--------------------|--------------------|
| Lower similarity threshold | Decreases | Increases |
| Higher top_k | Decreases | Increases |
| Tighter threshold | Increases | Decreases |
| Smaller top_k | Increases | Decreases |

**For RAG, recall usually matters more than precision.** Missing a relevant
chunk means the LLM can't answer correctly. Including an irrelevant chunk
is less harmful -- a good prompt tells the LLM to ignore noise.

**Building an Eval Set (Production Practice):**

1. Write 50-100 representative queries
2. For each query, human-label which chunks are relevant
3. Run every config change against this eval set
4. Track precision@k and recall@k over time

This is the difference between "it feels better" and "retrieval precision
went from 0.4 to 0.7."

**Workshop Exercise:**

Try the same 3 questions across all three engines. For each, note:
- How many of the returned chunks actually help answer the question?
- Did the engine miss any chunks you know are relevant?
- What's the rough precision and recall?

This manual process is exactly what you'd automate in production with
frameworks like RAGAS.

---

## Instructor Notes

- Emphasize the 95% recall of HNSW -- "We might miss 5%, but we're 100x faster"
- Draw the two-stage funnel on the board
- BM25 demo: show a query where dense retrieval misses an exact keyword match
- Evaluation discussion: pause and ask the room before showing the formulas
- Explain that Qdrant runs locally for this workshop (no Docker needed)
- ColBERT/RAGatouille: mention only, don't demo (post-workshop exploration)

---

## Slides

> **TODO:** Create presentation slides for this lecture

---

## Code References

- Engine: `src/workshop/rag/engines/qdrant.py`
- BM25 re-ranking solution: `src/workshop/rag/solutions/reranking.py`
- Re-ranking exercise: `src/workshop/rag/exercises/reranking.py`

---

## Further Reading

- [HNSW Algorithm Explained](https://www.pinecone.io/learn/hnsw/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [BM25 Original Paper (Robertson et al.)](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz)
- [ColBERT: Efficient and Effective Passage Search (Khattab & Zaharia)](https://arxiv.org/abs/2004.12832)
- [RAGatouille Library](https://github.com/bclavie/RAGatouille)
- [RAGAS Evaluation Framework](https://docs.ragas.io/)
- [Hybrid Search at Qdrant](https://qdrant.tech/articles/hybrid-search/)
