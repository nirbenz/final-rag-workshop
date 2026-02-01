# Lecture 02: Embeddings and Semantic Search

> **Duration:** 10 minutes
> **Phase:** Phase 3 - Embedding-Based Retrieval

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand how embeddings represent text as vectors
2. Know how cosine similarity measures semantic similarity
3. Understand bi-encoder architecture
4. Be ready to implement similarity functions

---

## Outline

### 1. Text to Vectors (3 minutes)

**What are embeddings?**

Neural networks compress meaning into dense vectors:

```
"I love pizza"     -> [0.2, 0.8, -0.1, 0.5, ...]  (1536 dimensions)
"Pizza is great"   -> [0.3, 0.7, -0.2, 0.6, ...]  (similar vector!)
"The weather is nice" -> [-0.4, 0.1, 0.9, -0.3, ...]  (different vector)
```

**Key Properties:**

- Similar meanings = similar vectors = close in space
- Learned from massive text corpora
- Capture semantic relationships, not just keywords

**Visual:** Imagine a 3D space where similar concepts cluster together.

---

### 2. Cosine Similarity (3 minutes)

**The Formula:**

```
cos(\theta) = (A   B) / (||A|| x ||B||)
```

Where:

- `A   B` is the dot product
- `||A||` is the magnitude (L2 norm) of A

**Why Cosine?**

| Property      | Benefit                                        |
| ------------- | ---------------------------------------------- |
| Normalized    | Scale-invariant (vector length doesn't matter) |
| Range [-1, 1] | Easy to interpret                              |
| Efficient     | Just dot product after normalization           |

**Alternatives:**

- Euclidean distance: Affected by vector magnitude
- Dot product: Not normalized, hard to threshold

**Intuition:** Cosine measures the angle between vectors, not the distance.

```
cos(0) = 1.0   -> Identical direction (most similar)
cos(90) = 0.0  -> Perpendicular (unrelated)
cos(180) = -1.0 -> Opposite direction (most dissimilar)
```

---

### 3. Bi-Encoder Architecture (2 minutes)

**How embedding models work:**

```
┌─────────────┐     ┌─────────────┐
│   Query     │     │  Document   │
│   Text      │     │   Text      │
└──────┬──────┘     └──────┬──────┘
       │                   │
       v                   v
┌─────────────┐     ┌─────────────┐
│  Encoder    │     │  Encoder    │
│  (same!)    │     │  (same!)    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       v                   v
   [query_vec]         [doc_vec]
       │                   │
       └───────┬───────────┘
               v
        cosine_similarity()
```

**Key Insight:** Query and documents are encoded independently.

**Advantages:**

- **Fast:** Pre-compute document embeddings once
- **Scalable:** Index millions of documents

**Disadvantage:**

- **Approximate:** Query and document never "see" each other

---

### 4. Top-k Retrieval (2 minutes)

**The Algorithm:**

```python
1. Embed the query
2. Compute similarity to ALL chunks
3. Filter by threshold (e.g., similarity > 0.3)
4. Sort by similarity (descending)
5. Return top k
```

**Parameters:**

| Parameter   | Purpose           | Typical Value |
| ----------- | ----------------- | ------------- |
| `threshold` | Minimum relevance | 0.2 - 0.5     |
| `k`         | Maximum results   | 5 - 20        |

**Question:** This is still O(n) - how do we scale to millions of documents?

(Answer in Phase 4: Approximate Nearest Neighbor search)

---

## Instructor Notes

- Draw vectors in 2D/3D to show similarity as proximity
- Show actual embedding dimensions (1536 for OpenAI, 768 for others)
- Emphasize: "Similar meaning, not similar words"
- Example: "king - man + woman = queen" (classic embedding demo)

---

## Slides

> **TODO:** Create presentation slides for this lecture

---

## Code References

- Exercise: `src/workshop/rag/exercises/similarity.py`
- Solution: `src/workshop/rag/solutions/similarity.py`
- Engine: `src/workshop/rag/engines/similarity.py`

---

## Further Reading

- [What are Embeddings?](https://vickiboykis.com/what_are_embeddings/)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
