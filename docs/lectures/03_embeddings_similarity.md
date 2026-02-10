# Lecture 03: Embeddings and Semantic Search

> **Phase:** Phase 3 - Embedding-Based Retrieval

---

## Where We Are

```
                  *** NEW ***              *** NEW ***
┌──────────┐   ┌───────────┐   ┌──────────────────┐   ┌───────────┐
│ Chunking │-->│ EMBEDDING │-->│ SIMILARITY SEARCH│-->│ Prompting │--> LLM
└──────────┘   └───────────┘   └──────────────────┘   └───────────┘
  Phase 1        Phase 3            Phase 3              Phase 2
```

*Phase 3 architecture: embedding and similarity search replace naive retrieval.*

No more sending ALL chunks. Now we find the **relevant** ones.

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand how embeddings represent text as vectors
2. Know how cosine similarity measures semantic similarity
3. Recognize why chunking boundaries matter for retrieval quality
4. Be ready to implement similarity functions

---

## Outline

### 1. Text to Vectors

**What are embeddings?**

Neural networks compress meaning into dense vectors:

```
"Let's grab dinner tonight"    -> [0.2, 0.8, -0.1, 0.5, ...]  (1536 dims)
"What should we eat?"          -> [0.3, 0.7, -0.2, 0.6, ...]  (similar!)
"The quarterly report is due"  -> [-0.4, 0.1, 0.9, -0.3, ...]  (different)
```

*Same meaning produces similar vectors; unrelated content maps far apart in embedding space.*

> **Slide guidance:** The vector examples, similarity table, and key properties are each a separate slide.

**Similarity scores you'd actually see:**

| Pair | Cosine Similarity |
|------|-------------------|
| "Let's grab dinner" vs "What should we eat?" | ~0.85 |
| "Let's grab dinner" vs "The quarterly report is due" | ~0.12 |
| "See you at 3" vs "Meeting at 15:00" | ~0.78 |
| "lol" vs "that's hilarious" | ~0.65 |

**Key Properties:**

- Similar meanings = similar vectors = close in space
- Captures semantic relationships, not just keywords
- "dinner" and "eat" are close despite sharing no letters
- Learned from massive text corpora

**Visual:** Imagine a high-dimensional space where similar concepts cluster together. "dinner", "eat", "restaurant", "food" are all neighbors.

---

### 2. Why Chunking Boundaries Matter

**The Entrance Code Problem:**

Consider this real conversation in your WhatsApp chat:

```
User A:  Anyone knows what the new entrance code is?
User B:  4569#
```

If your chunker splits these into different chunks:

```
Chunk 7: [...previous messages..., "Anyone knows what the new entrance code is?"]
Chunk 8: ["4569#", ...next messages...]
```

Now when you ask: **"What is the new door code?"**

- The query embeds close to Chunk 7 (the question about the code)
- But the **answer** is in Chunk 8 (which looks like a random number)
- Chunk 8 on its own -- "4569#" -- has almost no semantic content

**It gets worse with real chats:**

What if the code was mentioned 3 times across the chat? Different people asking, different answers over time:

```
January:  "The code is 4569#"
March:    "They changed it to 7821#"
June:     "New code: 3344#"
```

Your retrieval needs to find ALL of them and figure out which is current.

**This is why chunking is not a solved problem.**

**Discussion: How would you fix this?**

| Approach | Helps? | Limitation |
|----------|--------|------------|
| Increase overlap | Partially -- the answer might land in the overlapping zone | More redundancy, no guarantee |
| Bigger chunks | Maybe -- but dilutes relevance for everything else | Back to "needle in haystack" |
| Smarter boundaries | Yes -- split at conversation gaps, not fixed windows | Needs data-aware chunking |

The overlap parameter is a band-aid, not a cure. We revisit this problem in Lecture 04 (retrieval perspective) and Lecture 05 (chunking solutions).

---

### 3. Cosine Similarity

**The Formula:**

```
cos(theta) = (A . B) / (||A|| x ||B||)
```

Where:

- `A . B` is the dot product
- `||A||` is the magnitude (L2 norm) of A

**Why Cosine?**

| Property      | Benefit                                        |
| ------------- | ---------------------------------------------- |
| Normalized    | Scale-invariant (vector length doesn't matter) |
| Range [-1, 1] | Easy to interpret and threshold                |
| Efficient     | Just dot product after normalization           |

**Intuition:** Cosine measures the **angle** between vectors, not the distance.

```
cos(0)   = 1.0   -> Identical direction (most similar)
cos(90)  = 0.0   -> Perpendicular (unrelated)
cos(180) = -1.0  -> Opposite direction (most dissimilar)
```

**Alternatives and why we don't use them here:**

- Euclidean distance: Affected by vector magnitude
- Dot product: Not normalized, hard to threshold

---

### 4. Bi-Encoder Architecture

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

*Bi-encoder: query and document pass through the same encoder independently, then similarity is computed between the resulting vectors.*

**Key Insight:** Query and documents are encoded independently.
This means we can pre-compute all document embeddings **once** and reuse them for every query.

---

### 5. Top-k Retrieval

**The Algorithm:**

```python
1. Embed the query
2. Compute similarity to ALL chunks    # O(n) -- we'll fix this in Phase 4
3. Filter by threshold (e.g., > 0.3)
4. Sort by similarity (descending)
5. Return top k
```

**Parameters to tune:**

| Parameter   | Purpose           | Typical Value |
| ----------- | ----------------- | ------------- |
| `threshold` | Minimum relevance | 0.2 - 0.5    |
| `k`         | Maximum results   | 5 - 20        |

**Question for the room:** This compares against ALL chunks. What happens with 10 million chunks?

(Answer in Phase 4: Approximate Nearest Neighbor search)

---

## What Changed: Phase 2 -> Phase 3

| Before (Phase 2) | After (Phase 3) |
|---|---|
| ALL chunks sent to LLM | Only **relevant** chunks |
| O(n) tokens per query | O(k) tokens per query (k << n) |
| LLM finds the needle | Retrieval finds the needle |
| Slow with large chats | Fast regardless of chat size |

**Next up:** Phase 3 hands-on -- implement `cosine_similarity()` and `get_top_k()`.

---

## Instructor Notes

- Use the WhatsApp examples (dinner/eat vs quarterly report) for the embedding demo
- The entrance code example is the key "aha moment" for why chunking matters -- draw it on the board
- Emphasize: "Similar meaning, not similar words"
- Show actual embedding dimensions (1536 for OpenAI, 768 for sentence-transformers)
- Keep bi-encoder explanation to one sentence if running short on time
- For the entrance code problem: ask the room "how would you fix this?" before moving on

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
