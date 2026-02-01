# Lecture 05: Advanced Chunking Strategies

> **Duration:** 15 minutes
> **Phase:** Optional / Take-Home - Data-Aware Chunking

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand why fixed-window chunking is problematic
2. Know data-driven approaches to determine chunk boundaries
3. Understand time-based segmentation
4. Be aware of advanced strategies (semantic, contextual)

---

## Outline

### 1. The Fixed-Window Problem (3 minutes)

**What Goes Wrong:**

```
Actual Conversation:                    Fixed-Window Chunks:

Morning topic: Planning trip            [Chunk 1: Trip planning + random]
  "Let's go to Paris"                   [Chunk 2: Random + Dinner plans]
  "I found cheap flights"               [Chunk 3: Dinner + Project work]
  "Book for May?"

Evening topic: Dinner plans             Topics get SPLIT across chunks!
  "Where should we eat?"
  "How about Italian?"

Next day: Project work
  "Did you finish the report?"
  "Almost done"
```

**Problems:**

1. **Cuts mid-conversation**: Topic split between chunks
2. **Mixes unrelated content**: Different topics in same chunk
3. **Overlap is a hack**: Creates redundancy, doesn't fix boundary issues

---

### 2. Data-Driven Chunking Insights (4 minutes)

**Key Question:** How do we find natural conversation boundaries?

**Signal 1: Time Gaps**

```
Messages over time:

|----|||--|||----|--||--|||---|  ...  |--|||-|
      ^         ^              ^
   Active    Active         6-hour
   period    period           gap
                           (new segment!)
```

**Insight:** Silence often indicates topic change.

**Signal 2: Time Gap Histogram**

```
Gap Distribution:
|
|  *
|  *  *
|  *  *  *
|  *  *  *  *
|  *  *  *  *  *        *
+--1m-5m-1h-6h-12h-24h-----> gap duration

Bimodal = natural breakpoints
```

**Signal 3: Mean Messages Before Silence**

```python
# Approximate "natural conversation length"
avg_messages_before_gap = total_messages / num_gaps_over_threshold
# Use this as chunk_length hint!
```

---

### 3. Segmentation Strategies (4 minutes)

**Strategy 1: Time-Based Segmentation**

```
Rule: Start new segment when gap > N hours

Messages: [M1, M2, M3, ..., M10] [gap: 8 hours] [M11, M12, ..., M20]
                    |                                      |
             Segment 1                              Segment 2

Then chunk WITHIN segments (never cross boundaries)
```

**Parameters:**
- `time_gap_hours`: Threshold for new segment (try 4-8 hours)

**Strategy 2: Semantic Segmentation**

```
Rule: Start new segment when embedding similarity drops

Messages:  M1 -- M2 -- M3 -- M4 -- M5 -- M6 -- M7
Similarity:  0.9   0.85  0.3   0.88  0.82  0.4
                        ^                   ^
                   Topic change!       Topic change!

Segments: [M1-M3] [M4-M6] [M7...]
```

**Parameters:**
- `similarity_threshold`: Minimum similarity to stay in segment

**Strategy 3: Speaker-Based (Heuristic)**

```
Rule: New participant entering might indicate new context

Speakers: [John, Sarah, John, Sarah] [+Mike joins] [Mike, Sarah, John]
                                          ^
                                  Possible segment break
```

---

### 4. Hierarchical Chunking (2 minutes)

**The Pattern:**

```
Step 1: Segment by time gaps
        [Segment A: 15 msgs] [Segment B: 8 msgs] [Segment C: 22 msgs]

Step 2: Chunk within segments
        [A1: 6] [A2: 6] [A3: 6]   [B1: 8]   [C1: 6] [C2: 6] [C3: 6] [C4: 6]
        (with overlap)           (no split)  (with overlap)

Result: Chunks respect conversation boundaries!
```

**Benefits:**

- Topics stay together
- Overlap within segments only
- Variable segment sizes (content-aware)

---

### 5. Advanced Strategies (Lecture Only) (2 minutes)

**Contextual Chunking (Anthropic Pattern):**

```
1. Generate conversation summary once
2. Prepend summary to each chunk for embedding
3. Store original chunk text only

Embedding: "Summary: Friends planning a trip... | John: Let's go to Paris"
Storage:   "John: Let's go to Paris"
```

**Benefit:** Chunks understand global context for better retrieval.

**Day-Level Aggregation:**

```
After top-k retrieval:
1. Count which day has most matching chunks
2. Return entire day instead of scattered chunks
3. Better coherence, more context
```

**Hybrid Retrieval:**

```
BM25 (keyword match) + Semantic Search

Reciprocal Rank Fusion

Combined results
```

**Query Expansion:**

```
Original: "What about the trip?"
Expanded: "vacation travel holiday journey Paris flights booking"
```

---

## Instructor Notes

- Show actual time-gap histogram from sample data if possible
- Draw the segmentation visually
- Emphasize: "Let the data tell you where to split"
- Mention: These strategies can be combined

---

## Slides

> **TODO:** Create presentation slides for this lecture

---

## Code References

- Exercise: `src/workshop/rag/exercises/segmenting.py`
- Solution: `src/workshop/rag/solutions/segmenting.py`
- Chunker: `src/workshop/rag/chunkers/segmenting.py`
- Semantic chunker: `src/workshop/rag/chunkers/semantic.py`
- Contextual chunker: `src/workshop/rag/chunkers/contextual.py`

---

## Further Reading

- [Chunking Strategies for RAG](https://arxiv.org/abs/2312.06648)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex Chunking Guide](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/)
