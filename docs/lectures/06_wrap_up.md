# Lecture 06: Wrap-Up

> **Phase:** Closing

---

## The Journey

```
Phase 1: Baseline       "Look how bad this is"
  |
  v
Phase 2: Prompting      "Same retrieval, much better answers"
  |
  v
Phase 3: Embeddings     "Now we actually find relevant chunks"
  |
  v
Phase 4: Vector DB      "Scale it up, add precision"
```

*The four workshop phases, each building on the previous. One-line summary of what each taught.*

---

## The Full Pipeline -- All Components Active

```
┌──────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐
│ CHUNKING │-->│ EMBEDDING │-->│  INDEXING  │-->│ RETRIEVAL │-->│ RE-RANKING │-->│ PROMPTING │--> LLM
└──────────┘   └───────────┘   └───────────┘   └───────────┘   └────────────┘   └───────────┘
     |              |              |                 |               |                |
  Affects        Creates        Enables           Finds          Refines          Formats
  what can       semantic       fast ANN        candidates       accuracy         context
  be found       vectors        search
```

*Complete pipeline with all six components active and their roles labeled.*

Every component affects the final answer. There is no single "most important" piece --
they work as a system.

---

## Evaluating Retrieval Quality

> **Slide guidance:** Precision/recall example is one slide; the tradeoff table is a separate slide; eval set practice is a third.

After switching engines across phases -- how do you **know** it's better?
"It looks better in the UI" is not an answer you'd accept in production.

**Precision and Recall:**

```
All chunks in corpus: [A] [B] [C] [D] [E] [F] [G] [H] [I] [J]

Actually relevant:     [A]     [C]         [F]
Retrieved (top-5):     [A] [B] [C] [D]         [G]

Precision@5 = relevant in retrieved / retrieved     = 2/5 = 0.40
Recall@5    = relevant in retrieved / all relevant  = 2/3 = 0.67
```

*Concrete example: 10 chunks, 3 relevant, 5 retrieved. 2 overlap -- giving precision 0.40 and recall 0.67.*

**The Tradeoff:**

| Setting | Precision | Recall |
|---------|-----------|--------|
| Lower similarity threshold | Decreases | Increases |
| Higher top_k | Decreases | Increases |
| Tighter threshold | Increases | Decreases |
| Smaller top_k | Increases | Decreases |

**For RAG, recall usually matters more than precision.**
Missing a relevant chunk means the LLM cannot answer correctly.
Including an irrelevant chunk is less harmful -- a good prompt tells the LLM to ignore noise.

---

## Building an Eval Set (Production Practice)

```
1. Write 50-100 representative queries
2. Human-label which chunks are relevant for each
3. Run every config change against this eval set
4. Track precision@k and recall@k over time
```

This is the difference between "it feels better" and "retrieval precision went from 0.4 to 0.7."

Frameworks like [RAGAS](https://docs.ragas.io/) automate this.

---

## Tradeoffs Everywhere

| Decision             | Too Little           | Too Much          |
| -------------------- | -------------------- | ----------------- |
| Chunk size           | Loses context        | Loses precision   |
| Similarity threshold | Misses relevant      | Includes noise    |
| Re-rank candidates   | May miss best        | Slower processing |
| Prompt constraints   | Unpredictable output | Overly rigid      |

**Discussion:** Which tradeoff bit you today?

---

## What We Did Not Cover

- **Fine-tuning** embedding models for your domain
- **Evaluation automation** (RAGAS, faithfulness scores)
- **Query expansion** and HyDE (Hypothetical Document Embeddings)
- **Production concerns** (caching, rate limiting, cost optimization)
- **ColBERT** in production (introduced in Lecture 05, but production-grade late-interaction is beyond scope)
- **Agentic RAG** (multi-step retrieval, tool use)

---

## Your Next Steps

1. **Your own data:** Export and analyze your personal WhatsApp chats
2. **Advanced chunking:** Work through the optional segmenting exercise (take-home)
3. **Better re-ranking:** Try BM25 or cross-encoder if you used keyword overlap
4. **Evaluation:** Add RAGAS metrics to measure retrieval quality
5. **Production:** Deploy with managed vector database (Qdrant Cloud, Pinecone)

---

## Resources

- [Chunking Strategies Survey](https://arxiv.org/abs/2312.06648)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Qdrant Cloud](https://qdrant.tech/)
- [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/)
- [Cohere Rerank API](https://cohere.com/rerank)
- [ColBERT Paper (Khattab & Zaharia)](https://arxiv.org/abs/2004.12832)
- [RAGatouille Library](https://github.com/bclavie/RAGatouille)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

---

## Instructor Notes

- Keep this tight -- participants are tired
- The evaluation section is the most important "new" content; spend time here
- The tradeoffs table makes a good discussion prompt: "Which tradeoff bit you today?"
- End with energy: "You built a real RAG system today"
