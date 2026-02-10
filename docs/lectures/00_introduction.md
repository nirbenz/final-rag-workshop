# Lecture 00: Introduction to RAG

> **Phase:** Opening

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand the limitations of LLMs that RAG addresses
2. Know the three components of RAG (Retrieval, Augmentation, Generation)
3. Understand why WhatsApp chats are a good learning dataset
4. See a demo of the finished system

---

## Outline

### 1. LLM Limitations

**Key Points:**

- **Knowledge cutoff**: LLMs only know what they were trained on
- **Context window limits**: Cannot process unlimited text
- **Hallucinations**: Confident but incorrect answers
- **No access to private data**: Your documents, conversations, databases

**Discussion Question:** When have you seen an LLM confidently give wrong information?

---

### 2. RAG as the Solution

**The Three Parts of RAG:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  RETRIEVAL  │ --> │ AUGMENTATION│ --> │ GENERATION  │
│             │     │             │     │             │
│ Find        │     │ Format      │     │ LLM creates │
│ relevant    │     │ context     │     │ grounded    │
│ information │     │ for prompt  │     │ response    │
└─────────────┘     └─────────────┘     └─────────────┘
```

*The three-part RAG pipeline: find information, format it as context, generate a grounded answer.*

**Benefits:**

- External knowledge access
- Grounded, verifiable responses
- Updateable without retraining
- Works with private data

**Today's Focus:** Retrieval is the bottleneck. Bad retrieval = bad answers.

---

### 3. Why WhatsApp Chats?

WhatsApp chats are messy in all the right ways:

- Abbreviations, emojis, multiple languages
- Temporal structure: when do topics change?
- Multiple speakers: who said what matters
- Personal connection: you care about your own conversations

---

### 4. The Full Pipeline

**What we are building today:**

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐
│ Chunking │-->│ Embedding │-->│ Indexing  │-->│ Retrieval │-->│ Re-ranking │-->│ Prompting │--> LLM
└──────────┘   └───────────┘   └──────────┘   └───────────┘   └────────────┘   └───────────┘
   Phase 1        Phase 3        Phase 4        Phase 3/4        Phase 4         Phase 2
```

*The six pipeline components and which workshop phase introduces each. Each phase "lights up" a new box. Phases reflect teaching order, not pipeline order -- we teach prompting (Phase 2) before retrieval (Phase 3) so you can evaluate improvements immediately.*

Each phase lights up a new component. By the end, you have a complete RAG system.

**Workshop Goal:**

- Build intuition for retrieval quality
- Understand chunking tradeoffs
- Implement semantic search from scratch
- Design effective RAG prompts
- Scale with vector databases and re-ranking

**Demo:** Show the finished Phase 4 system answering questions about a chat.

---

## What's Next: Phase 1

We start with the baseline -- a deliberately naive system that shows you what we are improving.

**Next up:** Lecture 01, then Phase 1 hands-on exploration.

---

## Instructor Notes

- Keep the demo short but impressive
- Use a chat with interesting/funny content
- Show both good and bad retrieval examples
- Emphasize: "The rest is just LLM calls -- retrieval is what we're learning"
- The pipeline diagram will reappear at each phase with the active component highlighted

---

## Further Reading

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Original RAG paper
- [Building RAG Applications](https://www.anthropic.com/news/contextual-retrieval) - Anthropic's guide
