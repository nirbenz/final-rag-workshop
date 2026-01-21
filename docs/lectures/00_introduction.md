# Lecture 00: Introduction to RAG

> **Duration:** 15 minutes
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

### 1. LLM Limitations (5 minutes)

**Key Points:**

- **Knowledge cutoff**: LLMs only know what they were trained on
- **Context window limits**: Cannot process unlimited text
- **Hallucinations**: Confident but incorrect answers
- **No access to private data**: Your documents, conversations, databases

**Discussion Question:** When have you seen an LLM confidently give wrong information?

---

### 2. RAG as the Solution (5 minutes)

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

**Benefits:**

- External knowledge access
- Grounded, verifiable responses
- Updateable without retraining
- Works with private data

**Today's Focus:** Retrieval is the bottleneck. Bad retrieval = bad answers.

---

### 3. Why WhatsApp Chats? (3 minutes)

**Realistic challenges:**

- Messy data: abbreviations, emojis, multiple languages
- Temporal structure: when do topics change?
- Multiple speakers: who said what matters
- Personal connection: participants care about their own conversations

**Alternative datasets considered:**
- Documentation (too clean)
- News articles (no conversation structure)
- Code (different domain)

---

### 4. Workshop Goal (2 minutes)

**By the end of today:**

- Build intuition for retrieval quality
- Understand chunking tradeoffs
- Implement semantic search from scratch
- Design effective RAG prompts

**Demo:** Show the finished Phase 3 system answering questions about a chat.

---

## Instructor Notes

- Keep the demo short but impressive
- Use a chat with interesting/funny content
- Show both good and bad retrieval examples
- Emphasize: "The rest is just LLM calls - retrieval is what we're learning"

---

## Slides

> **TODO:** Create presentation slides for this lecture

---

## Further Reading

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Original RAG paper
- [Building RAG Applications](https://www.anthropic.com/news/contextual-retrieval) - Anthropic's guide
