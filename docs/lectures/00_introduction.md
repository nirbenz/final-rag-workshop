# Lecture 00: Introduction to LLMs and Retrieval

> **Phase:** Opening

---

## Learning Objectives

By the end of this lecture, participants will:

1. Understand how LLMs and embedding models are trained (next-token vs masked prediction)
2. Understand the limitations of LLMs that RAG addresses
3. Know the three components of RAG (Retrieval, Augmentation, Generation)
4. Understand why WhatsApp chats are a good learning dataset
5. See a demo of the finished system

---

## Outline

### 1. How LLMs Work

> **Slide guidance:** Each sub-section (1A-1D) is a separate slide. The punchline ("Two Branches") is its own closing slide.

#### 1A. Next Token Prediction

Everything an LLM does reduces to one operation: given a sequence of words, predict the next one.

*Diagram: A sentence "The cat sat on the ___" with an arrow pointing to a probability distribution bar chart. Bars show candidate next tokens with probabilities: "mat" 0.35, "floor" 0.22, "couch" 0.15, "table" 0.08, ... trailing off. The highest-probability token is highlighted.*

The model picks a token, appends it to the sequence, and repeats. This autoregressive loop is how ChatGPT, Claude, and Gemini generate text -- one token at a time, each conditioned on everything before it.

**Key point:** This is ALL it does. Chat, code generation, reasoning -- all emergent from next-token prediction at scale.

---

#### 1B. Training at Scale

Where does the "knowledge" come from? The model sees trillions of tokens during pre-training and compresses the patterns into its weights.

*Diagram: A funnel. At the top, wide, labeled data sources feed in: "Web Crawl (~60%)", "Books (~16%)", "Code (~5%)", "Wikipedia (~3%)", "Forums, Q&A, Other (~16%)". The funnel narrows into a box labeled "Transformer Model" at the bottom. A label on the side: "Early models: ~10B tokens. Modern models: 10-15T tokens."*

The model does not memorize text. It learns statistical relationships: what follows what, in what context, with what likelihood.

**Fine-tuning (one sentence):** After pre-training, RLHF and instruction tuning make the model conversational -- but the knowledge was already in the weights from pre-training.

---

#### 1C. Masked Training: A Different Approach

Not all models predict the next token. BERT-family models predict **missing** tokens.

*Diagram: Two panels side by side. Left panel labeled "Autoregressive (GPT-style)": a sentence "The cat sat on the ___" with arrows flowing only left-to-right, ending at a blank. Right panel labeled "Masked (BERT-style)": the sentence "The [MASK] sat on the [MASK]" with arrows pointing inward from BOTH sides toward each mask token. The key visual difference: left-to-right only vs. bidirectional.*

BERT masks random tokens in a sentence and trains the model to reconstruct them using context from **both sides** -- left and right. This produces models that deeply understand meaning but cannot generate text.

**Why this matters:** Bidirectional understanding is the foundation for embedding models -- the "R" side of RAG.

---

#### 1D. From Masked Models to Embeddings

A trained BERT model produces a vector for every token. Pool those into a single vector per sentence and you have an **embedding** -- a numerical fingerprint of meaning.

*Diagram: Two sentences ("Let's grab dinner" and "What should we eat?") each pass through a box labeled "Same Encoder (BERT)". Each produces a vector (arrow down). Between the two vectors, a double-headed arrow labeled "similarity score". Below, a training signal: a green pair labeled "similar -- pull together" and a red pair labeled "dissimilar -- push apart".*

Raw BERT embeddings are not good for similarity out of the box. They need **contrastive fine-tuning**: train the model to pull similar sentences close in vector space and push dissimilar ones apart. Sentence-BERT (2019) was the breakthrough that made this practical.

**This is exactly what powers Phase 3 of the workshop.**

---

#### The Punchline: Two Branches, One Architecture

Both model families share the same transformer backbone, trained differently:

*Diagram: A tree or fork. Single trunk labeled "Transformer Architecture" splits into two branches. Left branch: "Next-token prediction" leading to "GPT, Claude, Gemini" leading to "The G in RAG (Generation)". Right branch: "Masked prediction + Contrastive training" leading to "Sentence-BERT, OpenAI Embeddings" leading to "The R in RAG (Retrieval)". Both branches reconnect at the bottom into "RAG: Retrieval-Augmented Generation".*

You will work with both sides today. Now let's understand why generation alone is not enough.

---

### 2. LLM Limitations

**Key Points:**

- **Knowledge cutoff**: LLMs only know what they were trained on
- **Context window limits**: Cannot process unlimited text
- **Hallucinations**: Confident but incorrect answers
- **No access to private data**: Your documents, conversations, databases

**Discussion Question:** When have you seen an LLM confidently give wrong information?

---

### 3. RAG as the Solution

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

### 4. Why WhatsApp Chats?

WhatsApp chats are messy in all the right ways:

- Abbreviations, emojis, multiple languages
- Temporal structure: when do topics change?
- Multiple speakers: who said what matters
- Personal connection: you care about your own conversations

---

### 5. The Full Pipeline

> **Slide guidance:** The pipeline diagram is one slide; the workshop goals list is a second slide. Demo is live, not a slide.

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

- Section 1 (How LLMs Work) is skippable if the audience already has ML background -- gauge the room
- The "two branches" punchline is the key slide; even if you skip the details, land this one
- 1A and 1D are load-bearing for the rest of the workshop; 1B and 1C can be compressed
- Keep the demo short but impressive
- Use a chat with interesting/funny content
- Show both good and bad retrieval examples
- Emphasize: "The rest is just LLM calls -- retrieval is what we're learning"
- The pipeline diagram will reappear at each phase with the active component highlighted

---

## Further Reading

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - Original RAG paper
- [Building RAG Applications](https://www.anthropic.com/news/contextual-retrieval) - Anthropic's guide
- [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762) - The transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Masked language modeling
- [Sentence-BERT (Reimers & Gurevych)](https://arxiv.org/abs/1908.10084) - Contrastive training for sentence embeddings
