# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

# pyright: ignore
# ruff: noqa

"""
Prompt Engineering Exercise - Phase 3

This exercise teaches the "Generation" part of RAG:
1. System prompt design for retrieval-augmented responses
2. Context formatting strategies
3. Output structure and constraints
4. Chain-of-thought prompting

After completing this exercise, you'll understand:
- How prompts shape LLM behavior in RAG systems
- Tradeoffs between concise vs detailed instructions
- When to use structured output vs free-form
- How to handle missing or irrelevant context

The RAG pipeline has three parts:
1. Retrieval (Phase 2-3): Finding relevant chunks
2. Augmentation: Formatting context into the prompt
3. Generation (this exercise): Instructing the LLM how to respond

Common prompt engineering patterns for RAG:
- Role assignment ("You are a helpful assistant...")
- Context boundaries (XML tags, delimiters)
- Output constraints ("Be concise", "Cite sources")
- Fallback instructions ("If context doesn't help...")
- Chain-of-thought ("First analyze the context, then...")
"""

from typing import Callable, Dict, Any


def get_system_prompt() -> str:
    """
    Design a system prompt for the RAG assistant.

    The system prompt sets the LLM's behavior and personality.
    It should instruct the model how to use the provided context.

    Your prompt will receive context in this format:
        <context>
        [chunk 1 text]

        ---

        [chunk 2 text]

        ---

        [chunk 3 text]
        </context>

    Guidelines for good RAG system prompts:
    1. Role: Give the assistant a clear identity
    2. Task: Explain what it should do with context
    3. Constraints: Set boundaries (length, tone, citations)
    4. Fallback: What to do if context is insufficient
    5. Format: How to structure the response

    Example approaches:

    Minimal (current baseline):
        "Use the following context to answer the question:
        <context>{context}</context>"

    Structured:
        "You are a helpful assistant. Answer based ONLY on the
        provided context. If the context doesn't contain the
        answer, say 'I don't have enough information.'
        <context>{context}</context>"

    Chain-of-thought:
        "You are analyzing a conversation history.
        First, identify which parts of the context are relevant.
        Then, synthesize an answer from those parts.
        Finally, provide your response with brief citations.
        <context>{context}</context>"

    Persona-based:
        "You are a personal assistant who knows this person well.
        Use the conversation history to answer questions about
        their life, preferences, and relationships.
        Be warm but factual. Don't make up information.
        <context>{context}</context>"

    Returns:
        System prompt string with {context} placeholder

    TODO: Implement your system prompt
    """
    raise NotImplementedError(
        "Implement get_system_prompt.\n"
        "Hint: Start with a role and task description. "
        "Add constraints for how to use the context. "
        "Include a fallback for when context is insufficient."
    )


def format_context(chunks_text: list[str]) -> str:
    """
    Format retrieved chunks into a context string for the prompt.

    The default formatting joins chunks with "---" separators.
    You can improve this by:
    - Adding chunk numbers for citation
    - Including metadata (timestamps, speakers)
    - Summarizing very long chunks
    - Highlighting key information

    Args:
        chunks_text: List of chunk text strings from retrieval

    Returns:
        Formatted context string to insert into the prompt

    Example formats:

    Simple (current baseline):
        "chunk 1 text\n\n---\n\nchunk 2 text"

    Numbered:
        "[1] chunk 1 text\n\n[2] chunk 2 text"

    With metadata (if available):
        "[Chunk 1 - Jan 15, 2024]\nchunk 1 text\n\n[Chunk 2 - Jan 16, 2024]\n..."

    Relevance-ordered:
        "Most relevant:\nchunk 1 text\n\nAlso relevant:\nchunk 2 text"

    TODO: Implement your context formatter
    """
    raise NotImplementedError(
        "Implement format_context.\n"
        "Hint: Consider how the LLM will parse this. "
        "Numbered chunks help with citations. "
        "Clear separators prevent confusion between chunks."
    )


def get_output_instructions() -> str:
    """
    Define instructions for how the LLM should structure its output.

    These instructions guide the format and style of responses.
    They work alongside (or instead of) structured output types.

    Options to consider:
    - Length constraints ("Be concise", "2-3 sentences")
    - Citation requirements ("Quote relevant parts")
    - Confidence indication ("If unsure, say so")
    - Format structure ("Start with a summary, then details")

    Returns:
        Instructions string to append to the system prompt

    Example instructions:

    Concise:
        "Keep your response under 100 words. Be direct."

    Cited:
        "When answering, quote the specific text that supports
        your response. Use quotation marks for direct quotes."

    Structured:
        "Format your response as:
        ANSWER: [direct answer]
        EVIDENCE: [supporting quotes from context]
        CONFIDENCE: [high/medium/low]"

    Conversational:
        "Respond naturally as if chatting with a friend.
        Don't be overly formal or robotic."

    TODO: Implement your output instructions
    """
    raise NotImplementedError(
        "Implement get_output_instructions.\n"
        "Hint: Think about what makes a good answer for chat history queries. "
        "Should it cite specific messages? Be conversational?"
    )


def build_full_prompt(context: str) -> str:
    """
    Combine all prompt components into the final system prompt.

    This function assembles:
    1. System prompt (role, task, constraints)
    2. Context (retrieved and formatted chunks)
    3. Output instructions (format, style)

    The final prompt is what the LLM sees as its system message.

    Args:
        context: Pre-formatted context string (already processed by format_context)

    Returns:
        Complete system prompt ready for the LLM

    Example structure:
        "[System prompt with role and task]

        <context>
        [formatted context]
        </context>

        [Output instructions]"

    TODO: Implement by combining your other functions
    """
    raise NotImplementedError(
        "Implement build_full_prompt.\n"
        "Hint: Call get_system_prompt() and get_output_instructions(), "
        "then combine them with the context."
    )
