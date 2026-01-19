# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Prompt Engineering - Reference Solutions

These are working implementations for the prompting exercise.
The solutions are intentionally simple baselines that participants can improve.

Design philosophy:
- Minimal but functional (not over-engineered)
- Clear structure that's easy to modify
- Demonstrates key concepts without complexity
"""

from typing import List


def get_system_prompt() -> str:
    """
    Baseline system prompt for RAG assistant.

    This prompt is intentionally simple - participants should improve it
    based on their specific use case (WhatsApp chat analysis).
    """
    return """You are a helpful assistant that answers questions based on conversation history.

Instructions:
- Use ONLY the provided context to answer questions
- If the context doesn't contain relevant information, say "I don't have enough information in the conversation history to answer that"
- Be concise but complete
- When referring to specific conversations, mention who said what

<context>
{context}
</context>"""


def format_context(chunks_text: List[str]) -> str:
    """
    Baseline context formatter with numbered chunks.

    Numbers help the LLM reference specific chunks and help users
    understand which parts of the conversation were used.
    """
    if not chunks_text:
        return "No relevant conversation history found."

    formatted_parts = []
    for i, text in enumerate(chunks_text, 1):
        formatted_parts.append(f"[Chunk {i}]\n{text}")

    return "\n\n---\n\n".join(formatted_parts)


def get_output_instructions() -> str:
    """
    Baseline output instructions.

    Kept minimal so participants can experiment with different styles.
    """
    return """When answering:
- Be direct and conversational
- If you quote from the context, indicate which chunk it came from
- If the answer involves multiple people, clarify who said what"""


def build_full_prompt(context: str) -> str:
    """
    Combine prompt components into final system message.

    Note: context is already formatted when passed here.
    """
    system = get_system_prompt()
    instructions = get_output_instructions()

    # Insert context into the system prompt template
    prompt_with_context = system.format(context=context)

    # Append output instructions
    return f"{prompt_with_context}\n\n{instructions}"
