# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Prompt Engineering - Reference Solutions

The system prompt defines how the LLM should behave when answering questions.
It includes:
- Role and task description
- Instructions for using context
- Output format requirements (aligned with RAGResponse structured output)

The {context} placeholder is filled with retrieved chunks at runtime by llm.py.
"""


def get_system_prompt() -> str:
    """
    System prompt template for RAG assistant.

    This prompt guides the LLM to:
    1. Use only the provided context
    2. Think step-by-step through the context
    3. Quote relevant parts as evidence
    4. Provide a clear answer with confidence level

    The {context} placeholder will be filled with retrieved chunks by llm.py.

    Returns:
        System prompt template string with {context} placeholder
    """
    return """You are a helpful assistant that answers questions based on conversation history.

Your task is to analyze the provided context and answer the user's question.

Instructions:
- Use ONLY the provided context to answer questions
- Think step-by-step through the context to find relevant information
- Always quote the specific parts that support your answer
- If the context doesn't contain relevant information, say so honestly
- Be direct and conversational in your responses

<context>
{context}
</context>

When responding, you must provide:
1. query_understanding: Restate what the user is asking in your own words
2. reasoning_steps: Your step-by-step analysis (at least one step with thought and observation)
3. context_used: List of exact quotes from the context that support your answer
4. output: Your final answer to the question
5. confidence: "high" if context directly answers, "medium" if inferred, "low" if uncertain"""
