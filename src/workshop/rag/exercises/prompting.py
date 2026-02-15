# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

# pyright: ignore
# ruff: noqa

"""
Prompt Engineering Exercise

This exercise teaches the "Generation" part of RAG - how to instruct the LLM
to use retrieved context effectively.

Your task: Implement get_system_prompt() to return a system prompt that:
1. Tells the LLM its role and task
2. Provides instructions for using the context
3. Specifies the output format (aligned with RAGResponse)
4. Handles edge cases (missing/irrelevant context)

The {context} placeholder in your prompt will be filled with retrieved chunks
by the LLM module (llm.py) at runtime.

Tips for good RAG prompts:
- Be explicit about using ONLY the provided context
- Ask for step-by-step reasoning to improve accuracy
- Request citations/quotes to ground responses in context
- Include fallback instructions for when context is insufficient

See the solution in solutions/prompting.py for a reference implementation.
"""


def get_system_prompt() -> str:
    """
    Design a system prompt template for the RAG assistant.

    The prompt should:
    1. Define the assistant's role (helpful, factual, grounded in context)
    2. Explain how to use the <context>...</context> block
    3. Specify the output format to match RAGResponse:
       - query_understanding: Restate the question
       - reasoning_steps: List of {thought, observation} steps
       - context_used: List of exact quotes from context
       - output: The final answer
       - confidence: high/medium/low
    4. Handle cases where context is insufficient

    The {context} placeholder will be replaced with retrieved chunks by llm.py.

    Example prompt structure:
        '''You are a helpful assistant...

        Instructions:
        - Use ONLY the provided context...
        - Think step-by-step...

        <context>
        {context}
        </context>

        When responding, provide:
        1. query_understanding: ...
        2. reasoning_steps: ...
        ...'''

    Returns:
        System prompt string with {context} placeholder

    TODO: Implement your system prompt
    TODO: REMOVE THE EXCEPTION
    """
    raise NotImplementedError(
        "Implement get_system_prompt.\n\n"
        "Your prompt should:\n"
        "1. Define the assistant's role\n"
        "2. Include <context>{context}</context> block\n"
        "3. Specify output format matching RAGResponse fields\n"
        "4. Handle missing context gracefully\n\n"
        "See solutions/prompting.py for reference."
    )
