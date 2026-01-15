# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Contextual chunker - Phase 5 advanced extension stub.

This chunker adds conversation-level context (summary) to chunk metadata for
contextual embeddings. This implements the Anthropic "contextual retrieval" pattern.

Workshop participants implement this as an advanced extension to learn about:
- Contextual embeddings for improved retrieval
- Conversation summarization
- Metadata enrichment for embedding context
- Engine-side context stripping before returning results
"""

from typing import Callable, Optional, Sequence

from pydantic import Field

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers.types import BaseChunkerParams
from workshop.rag.engines.types import ChunkObject


class ContextualChunkerParams(BaseChunkerParams):
    """
    Hyperparameters for contextual chunker.

    Inherits conversation windowing from BaseChunkerParams.
    Adds contextual embedding parameters.
    """

    base_chunk_length: int = Field(
        default=6,
        ge=1,
        le=50,
        json_schema_extra={"step": 1, "label": "Messages per Chunk"},
    )
    include_summary: bool = Field(
        default=True,
        json_schema_extra={"label": "Include Conversation Summary"},
    )


class ContextualChunker:
    """
    Chunker that adds conversation summary to chunk metadata for contextual embeddings.

    Key implementation considerations:
    - Generate conversation summary once (beginning of conversation)
    - Add summary to metadata["embedding_context"] for ALL chunks
    - Engine embeds chunk.text + metadata["embedding_context"]
    - Engine strips metadata["embedding_context"] before returning chunks to LLM

    Why this is useful:
    - Implements Anthropic's "contextual retrieval" pattern
    - Chunks embedded with global context improve retrieval accuracy
    - Example: chunk says "he resigned", summary says "conversation about CEO departure"
    - Without context: "he resigned" matches generic resignation queries
    - With context: "CEO resignation" correctly matches corporate news queries

    Implementation strategy:
    1. Generate conversation summary (simple: first N + last N messages, or LLM summary)
    2. Chunk messages normally (reuse MessageCountChunker logic)
    3. Add summary to metadata["embedding_context"] for each chunk

    Engine-side changes needed:
    - When embedding: text_to_embed = chunk.text + chunk.metadata.get("embedding_context", "")
    - When returning: strip metadata["embedding_context"] from results

    Example with base_chunk_length=3, include_summary=True:
    Messages:
      [0] "John submitted his resignation"
      [1] "The board accepted it"
      [2] "CEO search begins next week"
      [3] "What are the candidate profiles?"
      [4] "Looking for tech experience"

    Summary: "Discussion about CEO John's resignation and succession planning"

    Chunks:
      Chunk 0:
        text: "John submitted his resignation\\nThe board accepted it\\nCEO search begins next week"
        metadata["embedding_context"]: "Discussion about CEO John's resignation and succession planning"
      Chunk 1:
        text: "CEO search begins next week\\nWhat are the candidate profiles?\\nLooking for tech experience"
        metadata["embedding_context"]: "Discussion about CEO John's resignation and succession planning"

    When embedded:
      Chunk 0 embedding includes context - "CEO resignation discussion: John submitted..."
      Chunk 1 embedding includes context - "CEO resignation discussion: CEO search begins..."

    When retrieved:
      Chunk 0 returned WITHOUT embedding_context - only original text to LLM

    Participants need to:
    - Generate conversation summary (simple or LLM-based)
    - Reuse message-count chunking logic
    - Add embedding_context to ALL chunks
    - Understand that engines must handle context stripping
    """

    def __init__(
        self,
        params: Optional[ContextualChunkerParams] = None,
        summary_fn: Optional[Callable[[Sequence[WhatsappMessage]], str]] = None,
    ):
        """
        Initialize chunker with hyperparameters.

        Args:
            params: Hyperparameters (default: ContextualChunkerParams())
            summary_fn: Function to generate conversation summary (optional)
        """
        self.params = params or ContextualChunkerParams()
        self._summary_fn = summary_fn or self._default_summary

    def _default_summary(self, messages: Sequence[WhatsappMessage]) -> str:
        """
        Generate simple conversation summary.

        Simple version: "Conversation between {speakers} from {start_date} to {end_date}"
        Advanced version: LLM-generated summary of topics discussed

        Args:
            messages: Sequence of WhatsappMessage objects

        Returns:
            Summary string to add to chunk metadata

        TODO for participants: Implement summary generation
        """
        raise NotImplementedError(
            "Participants implement this. "
            "Simple approach: Extract speakers, dates, and format as string. "
            "Advanced approach: Use LLM to summarize first N and last N messages."
        )

    def chunk_messages(self, messages: Sequence[WhatsappMessage]) -> Sequence[ChunkObject]:
        """
        Transform messages into chunks with conversation context.

        Args:
            messages: Sequence of WhatsappMessage objects from conversation

        Returns:
            Sequence of ChunkObjects with:
            - text: Combined message text
            - message_ids: Indices in original message list
            - metadata: start_idx, end_idx, timestamps, speakers, embedding_context

        Implementation steps:
        1. Generate conversation summary (if include_summary=True)
        2. Chunk messages normally (message-count sliding window)
        3. For each chunk: add summary to metadata["embedding_context"]
        4. Engine will use embedding_context when embedding
        5. Engine will strip embedding_context before returning to LLM

        TODO for participants: Implement full pipeline
        """
        raise NotImplementedError(
            "Advanced extension. "
            "Hint: Reuse MessageCountChunker logic, generate summary once, "
            "add to metadata['embedding_context'] for all chunks."
        )
