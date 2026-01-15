# Created by Nir Ben-Zvi
# As part of workshop project
# me@nirbnzvi.com

"""
Sentence-boundary-aware chunker - Phase 2 stub.

This chunker respects sentence boundaries when creating chunks, avoiding
mid-sentence splits that can hurt embedding quality.

Workshop participants implement this to learn about:
- Sentence segmentation with spaCy or NLTK
- Token-based chunking with soft boundaries
- Overlap by sentence count rather than message count
"""

from typing import Optional, Sequence

from pydantic import Field

from workshop.chat import WhatsappMessage
from workshop.rag.chunkers.types import BaseChunkerParams
from workshop.rag.engines.types import ChunkObject


class SentenceBoundaryParams(BaseChunkerParams):
    """
    Hyperparameters for sentence-boundary chunker.

    Inherits conversation windowing from BaseChunkerParams.
    Adds sentence-aware chunking parameters.
    """

    max_chunk_tokens: int = Field(
        default=500,
        ge=100,
        le=2000,
        json_schema_extra={"step": 50, "label": "Max Tokens per Chunk"},
    )
    overlap_sentences: int = Field(
        default=4,
        ge=0,
        le=10,
        json_schema_extra={"step": 1, "label": "Overlap Sentences"},
    )


class SentenceBoundaryChunker:
    """
    Sentence-aware chunker that respects sentence boundaries.

    Key implementation considerations:
    - Use spaCy/NLTK for sentence segmentation
    - Messages can be split mid-text if they contain multiple sentences
    - Chunks accumulate sentences until token limit is hit
    - Last N sentences from previous chunk overlap with next chunk

    Implementation strategy:
    1. Extract all sentences from all messages, tracking message_id for each sentence
    2. Accumulate sentences into chunks until token limit
    3. Track which message IDs contributed to each chunk
    4. Add overlap by including last N sentences from previous chunk

    Example with max_tokens=200, overlap_sentences=2:
    Messages: ["Hi! How are you?", "I'm good. What about you?", "Great!"]
    Sentences: ["Hi!", "How are you?", "I'm good.", "What about you?", "Great!"]
    Chunk 0: ["Hi!", "How are you?", "I'm good."]  # ~150 tokens
    Chunk 1: ["I'm good.", "What about you?", "Great!"]  # Overlap 2 sentences

    Participants need to:
    - Load spaCy model or use NLTK sentence tokenizer
    - Count tokens per sentence (use tiktoken or approximate)
    - Accumulate sentences while tracking total tokens
    - Handle overlap by sentence boundary
    - Track message_ids for traceability
    """

    def __init__(self, params: Optional[SentenceBoundaryParams] = None):
        """
        Initialize chunker with hyperparameters.

        Args:
            params: Hyperparameters (default: SentenceBoundaryParams())
        """
        self.params = params or SentenceBoundaryParams()

    def chunk_messages(self, messages: Sequence[WhatsappMessage]) -> Sequence[ChunkObject]:
        """
        Transform messages into chunks respecting sentence boundaries.

        Args:
            messages: Sequence of WhatsappMessage objects from conversation

        Returns:
            Sequence of ChunkObjects with:
            - text: Combined sentence text (sentences may span messages)
            - message_ids: Indices of messages that contributed sentences
            - metadata: start_idx, end_idx, timestamps, speakers

        Implementation steps:
        1. Segment all messages into sentences, tracking (sentence_text, message_idx) tuples
        2. Initialize empty chunk, token_count = 0
        3. For each sentence:
           - Count tokens in sentence
           - If adding sentence exceeds max_tokens:
             - Save current chunk
             - Start new chunk with overlap (last N sentences from previous)
           - Add sentence to current chunk
        4. Save final chunk

        TODO for participants: Implement the full logic
        """
        raise NotImplementedError(
            "Participants implement this. "
            "Hints: Use spacy.load('en_core_web_sm') for sentence segmentation, "
            "tiktoken for token counting, and track message_ids for each sentence."
        )
