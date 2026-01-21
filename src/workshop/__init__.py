# Workshop module for RAG workshop
# Contains simplified chat loading, types, and LLM utilities

from workshop.chat import ChatContext, WhatsappMessage, load_whatsapp_chat, naive_token_counter
from workshop.structured_types import RAGResponse, RetrievalCoT, RetrievalResult
from workshop.types import LLMConfig

__all__ = [
    "WhatsappMessage",
    "load_whatsapp_chat",
    "naive_token_counter",
    "ChatContext",
    "LLMConfig",
    "RAGResponse",
    # Legacy aliases (deprecated)
    "RetrievalCoT",
    "RetrievalResult",
]
