# Workshop chat loading module
# Simplified WhatsApp message parsing using whatstk

import datetime
from pathlib import Path
import re
from typing import Callable, Optional, Sequence

from pydantic import BaseModel
from whatstk import df_from_whatsapp

# Emoji pattern for sanitization
EMOJI_PATTERN = (
    r"[\U0001F1E6-\U0001F1FF"  # flags
    r"\U0001F300-\U0001F5FF"  # misc symbols & pictographs
    r"\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F700-\U0001F77F"  # alchemical symbols
    r"\U0001F780-\U0001F7FF"  # geometric ext.
    r"\U0001F800-\U0001F8FF"  # supplemental arrows-C
    r"\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    r"\U0001FA00-\U0001FAFF"  # symbols & pictographs ext.-A
    r"\u2600-\u26FF"  # misc symbols
    r"\u2700-\u27BF"  # dingbats
    r"\u2190-\u21FF"  # arrows
    r"\u2300-\u23FF"  # misc technical
    r"\u2B00-\u2BFF"  # arrows etc.
    r"\u2264\u2265"  # math symbols
    r"\uFE0F\u200D]"  # VS16, ZWJ
)


class WhatsappMessage(BaseModel):
    """
    Simple message model for WhatsApp chat exports.

    No inheritance complexity - just the fields we need for the workshop.
    """

    id: Optional[int] = None
    timestamp: datetime.datetime
    user: Optional[str] = None
    text: str

    def timed_form(self, seconds: bool = False) -> str:
        """
        Format message with timestamp for display.

        Args:
            seconds: Include seconds in timestamp

        Returns:
            Formatted string with timestamp and message
        """
        fmt = "%y-%m-%d %H:%M:%S" if seconds else "%y-%m-%d %H:%M"
        ts = self.timestamp.strftime(fmt)
        return f"{ts} {self.compact_form()}".strip()

    def compact_form(self) -> str:
        """
        Format message without timestamp.

        Returns:
            Formatted string with user and text
        """
        body = self.text or ""
        prefix = f"{self.user}: " if self.user else ""
        return f"{prefix}{body}".strip()


def _clean_user(user: str) -> str:
    """
    Sanitize username by removing emojis and special characters.

    Args:
        user: Raw username from WhatsApp export

    Returns:
        Cleaned username
    """
    user_clean = re.sub(EMOJI_PATTERN, "", user)
    user_clean = user_clean.replace("~", "")
    user_clean = user_clean.strip()
    return user_clean


def _clean_text(text: str) -> str:
    """
    Sanitize message text.

    Args:
        text: Raw message text

    Returns:
        Cleaned message text
    """
    if text is None:
        return ""
    # Normalize unicode whitespace
    text = text.replace("\u202f", " ")  # narrow no-break space
    text = text.replace("\u200e", "")  # left-to-right mark
    text = text.replace("\u200f", "")  # right-to-left mark
    return text.strip()


def load_whatsapp_chat(path: str | Path) -> Sequence[WhatsappMessage]:
    """
    Load WhatsApp chat export using whatstk library.

    Args:
        path: Path to WhatsApp .txt export file

    Returns:
        Sequence of WhatsappMessage objects
    """
    df = df_from_whatsapp(str(path))

    messages = []
    for idx, row in df.iterrows():
        user = row.get("username")
        text = row.get("message")

        messages.append(
            WhatsappMessage(
                id=idx if isinstance(idx, int) else None,
                timestamp=row["date"],  # pyright: ignore[reportArgumentType]
                user=_clean_user(user) if user else None,
                text=_clean_text(text) if text else "",
            )
        )
    return messages


def naive_token_counter(text: str) -> int:
    """
    Simple word-based token estimate.

    For workshop purposes, a rough estimate is sufficient.
    Production should use tiktoken or similar.

    Args:
        text: Text to count tokens for

    Returns:
        Estimated token count
    """
    return len(text.split())


class ChatContext:
    """
    Context manager for WhatsApp chat conversations.

    Loads a chat export and provides filtering by token count or time window.
    This is used by the NiceGUI app to provide context to the LLM.
    """

    def __init__(
        self,
        chat_path: str,
        token_counter: Optional[Callable[[str], int]] = None,
        token_limit: int = 1_000_000,
        num_days: int = 1_000_000,
    ):
        """
        Initialize chat context from a WhatsApp export file.

        Args:
            chat_path: Path to WhatsApp .txt export file
            token_counter: Function to count tokens (default: naive_token_counter)
            token_limit: Maximum tokens in context window
            num_days: Maximum days of history to include
        """
        self.chat_path = chat_path
        self.token_counter = token_counter or naive_token_counter
        self._token_limit = token_limit
        self._num_days = num_days

        # Load messages
        self._all_messages = load_whatsapp_chat(chat_path)
        self._context: Optional[Sequence[WhatsappMessage]] = None

    def get_context(
        self,
        token_limit: int = -1,
        num_days: int = -1,
    ) -> Sequence[WhatsappMessage]:
        """
        Get filtered messages based on token limit or time window.

        Args:
            token_limit: Maximum tokens (-1 = use instance default)
            num_days: Maximum days of history (-1 = use instance default)

        Returns:
            Filtered sequence of messages
        """
        if token_limit != -1:
            self._token_limit = token_limit
        if num_days != -1:
            self._num_days = num_days

        # Filter by num_days
        if self._num_days < 1_000_000 and self._all_messages:
            end_time = self._all_messages[-1].timestamp
            start_time = end_time - datetime.timedelta(days=self._num_days + 1)
            self._context = [m for m in self._all_messages if m.timestamp >= start_time]
        else:
            self._context = list(self._all_messages)

        return self._context

    @property
    def context(self) -> Sequence[WhatsappMessage]:
        """Get the current filtered context."""
        if self._context is None:
            self._context = list(self._all_messages)
        return self._context

    @property
    def text_context(self) -> str:
        """Get context as a single text string."""
        return "\n".join([m.timed_form() for m in self.context])

    @property
    def start_time(self) -> datetime.datetime:
        """Get timestamp of first message in context."""
        return self.context[0].timestamp

    @property
    def end_time(self) -> datetime.datetime:
        """Get timestamp of last message in context."""
        return self.context[-1].timestamp

    @property
    def num_tokens(self) -> int:
        """Get estimated token count for context."""
        return self.token_counter(self.text_context)

    @property
    def num_messages(self) -> int:
        """Get number of messages in context."""
        return len(self.context)

    def __json__(self) -> dict:
        """JSON serialization support."""
        return {"__chatcontext__": self.chat_path}
