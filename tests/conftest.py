"""Pytest configuration and shared fixtures for RAG workshop tests."""

from datetime import datetime
from pathlib import Path
import sys

from dotenv import load_dotenv
import pytest

# Load .env so API keys are visible to os.environ.get() in tests
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workshop.chat import WhatsappMessage


@pytest.fixture
def test_messages():
    """Create test WhatsApp messages."""
    messages = []
    for i in range(20):
        msg = WhatsappMessage(
            timestamp=datetime(2024, 1, 1, 10, i),
            user=f"User{i % 3}",
            text=f"Test message number {i}",
        )
        messages.append(msg)
    return messages


@pytest.fixture
def small_test_messages():
    """Create small set of test messages."""
    messages = []
    for i in range(10):
        msg = WhatsappMessage(
            timestamp=datetime(2024, 1, 1, 10, i),
            user=f"User{i % 2}",
            text=f"Message {i}",
        )
        messages.append(msg)
    return messages
