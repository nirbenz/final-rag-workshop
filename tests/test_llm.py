"""Tests for LLM module."""

from unittest.mock import MagicMock, patch

from pydantic import BaseModel
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
import pytest

from workshop.llm import (
    ModelMessageError,
    _add_history_handling,
)


class TestHistoryHandling:
    """Tests for history canonization logic."""

    def test_canonize_dict_user_message(self):
        """Dict user messages are converted to ModelRequest."""
        mock_agent = MagicMock()
        mock_agent.history_processors = []

        _add_history_handling(mock_agent)

        # Get the canonize function
        canonize = mock_agent.history_processors[0]

        messages = [{"role": "user", "content": "Hello"}]
        result = canonize(messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == "Hello"

    def test_canonize_dict_assistant_message(self):
        """Dict assistant messages are converted to ModelResponse."""
        mock_agent = MagicMock()
        mock_agent.history_processors = []

        _add_history_handling(mock_agent)
        canonize = mock_agent.history_processors[0]

        messages = [{"role": "assistant", "content": "Hi there!"}]
        result = canonize(messages)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].content == "Hi there!"

    def test_canonize_model_messages_passthrough(self):
        """ModelRequest/ModelResponse objects pass through unchanged."""
        mock_agent = MagicMock()
        mock_agent.history_processors = []

        _add_history_handling(mock_agent)
        canonize = mock_agent.history_processors[0]

        request = ModelRequest(parts=[UserPromptPart(content="test")])
        response = ModelResponse(parts=[TextPart(content="response")])

        result = canonize([request, response])

        assert result[0] is request
        assert result[1] is response

    def test_canonize_mixed_messages(self):
        """Mixed dict and ModelMessage types work correctly."""
        mock_agent = MagicMock()
        mock_agent.history_processors = []

        _add_history_handling(mock_agent)
        canonize = mock_agent.history_processors[0]

        request = ModelRequest(parts=[UserPromptPart(content="existing")])
        messages = [
            request,
            {"role": "assistant", "content": "dict response"},
            {"role": "user", "content": "dict user"},
        ]

        result = canonize(messages)

        assert len(result) == 3
        assert result[0] is request
        assert isinstance(result[1], ModelResponse)
        assert isinstance(result[2], ModelRequest)

    def test_canonize_unknown_type_raises(self):
        """Unknown message types raise ModelMessageError."""
        mock_agent = MagicMock()
        mock_agent.history_processors = []

        _add_history_handling(mock_agent)
        canonize = mock_agent.history_processors[0]

        with pytest.raises(ModelMessageError) as exc_info:
            canonize(["invalid string message"])

        assert "Unknown message type" in str(exc_info.value)

    def test_canonize_empty_list(self):
        """Empty message list returns empty list."""
        mock_agent = MagicMock()
        mock_agent.history_processors = []

        _add_history_handling(mock_agent)
        canonize = mock_agent.history_processors[0]

        result = canonize([])
        assert result == []


class TestModelMessageError:
    """Tests for ModelMessageError exception."""

    def test_exception_message(self):
        """Exception carries message correctly."""
        error = ModelMessageError("invalid message type")
        assert str(error) == "invalid message type"

    def test_exception_inheritance(self):
        """Exception is a proper Exception subclass."""
        error = ModelMessageError("test")
        assert isinstance(error, Exception)


class TestAgentCreation:
    """Tests for agent creation (mocked to avoid API calls)."""

    @patch("workshop.llm.pydantic_ai.Agent")
    @patch("workshop.llm.pydantic_ai.ModelSettings")
    def test_create_agent_calls_agent_constructor(self, mock_settings, mock_agent_class):
        """create_agent calls Agent with correct parameters."""
        from workshop.llm import create_agent

        mock_agent = MagicMock()
        mock_agent.history_processors = []
        mock_agent_class.return_value = mock_agent

        config = {"model_name": "openai:gpt-4o-mini", "kwargs": {"temperature": 0.7}}
        create_agent(config)

        mock_agent_class.assert_called_once()
        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["model"] == "openai:gpt-4o-mini"
        assert call_kwargs["output_type"] is str
        assert call_kwargs["retries"] == 5

    @patch("workshop.llm.pydantic_ai.Agent")
    @patch("workshop.llm.pydantic_ai.ModelSettings")
    def test_create_agent_with_structured_output(self, mock_settings, mock_agent_class):
        """create_agent passes structured output type."""
        from workshop.llm import create_agent

        class ResponseModel(BaseModel):
            answer: str

        mock_agent = MagicMock()
        mock_agent.history_processors = []
        mock_agent_class.return_value = mock_agent

        config = {"model_name": "openai:gpt-4o-mini", "kwargs": {}}
        create_agent(config, structured_output_type=ResponseModel)

        call_kwargs = mock_agent_class.call_args[1]
        assert call_kwargs["output_type"] == ResponseModel

    @patch("workshop.llm.pydantic_ai.Agent")
    @patch("workshop.llm.pydantic_ai.ModelSettings")
    def test_create_agent_adds_history_handling(self, mock_settings, mock_agent_class):
        """create_agent configures history processor."""
        from workshop.llm import create_agent

        mock_agent = MagicMock()
        mock_agent.history_processors = []
        mock_agent_class.return_value = mock_agent

        config = {"model_name": "openai:gpt-4o-mini", "kwargs": {}}
        result = create_agent(config)

        # History processor should be added
        assert len(result.history_processors) == 1


class TestGetPydanticAgent:
    """Tests for RAG agent creation (mocked)."""

    @patch("workshop.llm.create_agent")
    def test_get_pydantic_agent_calls_create_agent(self, mock_create_agent):
        """get_pydantic_agent delegates to create_agent."""
        from workshop.llm import get_pydantic_agent

        mock_agent = MagicMock()
        mock_agent.instructions = MagicMock(return_value=lambda f: f)
        mock_create_agent.return_value = mock_agent

        config = {"model_name": "openai:gpt-4o-mini", "kwargs": {}}
        get_pydantic_agent(config)

        mock_create_agent.assert_called_once_with(config, structured_output_type=None)

    @patch("workshop.llm.create_agent")
    def test_get_pydantic_agent_adds_instructions(self, mock_create_agent):
        """get_pydantic_agent adds system prompt instructions."""
        from workshop.llm import get_pydantic_agent

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        config = {"model_name": "openai:gpt-4o-mini", "kwargs": {}}
        get_pydantic_agent(config)

        # instructions decorator should be called
        mock_agent.instructions.assert_called_once()
