"""Unit tests for the customer service agent."""

import pytest
from src.agent import DialogueAgent, SYSTEM_PROMPT


class TestDialogueAgent:
    """Test cases for DialogueAgent."""

    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = DialogueAgent()
        assert agent is not None
        assert agent.system_prompt == SYSTEM_PROMPT
        assert agent.conversation_history == []

    def test_reset_history(self):
        """Test conversation history can be reset."""
        agent = DialogueAgent()
        agent.add_message("user", "Hello")
        assert len(agent.conversation_history) == 1

        agent.reset_history()
        assert agent.conversation_history == []

    def test_add_message(self):
        """Test adding messages to history."""
        agent = DialogueAgent()
        agent.add_message("user", "Hello")
        agent.add_message("assistant", "Hi there!")

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0]["role"] == "user"
        assert agent.conversation_history[1]["role"] == "assistant"

    def test_get_messages(self):
        """Test getting all messages including system prompt."""
        agent = DialogueAgent()
        agent.add_message("user", "Hello")

        messages = agent.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_format_response(self):
        """Test response formatting."""
        agent = DialogueAgent()
        response = agent.format_response("Hello!")
        assert "response" in response
        assert "history_length" in response
        assert response["response"] == "Hello!"


@pytest.mark.asyncio
class TestChat:
    """Test cases for chat functionality."""

    async def test_chat_returns_response(self):
        """Test chat returns a response."""
        agent = DialogueAgent()
        response = await agent.chat("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    async def test_chat_adds_to_history(self):
        """Test chat adds user message to history."""
        agent = DialogueAgent()
        initial_length = len(agent.conversation_history)

        await agent.chat("Hello")

        assert len(agent.conversation_history) == initial_length + 1
        assert agent.conversation_history[-1]["role"] == "user"
        assert agent.conversation_history[-1]["content"] == "Hello"
