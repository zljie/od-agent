"""Unit tests for the customer service agent."""

import pytest
from src.agent import CustomerServiceAgent


class TestCustomerServiceAgent:
    """Test cases for CustomerServiceAgent."""

    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = CustomerServiceAgent()
        assert agent is not None

    def test_agent_has_react_agent(self):
        """Test agent has ReActAgent instance."""
        agent = CustomerServiceAgent()
        assert hasattr(agent, 'agent')
        assert agent.agent is not None

    def test_agent_has_model(self):
        """Test agent has model configured."""
        agent = CustomerServiceAgent()
        assert hasattr(agent, 'model')
        assert agent.model is not None

    @pytest.mark.asyncio
    async def test_reset_history(self):
        """Test conversation history can be reset."""
        agent = CustomerServiceAgent()
        await agent.reset_history()
        assert agent.agent.memory is not None
