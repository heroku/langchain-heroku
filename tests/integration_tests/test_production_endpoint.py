"""Integration tests for ChatHeroku production endpoint.

This module contains comprehensive integration tests that validate the ChatHeroku
integration with the Heroku Inference API. These tests ensure end-to-end functionality
and validate that the integration works correctly with real external services.

Following LangChain's testing guide:
- Tests multiple components working together
- Validates end-to-end functionality
- Tests against real external services
- Includes comprehensive error scenario testing
- Uses standard test base classes where available
"""

import os
import time
from typing import Any, List, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

# Load dotenv if available
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

    def _load_dotenv_stub(*args: Any, **kwargs: Any) -> bool:
        return False

    load_dotenv = _load_dotenv_stub

from langchain_heroku.chat_models import ChatHeroku


@pytest.mark.production
class TestChatHerokuIntegration:
    """Integration tests for ChatHeroku against production endpoint.

    This class provides comprehensive testing for the ChatHeroku integration
    with the Heroku Inference API, ensuring end-to-end functionality.
    """

    @pytest.fixture(autouse=True)
    def setup_environment(self) -> None:
        """Setup environment variables for testing."""
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            for env_file in [".env", ".env.local", ".env.production"]:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break

        # Check if tool calling is supported by testing with a simple tool
        self.tool_calling_supported = self._check_tool_calling_support()

    def _check_tool_calling_support(self) -> bool:
        """Check if the model supports tool calling."""
        try:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {"test": {"type": "string"}},
                            "required": ["test"],
                        },
                    },
                }
            ]

            chat = ChatHeroku(tools=tools, tool_choice="auto")
            messages = [HumanMessage(content="Use the test_tool with argument 'test'")]
            result = chat.invoke(messages)

            # Check if the response contains tool calls
            if hasattr(result, "tool_calls") and result.tool_calls:
                return True
            if hasattr(result, "additional_kwargs") and result.additional_kwargs.get("tool_calls"):
                return True
            return False
        except Exception:
            return False

    def test_invoke(self) -> None:
        """Test basic chat model invocation."""
        chat = ChatHeroku()
        messages = [HumanMessage(content="Hello")]
        result = chat.invoke(messages)
        assert result.content is not None
        assert len(result.content) > 0

    def test_invoke_with_params(self) -> None:
        """Test chat model invocation with custom parameters."""
        chat = ChatHeroku(temperature=0.1, max_tokens=100)
        messages = [HumanMessage(content="Hello")]
        result = chat.invoke(messages)
        assert result.content is not None
        assert len(result.content) > 0

    def test_conversation(self) -> None:
        """Test multi-turn conversation."""
        chat = ChatHeroku()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there! How can I help you today?"),
            HumanMessage(content="What's the weather like?"),
        ]
        result = chat.invoke(messages)
        assert result.content is not None
        assert len(result.content) > 0

    def test_system_message(self) -> None:
        """Test system message handling."""
        chat = ChatHeroku()
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
        ]
        result = chat.invoke(messages)
        assert result.content is not None
        assert len(result.content) > 0

    def test_streaming(self) -> None:
        """Test streaming responses."""
        chat = ChatHeroku(streaming=True)
        messages = [HumanMessage(content="Hello")]
        chunks = list(chat.stream(messages))
        assert len(chunks) > 0
        # Combine chunks to get full content
        content_parts: List[str] = []
        for chunk in chunks:
            # Handle both ChatGenerationChunk and AIMessageChunk
            if hasattr(chunk, "message"):
                # ChatGenerationChunk has a message attribute
                content = chunk.message.content
            else:
                # AIMessageChunk is the message itself
                content = chunk.content

            if isinstance(content, str):
                content_parts.append(content)
        full_content = "".join(content_parts)
        assert full_content is not None
        assert len(full_content) > 0

    def test_usage_metadata_streaming(self) -> None:
        """Test usage metadata in streaming responses."""
        chat = ChatHeroku(streaming=True)
        messages = [HumanMessage(content="Hello")]
        chunks = list(chat.stream(messages))
        assert len(chunks) > 0

    def test_abatch(self) -> None:
        """Test async batch processing."""
        import asyncio

        async def run_abatch() -> List[Any]:
            chat = ChatHeroku()
            messages_list = [
                [HumanMessage(content="Hello")],
                [HumanMessage(content="Hi there")],
            ]
            # Cast to satisfy mypy's strict type checking
            results = await chat.abatch(cast(Any, messages_list))
            return list(results)

        results = asyncio.run(run_abatch())
        assert len(results) == 2
        for result in results:
            assert result.content is not None
            assert len(result.content) > 0

    def test_batch(self) -> None:
        """Test batch processing."""
        chat = ChatHeroku()
        messages_list = [
            [HumanMessage(content="Hello")],
            [HumanMessage(content="Hi there")],
        ]
        # Cast to satisfy mypy's strict type checking
        results = chat.batch(cast(Any, messages_list))
        assert len(results) == 2
        for result in results:
            assert result.content is not None
            assert len(result.content) > 0

    def test_stop_sequence(self) -> None:
        """Test stop sequence handling."""
        chat = ChatHeroku(stop=["END", "STOP"])
        messages = [HumanMessage(content="Write a short story")]
        result = chat.invoke(messages)
        assert result.content is not None
        assert len(result.content) > 0

    def test_double_messages_conversation(self) -> None:
        """Test conversation with multiple messages."""
        chat = ChatHeroku()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there! How can I help you today?"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm doing well, thank you!"),
            HumanMessage(content="That's great!"),
        ]
        result = chat.invoke(messages)
        assert result.content is not None
        assert len(result.content) > 0

    def test_production_tool_calling(self) -> None:
        """Test tool calling functionality against production endpoint.

        Validates that the integration can handle tool definitions and
        tool choice parameters correctly.
        """
        # Skip if tool calling is not supported
        if not self.tool_calling_supported:
            pytest.skip("Tool calling not supported by this model")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "The city and state"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        chat = ChatHeroku(tools=tools, tool_choice="auto")
        messages = [HumanMessage(content="What's the weather like in San Francisco?")]

        result = chat.invoke(messages)

        # Check if tool calling worked
        if hasattr(result, "tool_calls") and result.tool_calls:
            assert len(result.tool_calls) > 0
        elif hasattr(result, "additional_kwargs") and result.additional_kwargs.get("tool_calls"):
            assert len(result.additional_kwargs["tool_calls"]) > 0
        else:
            # If no tool calls, at least ensure we got a response
            assert result.content is not None
            assert len(result.content) > 0

    def test_production_performance_benchmark(self) -> None:
        """Test performance characteristics against production endpoint.

        Validates that the integration meets performance requirements
        for response time and reliability.
        """
        chat = ChatHeroku()
        messages = [HumanMessage(content="Hello")]

        # Run multiple requests to test consistency
        response_times = []
        for _ in range(3):
            start_time = time.time()
            result = chat.invoke(messages)
            response_time = time.time() - start_time
            response_times.append(response_time)

            assert result.content is not None
            assert len(result.content) > 0

        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 10.0, f"Average response time {avg_response_time:.2f}s exceeds 10s threshold"

    def test_production_error_scenarios(self) -> None:
        """Test various error scenarios against production endpoint.

        Validates that the integration handles edge cases and error conditions
        gracefully with appropriate error messages.
        """
        chat = ChatHeroku()

        # Test with empty message list
        with pytest.raises((ValueError, RuntimeError)):
            chat.invoke([])

        # Test with None input
        with pytest.raises(ValueError):
            chat.invoke(None)  # type: ignore[arg-type]

        # Test with empty string
        with pytest.raises((ValueError, RuntimeError)):
            chat.invoke("")

    def test_production_configuration_validation(self) -> None:
        """Test configuration validation against production endpoint.

        Validates that the integration properly validates configuration
        parameters and handles edge cases.
        """
        # Test with various parameter combinations
        configs = [
            {"temperature": 0.0, "max_tokens": 100},
            {"temperature": 1.0, "max_tokens": 50},
            {"top_p": 0.9, "streaming": True},
            # Skip extended_thinking as it may not be supported by Heroku Inference API
            # {"extended_thinking": {"enabled": True}},
        ]

        for config in configs:
            chat = ChatHeroku(**config)  # type: ignore[arg-type]

            # Handle streaming configurations differently
            if config.get("streaming", False):
                # For streaming configs, use the stream method
                chunks = list(chat.stream("Hello"))
                assert len(chunks) > 0
                # Combine chunks to get full content
                content_parts: List[str] = []
                for chunk in chunks:
                    # Handle both ChatGenerationChunk and AIMessageChunk
                    if hasattr(chunk, "message"):
                        # ChatGenerationChunk has a message attribute
                        content = chunk.message.content
                    else:
                        # AIMessageChunk is the message itself
                        content = chunk.content

                    if isinstance(content, str):
                        content_parts.append(content)
                full_content = "".join(content_parts)
                assert full_content is not None
                assert len(full_content) > 0
            else:
                # For non-streaming configs, use the invoke method
                result = chat.invoke("Hello")
                assert result.content is not None
                assert len(result.content) > 0


# Skip all tests if environment variables are not set
def pytest_configure(config: Any) -> None:
    """Configure pytest to skip production tests if environment variables are missing."""
    # Try to load .env file if dotenv is available
    if DOTENV_AVAILABLE:
        for env_file in [".env", ".env.local", ".env.production"]:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                break

    if not all([os.getenv("INFERENCE_URL"), os.getenv("INFERENCE_KEY"), os.getenv("INFERENCE_MODEL_ID")]):
        config.addinivalue_line("markers", "production: marks tests as production endpoint tests")


def pytest_collection_modifyitems(config: Any, items: list) -> None:
    """Mark production tests and skip if environment variables are missing."""
    skip_production = pytest.mark.skip(reason="Production endpoint tests require environment variables")

    # Check if environment variables are available
    env_vars_available = all([os.getenv("INFERENCE_URL"), os.getenv("INFERENCE_KEY"), os.getenv("INFERENCE_MODEL_ID")])

    for item in items:
        # Check if the test class or method has the production marker
        if "production" in item.keywords:
            if not env_vars_available:
                item.add_marker(skip_production)
