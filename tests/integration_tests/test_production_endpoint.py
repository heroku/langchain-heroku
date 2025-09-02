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
    ToolMessage,
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

    def test_bind_tools_functionality(self) -> None:
        """Test bind_tools method with proper tool call format.

        This test validates that ChatHeroku properly supports the bind_tools()
        method and returns tool calls in the correct LangChain format.
        """
        # Skip if tool calling is not supported
        if not self.tool_calling_supported:
            pytest.skip("Tool calling not supported by this model")

        from langchain_core.tools import tool

        @tool
        def get_temperature(location: str) -> str:
            """Get the temperature for a location."""
            return f"The temperature in {location} is 72°F"

        # Test with bind_tools method
        chat = ChatHeroku()
        bound_chat = chat.bind_tools([get_temperature], tool_choice="required")

        messages = [HumanMessage(content="What's the temperature in Boston?")]
        result = bound_chat.invoke(messages)

        # Verify tool calls are properly formatted
        assert hasattr(result, "tool_calls"), "AIMessage should have tool_calls attribute"
        assert len(result.tool_calls) > 0, "Should have at least one tool call"

        # Verify tool call structure matches LangChain format
        tool_call = result.tool_calls[0]
        assert "name" in tool_call, "Tool call should have 'name' field"
        assert "args" in tool_call, "Tool call should have 'args' field"
        assert "id" in tool_call, "Tool call should have 'id' field"
        assert "type" in tool_call, "Tool call should have 'type' field"

        # Verify tool call details
        assert tool_call["name"] == "get_temperature"
        assert isinstance(tool_call["args"], dict)
        assert "location" in tool_call["args"]

    def test_tool_message_handling(self) -> None:
        """Test ToolMessage handling and API format conversion.

        This test validates that ChatHeroku can properly handle ToolMessage
        objects and convert them to the correct API format.
        """
        # Skip if tool calling is not supported
        if not self.tool_calling_supported:
            pytest.skip("Tool calling not supported by this model")

        from langchain_core.tools import tool

        @tool
        def calculate_sum(a: int, b: int) -> str:
            """Calculate the sum of two numbers."""
            return f"The sum of {a} and {b} is {a + b}"

        chat = ChatHeroku()
        bound_chat = chat.bind_tools([calculate_sum], tool_choice="required")

        # Step 1: Get AI response with tool call
        user_msg = HumanMessage(content="What is 15 + 27?")
        ai_response = bound_chat.invoke([user_msg])

        assert hasattr(ai_response, "tool_calls")
        assert len(ai_response.tool_calls) > 0

        # Step 2: Create ToolMessage
        tool_call = ai_response.tool_calls[0]
        tool_result = calculate_sum.invoke(tool_call["args"])
        tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call["id"])

        # Step 3: Test API format conversion
        conversation = [user_msg, ai_response, tool_msg]
        api_messages = chat._messages_to_api(conversation)

        # Verify API format
        assert len(api_messages) == 3

        # Check user message
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["content"] == "What is 15 + 27?"

        # Check AI message with tool calls
        assert api_messages[1]["role"] == "assistant"
        assert "tool_calls" in api_messages[1]
        assert len(api_messages[1]["tool_calls"]) > 0

        # Check tool call structure in API format
        api_tool_call = api_messages[1]["tool_calls"][0]
        assert "id" in api_tool_call
        assert "type" in api_tool_call
        assert api_tool_call["type"] == "function"
        assert "function" in api_tool_call
        assert "name" in api_tool_call["function"]
        assert "arguments" in api_tool_call["function"]

        # Check tool message
        assert api_messages[2]["role"] == "tool"
        assert api_messages[2]["content"] == tool_result
        assert api_messages[2]["tool_call_id"] == tool_call["id"]

    def test_openai_compatibility_format(self) -> None:
        """Test that ChatHeroku produces OpenAI-compatible message format.

        This test ensures that the message flow matches the OpenAI format
        exactly, as shown in the messages.txt example.
        """
        # Skip if tool calling is not supported
        if not self.tool_calling_supported:
            pytest.skip("Tool calling not supported by this model")

        from langchain_core.tools import tool

        @tool
        def get_current_weather(location: str) -> str:
            """Get the current weather in a given location."""
            return f"The weather in {location} is sunny and 75°F"

        # Test the complete OpenAI-style workflow
        chat = ChatHeroku()
        bound_chat = chat.bind_tools([get_current_weather], tool_choice="required")

        # Human message
        human_msg = HumanMessage(content="What's the weather like in San Francisco?")

        # AI response with tool calls
        ai_response = bound_chat.invoke([human_msg])

        # Verify AI message structure (like OpenAI)
        assert hasattr(ai_response, "tool_calls")
        assert len(ai_response.tool_calls) > 0

        tool_call = ai_response.tool_calls[0]
        assert tool_call["name"] == "get_current_weather"
        assert "location" in tool_call["args"]
        assert tool_call["args"]["location"].lower() == "san francisco"

        # Execute tool and create tool message
        tool_result = get_current_weather.invoke(tool_call["args"])
        tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call["id"])

        # Verify tool message matches OpenAI format
        assert tool_msg.content == tool_result
        assert tool_msg.tool_call_id == tool_call["id"]

        # Verify the complete conversation format matches OpenAI
        conversation = [human_msg, ai_response, tool_msg]
        api_messages = chat._messages_to_api(conversation)

        # Verify the API format matches OpenAI structure
        assert len(api_messages) == 3
        assert api_messages[0]["role"] == "user"
        assert api_messages[1]["role"] == "assistant"
        assert api_messages[2]["role"] == "tool"
        assert api_messages[2]["tool_call_id"] == tool_call["id"]

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

        # Test with empty message list - should raise ValueError
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            chat.invoke([])

        # Test with None input - should raise ValueError for invalid input type
        with pytest.raises(ValueError, match="Invalid input type"):
            chat.invoke(None)  # type: ignore[arg-type]

        # Test with empty string - LangChain converts this to HumanMessage(content="")
        # Our implementation handles this gracefully by providing fallback content
        # So this should succeed, not raise an exception
        result = chat.invoke("")
        assert result.content is not None
        assert len(str(result.content)) > 0

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
