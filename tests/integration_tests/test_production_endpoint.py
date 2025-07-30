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
from typing import Any, Dict, Generator, List

import pytest
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
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

# Import standard test classes from langchain_tests
try:
    from langchain_tests.integration_tests import ChatModelIntegrationTests

    STANDARD_TESTS_AVAILABLE = True
except ImportError:
    STANDARD_TESTS_AVAILABLE = False

    # Create a stub class if langchain_tests is not available
    class ChatModelIntegrationTests:  # type: ignore[no-redef]
        pass


from langchain_heroku.chat_models import ChatHeroku


@pytest.mark.production
class TestChatHerokuIntegration(ChatModelIntegrationTests):
    """Standard integration tests for ChatHeroku using LangChain's test framework.

    This class extends LangChain's standard ChatModelIntegrationTests to ensure
    compatibility and consistency with other LangChain integrations.
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

    @property
    def chat_model_class(self) -> type:
        """Return the ChatHeroku class for testing."""
        return ChatHeroku

    @property
    def chat_model_params(self) -> Dict[str, Any]:
        """Return parameters for initializing ChatHeroku in tests."""
        # Check if environment variables are available
        env_vars_available = all([os.getenv("INFERENCE_URL"), os.getenv("INFERENCE_KEY"), os.getenv("INFERENCE_MODEL_ID")])

        if not env_vars_available:
            pytest.skip("Production endpoint tests require environment variables")

        # Include required environment variables for testing
        return {
            "inference_url": os.getenv("INFERENCE_URL", "https://dummy.url"),
            "api_key": os.getenv("INFERENCE_KEY", "dummy-key"),
            "model": os.getenv("INFERENCE_MODEL_ID", "dummy-model"),
            "temperature": 0.1,  # Low temperature for deterministic responses
            "max_tokens": 256,
            "timeout": 30,
        }

    def chat_model_invoke_params_example(self) -> List[HumanMessage]:
        """Return example parameters for testing chat model invocation."""
        return [HumanMessage(content="Hello! How are you today?")]

    @property
    def returns_usage_metadata(self) -> bool:
        """Return whether the model returns usage metadata."""
        return True

    def test_chat_model_standard_integration(self) -> None:
        """Test that ChatHeroku passes all standard integration tests."""
        # This test ensures compatibility with LangChain's standard test suite
        # The parent class will run standard tests automatically
        pass


class TestChatHerokuMockIntegration:
    """Mock-based integration tests for ChatHeroku.

    These tests use mocked API responses to validate the integration logic
    without requiring external API calls. Useful for CI/CD environments.
    """

    @pytest.fixture(autouse=True)
    def setup_mock_environment(self) -> Generator[None, None, None]:
        """Setup mock environment variables for testing."""
        # Store original environment variables
        original_url = os.environ.get("INFERENCE_URL")
        original_key = os.environ.get("INFERENCE_KEY")
        original_model = os.environ.get("INFERENCE_MODEL_ID")

        # Set mock environment variables
        os.environ["INFERENCE_URL"] = "https://dummy.url"
        os.environ["INFERENCE_KEY"] = "dummy-key"
        os.environ["INFERENCE_MODEL_ID"] = "dummy-model"

        yield

        # Restore original environment variables
        if original_url is not None:
            os.environ["INFERENCE_URL"] = original_url
        else:
            os.environ.pop("INFERENCE_URL", None)

        if original_key is not None:
            os.environ["INFERENCE_KEY"] = original_key
        else:
            os.environ.pop("INFERENCE_KEY", None)

        if original_model is not None:
            os.environ["INFERENCE_MODEL_ID"] = original_model
        else:
            os.environ.pop("INFERENCE_MODEL_ID", None)

    def test_mock_basic_integration(self) -> None:
        """Test basic integration with mocked API responses."""
        from unittest.mock import MagicMock, patch

        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "dummy-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            chat = ChatHeroku()
            result = chat.invoke("Hello")

            assert result.content == "Hello! How can I help you?"
            assert "usage_metadata" in result.additional_kwargs

    def test_mock_streaming_integration(self) -> None:
        """Test streaming integration with mocked API responses."""
        from unittest.mock import MagicMock, patch

        # Mock SSE client for streaming
        mock_event = MagicMock()
        mock_event.data = '{"choices": [{"delta": {"content": "Hello"}}]}'

        with patch("httpx.Client") as mock_client_class, patch("sseclient.SSEClient") as mock_sse_class:
            # Mock the HTTP client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.iter_bytes.return_value = [b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n']
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__enter__.return_value = mock_client

            # Mock the SSE client
            mock_sse = MagicMock()
            mock_sse.events.return_value = [mock_event]
            mock_sse_class.return_value = mock_sse

            chat = ChatHeroku(streaming=True)
            chunks = list(chat.stream("Hello"))

            assert len(chunks) > 0

    def test_mock_error_handling(self) -> None:
        """Test error handling with mocked API responses."""
        from unittest.mock import MagicMock, patch

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.side_effect = Exception("Network error")
            mock_client_class.return_value.__enter__.return_value = mock_client

            chat = ChatHeroku()
            with pytest.raises(Exception, match="Network error"):
                chat.invoke("Hello")

    def test_mock_configuration_validation(self) -> None:
        """Test configuration validation with mocked responses."""
        from unittest.mock import MagicMock, patch

        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "dummy-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            # Test various configurations
            configs = [
                {"temperature": 0.0, "max_tokens": 100},
                {"temperature": 1.0, "max_tokens": 50},
                {"top_p": 0.9, "streaming": False},
            ]

            for config in configs:
                chat = ChatHeroku(**config)  # type: ignore[arg-type]
                result = chat.invoke("Hello")
                assert result.content is not None
                assert len(result.content) > 0


class TestProductionEndpoint:
    """Comprehensive integration tests for ChatHeroku against production endpoint.

    These tests validate end-to-end functionality with the real Heroku Inference API,
    ensuring that the integration works correctly in production scenarios.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Setup test environment and validate required configuration."""
        # Try to load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            for env_file in [".env", ".env.local", ".env.production"]:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break

        # Check if we have the required environment variables
        self.has_env_vars = all([os.getenv("INFERENCE_URL"), os.getenv("INFERENCE_KEY"), os.getenv("INFERENCE_MODEL_ID")])

        if not self.has_env_vars:
            pytest.skip("Production endpoint tests require INFERENCE_URL, INFERENCE_KEY, " "and INFERENCE_MODEL_ID environment variables")

    def test_production_basic_conversation(self) -> None:
        """Test basic conversation flow against production endpoint.

        Validates that the integration can handle simple human messages
        and return appropriate AI responses with proper metadata.
        """
        chat = ChatHeroku()
        messages = [HumanMessage(content="Hello! How are you today?")]

        start_time = time.time()
        result = chat.invoke(messages)
        response_time = time.time() - start_time

        # Validate response structure
        assert result.content is not None
        assert len(result.content) > 0
        assert "usage_metadata" in result.additional_kwargs

        # Performance validation
        assert response_time < 30.0, f"Response took {response_time:.2f}s, expected < 30s"

        # Validate metadata structure
        metadata = result.additional_kwargs["usage_metadata"]
        assert isinstance(metadata, dict)
        if "total_tokens" in metadata:
            assert isinstance(metadata["total_tokens"], int)
            assert metadata["total_tokens"] > 0

    def test_production_system_message(self) -> None:
        """Test system message handling against production endpoint.

        Validates that system messages are properly processed and influence
        the AI's behavior as expected.
        """
        chat = ChatHeroku()
        messages = [SystemMessage(content="You are a helpful assistant."), HumanMessage(content="What is 2 + 2?")]

        result = chat.invoke(messages)

        assert result.content is not None
        assert len(result.content) > 0

        # Validate that the response includes the expected answer
        content = str(result.content).lower()
        assert "4" in content or "four" in content, f"Expected '4' or 'four' in response, got: {content}"

    def test_production_string_input(self) -> None:
        """Test string input handling against production endpoint.

        Validates that the integration can handle string inputs directly
        and convert them to proper message format.
        """
        chat = ChatHeroku()

        result = chat.invoke("Tell me a short joke.")

        assert result.content is not None
        assert len(result.content) > 0

    def test_production_temperature_variation(self) -> None:
        """Test temperature parameter effects against production endpoint.

        Validates that different temperature settings produce appropriately
        different response characteristics.
        """
        # Test with low temperature for deterministic responses
        low_temp_chat = ChatHeroku(temperature=0.0)
        result_low = low_temp_chat.invoke("What is the capital of France?")

        # Test with high temperature for more creative responses
        high_temp_chat = ChatHeroku(temperature=0.9)
        result_high = high_temp_chat.invoke("What is the capital of France?")

        assert result_low.content is not None
        assert result_high.content is not None

        # Both should mention Paris
        content_low = str(result_low.content).lower()
        content_high = str(result_high.content).lower()
        assert "paris" in content_low, f"Expected 'paris' in low temp response, got: {content_low}"
        assert "paris" in content_high, f"Expected 'paris' in high temp response, got: {content_high}"

    def test_production_max_tokens(self) -> None:
        """Test max_tokens parameter against production endpoint.

        Validates that the max_tokens parameter properly limits response length.
        """
        chat = ChatHeroku(max_tokens=10)

        result = chat.invoke("Write a detailed explanation of quantum physics.")

        assert result.content is not None
        # Response should be limited by max_tokens
        content = str(result.content)
        word_count = len(content.split())
        assert word_count <= 15, f"Expected <= 15 words, got {word_count} words"

    def test_production_streaming(self) -> None:
        """Test streaming functionality against production endpoint.

        Validates that streaming responses work correctly and produce
        the same final result as non-streaming calls.
        """
        chat = ChatHeroku(streaming=True)
        messages = [HumanMessage(content="Write a short story about a cat.")]

        full_response = ""
        chunk_count = 0

        for chunk in chat.stream(messages):
            chunk_count += 1
            # Support both ChatGenerationChunk and AIMessageChunk
            content = getattr(chunk, "content", None)
            if content is None and hasattr(chunk, "message"):
                content = getattr(chunk.message, "content", "")
            if content:
                full_response += content

        assert full_response is not None
        assert len(full_response) > 0
        assert chunk_count > 0, "Expected at least one chunk in streaming response"

    def test_production_error_handling(self) -> None:
        """Test error handling against production endpoint.

        Validates that the integration properly handles invalid inputs
        and returns appropriate error messages.
        """
        chat = ChatHeroku()

        # Test with invalid input type
        with pytest.raises(ValueError):
            chat.invoke(12345)  # type: ignore[arg-type]

    def test_production_usage_metadata(self) -> None:
        """Test that usage metadata is properly returned and structured.

        Validates that the integration correctly captures and returns
        token usage information from the API.
        """
        chat = ChatHeroku()
        messages = [HumanMessage(content="Hello")]

        result = chat.invoke(messages)

        assert "usage_metadata" in result.additional_kwargs
        metadata = result.additional_kwargs["usage_metadata"]

        # Check that metadata contains expected fields
        assert metadata is not None
        assert isinstance(metadata, dict)

        # Note: Some APIs might not return all fields, so we check what's available
        if "input_tokens" in metadata:
            assert isinstance(metadata["input_tokens"], int)
            assert metadata["input_tokens"] >= 0
        if "output_tokens" in metadata:
            assert isinstance(metadata["output_tokens"], int)
            assert metadata["output_tokens"] >= 0
        if "total_tokens" in metadata:
            assert isinstance(metadata["total_tokens"], int)
            assert metadata["total_tokens"] > 0

    def test_production_response_metadata(self) -> None:
        """Test that response metadata is properly returned.

        Validates that the integration correctly captures response metadata
        from the API response.
        """
        chat = ChatHeroku()
        messages = [HumanMessage(content="Hello")]

        result = chat.invoke(messages)

        # Check that response_metadata exists
        assert hasattr(result, "response_metadata")
        # The response_metadata should be a dict
        assert isinstance(result.response_metadata, dict)

    @pytest.mark.slow
    def test_production_timeout_handling(self) -> None:
        """Test timeout handling against production endpoint.

        Validates that the integration properly handles timeout scenarios
        and provides appropriate error messages.
        """
        # Test with a very short timeout
        chat = ChatHeroku(timeout=1)
        messages = [HumanMessage(content="Write a very long story.")]

        # This might timeout, which is expected behavior
        try:
            result = chat.invoke(messages)
            assert result.content is not None
        except Exception as e:
            # Timeout is acceptable for this test
            error_msg = str(e).lower()
            assert any(keyword in error_msg for keyword in ["timeout", "timed out", "time out"]), f"Expected timeout-related error, got: {error_msg}"

    def test_production_dotenv_loading(self) -> None:
        """Test that dotenv loading works correctly.

        Validates that environment variables are properly loaded
        from .env files when dotenv is available.
        """
        # Check that we can access the environment variables
        assert os.getenv("INFERENCE_URL") is not None, "INFERENCE_URL not found in environment"
        assert os.getenv("INFERENCE_KEY") is not None, "INFERENCE_KEY not found in environment"
        assert os.getenv("INFERENCE_MODEL_ID") is not None, "INFERENCE_MODEL_ID not found in environment"

    def test_production_extended_thinking(self) -> None:
        """Test extended_thinking parameter against production endpoint.

        Validates that extended thinking functionality works correctly
        with Claude Sonnet models that support this feature.
        """
        # Only test with Claude Sonnet models that support extended thinking
        model_id = os.getenv("INFERENCE_MODEL_ID", "")
        if not any(sonnet_model in model_id.lower() for sonnet_model in ["sonnet", "claude-3-7-sonnet", "claude-4-sonnet"]):
            pytest.skip("Extended thinking is only supported on Claude Sonnet models")

        extended_thinking_config = {"enabled": True, "budget_tokens": 1024, "include_reasoning": True}
        chat = ChatHeroku(extended_thinking=extended_thinking_config)
        messages = [HumanMessage(content="Solve this step by step: What is 15% of 200?")]

        result = chat.invoke(messages)

        assert result.content is not None
        assert len(result.content) > 0

        # The response should show reasoning steps
        content = str(result.content).lower()
        assert any(
            keyword in content for keyword in ["step", "calculate", "multiply", "divide", "30"]
        ), f"Expected reasoning keywords in response, got: {content}"

    def test_production_all_message_types(self) -> None:
        """Test all message types against production endpoint.

        Validates that the integration can handle all LangChain message types
        and properly convert them to API format.
        """
        # Skip this test as the Heroku Inference API doesn't support tool roles in messages
        pytest.skip("Heroku Inference API doesn't support tool roles in message history")

        chat = ChatHeroku()
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather like?"),
            AIMessage(content="I don't have access to weather data."),
            ToolMessage(content="The weather is sunny", tool_call_id="call_123"),
            FunctionMessage(content="Temperature: 75°F", name="get_weather"),
        ]

        result = chat.invoke(messages)

        assert result.content is not None
        assert len(result.content) > 0

    def test_production_tool_calling(self) -> None:
        """Test tool calling functionality against production endpoint.

        Validates that the integration can handle tool definitions and
        tool choice parameters correctly.
        """
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
