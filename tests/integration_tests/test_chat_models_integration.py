"""Integration tests for ChatHeroku following LangChain testing standards.

This module contains integration tests that validate the ChatHeroku integration
with the Heroku Inference API, following LangChain's contributor testing guidelines.

Key aspects:
- Extends LangChain's standard ChatModelIntegrationTests
- Tests real API integration scenarios  
- Validates end-to-end functionality
- Includes comprehensive error scenario testing
- Uses environment variable configuration with graceful skipping
"""

import os
from typing import Any, Dict, List

import pytest
from langchain_core.messages import HumanMessage

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


@pytest.mark.integration
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
            for env_file in [".env", ".env.local", ".env.integration"]:
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
        env_vars_available = all([
            os.getenv("INFERENCE_URL"),
            os.getenv("INFERENCE_KEY"),
            os.getenv("INFERENCE_MODEL_ID")
        ])

        if not env_vars_available:
            pytest.skip("Integration tests require INFERENCE_URL, INFERENCE_KEY, and INFERENCE_MODEL_ID environment variables")

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


@pytest.mark.integration
class TestChatHerokuIntegrationScenarios:
    """Additional integration test scenarios specific to ChatHeroku.
    
    These tests complement the standard LangChain integration tests with
    Heroku-specific functionality and edge cases.
    """

    @pytest.fixture(autouse=True)
    def setup_environment(self) -> None:
        """Setup environment variables for testing."""
        # Load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            for env_file in [".env", ".env.local", ".env.integration"]:
                if os.path.exists(env_file):
                    load_dotenv(env_file)
                    break

        # Check if we have the required environment variables
        self.has_env_vars = all([
            os.getenv("INFERENCE_URL"),
            os.getenv("INFERENCE_KEY"),
            os.getenv("INFERENCE_MODEL_ID")
        ])

        if not self.has_env_vars:
            pytest.skip("Integration tests require INFERENCE_URL, INFERENCE_KEY, and INFERENCE_MODEL_ID environment variables")

    def test_basic_conversation_flow(self) -> None:
        """Test basic conversation flow with real API integration."""
        chat = ChatHeroku()
        messages = [HumanMessage(content="What is 2 + 2?")]

        result = chat.invoke(messages)

        # Validate response structure
        assert result.content is not None
        assert len(result.content) > 0
        
        # Check for expected mathematical answer
        content = str(result.content).lower()
        assert "4" in content or "four" in content

        # Validate metadata
        assert "usage_metadata" in result.additional_kwargs
        metadata = result.additional_kwargs["usage_metadata"]
        assert isinstance(metadata, dict)

    def test_streaming_integration(self) -> None:
        """Test streaming functionality in integration environment."""
        chat = ChatHeroku(streaming=True)
        messages = [HumanMessage(content="Count from 1 to 3.")]

        chunks = list(chat.stream(messages))
        
        # Should receive multiple chunks
        assert len(chunks) > 0
        
        # Combine chunks to verify complete response
        full_response = ""
        for chunk in chunks:
            if hasattr(chunk, "content") and chunk.content:
                full_response += str(chunk.content)
        
        assert len(full_response) > 0

    def test_tool_calling_integration(self) -> None:
        """Test tool calling functionality in integration environment."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string", 
                                "description": "The city and state"
                            }
                        },
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

    def test_extended_thinking_integration(self) -> None:
        """Test extended thinking functionality with Claude Sonnet models."""
        # Only test with Claude Sonnet models that support extended thinking
        model_id = os.getenv("INFERENCE_MODEL_ID", "")
        if not any(sonnet_model in model_id.lower() for sonnet_model in ["sonnet", "claude-3-7-sonnet", "claude-4-sonnet"]):
            pytest.skip("Extended thinking is only supported on Claude Sonnet models")

        extended_thinking_config = {
            "enabled": True,
            "budget_tokens": 1024,
            "include_reasoning": True
        }
        
        chat = ChatHeroku(extended_thinking=extended_thinking_config)
        messages = [HumanMessage(content="Solve this step by step: What is 15% of 200?")]

        result = chat.invoke(messages)

        assert result.content is not None
        assert len(result.content) > 0

        # The response should show reasoning steps
        content = str(result.content).lower()
        assert any(
            keyword in content 
            for keyword in ["step", "calculate", "multiply", "divide", "30"]
        )

    def test_parameter_variations(self) -> None:
        """Test various parameter combinations in integration environment."""
        test_configs = [
            {"temperature": 0.0, "max_tokens": 50},
            {"temperature": 0.7, "max_tokens": 100},
            {"top_p": 0.9},
        ]

        for config in test_configs:
            chat = ChatHeroku(**config)  # type: ignore[arg-type]
            result = chat.invoke("Hello")
            
            assert result.content is not None
            assert len(result.content) > 0

    def test_error_handling_integration(self) -> None:
        """Test error handling scenarios in integration environment."""
        chat = ChatHeroku()

        # Test with invalid input types
        with pytest.raises(ValueError):
            chat.invoke(12345)  # type: ignore[arg-type]

        # Test with empty input
        with pytest.raises((ValueError, RuntimeError)):
            chat.invoke("")

        # Test with empty message list
        with pytest.raises((ValueError, RuntimeError)):
            chat.invoke([])

    def test_concurrent_requests(self) -> None:
        """Test concurrent API requests."""
        import concurrent.futures

        chat = ChatHeroku()
        message = "What is the capital of France?"
        
        def make_request() -> Any:
            return chat.invoke(message)

        # Make 3 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        assert len(results) == 3
        for result in results:
            assert result.content is not None
            assert len(result.content) > 0
            assert "paris" in str(result.content).lower()

    def test_large_input_handling(self) -> None:
        """Test handling of large input texts."""
        # Create a reasonably large input (not too large to avoid timeout)
        large_text = "Summarize this text: " + "This is a test sentence. " * 100
        
        chat = ChatHeroku(max_tokens=100)
        
        try:
            result = chat.invoke(large_text)
            assert result.content is not None
            assert len(result.content) > 0
        except Exception as e:
            # Some models may have input length limits
            error_msg = str(e).lower()
            assert any(
                keyword in error_msg 
                for keyword in ["token", "length", "limit", "size"]
            ), f"Expected length-related error, got: {error_msg}"

    def test_response_metadata_integration(self) -> None:
        """Test that response metadata is properly captured."""
        chat = ChatHeroku()
        result = chat.invoke("Hello")

        # Check response metadata structure
        assert hasattr(result, "response_metadata")
        assert isinstance(result.response_metadata, dict)

        # Check usage metadata structure
        assert "usage_metadata" in result.additional_kwargs
        usage_metadata = result.additional_kwargs["usage_metadata"]
        assert isinstance(usage_metadata, dict)

        # Validate common usage fields (if present)
        if "total_tokens" in usage_metadata:
            assert isinstance(usage_metadata["total_tokens"], int)
            assert usage_metadata["total_tokens"] > 0