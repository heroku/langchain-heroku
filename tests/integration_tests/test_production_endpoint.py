"""Integration tests for MiaChat production endpoint."""

import os
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

# Load dotenv if available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

from langchain_heroku.chat_models import MiaChat


class TestProductionEndpoint:
    """Test MiaChat against production endpoint."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Try to load .env file if dotenv is available
        if DOTENV_AVAILABLE:
            # Try common .env file names
            for env_file in [".env", ".env.local", ".env.production"]:
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
            pytest.skip("Production endpoint tests require INFERENCE_URL, INFERENCE_KEY, and INFERENCE_MODEL_ID environment variables")
    
    def test_production_basic_conversation(self):
        """Test basic conversation against production endpoint."""
        chat = MiaChat()
        messages = [HumanMessage(content="Hello! How are you today?")]
        
        result = chat.invoke(messages)
        
        assert result.content is not None
        assert len(result.content) > 0
        assert hasattr(result, 'usage_metadata')
    
    def test_production_system_message(self):
        """Test system message against production endpoint."""
        chat = MiaChat()
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is 2 + 2?")
        ]
        
        result = chat.invoke(messages)
        
        assert result.content is not None
        assert len(result.content) > 0
        # Should mention 4 in the response
        assert "4" in result.content or "four" in result.content.lower()
    
    def test_production_string_input(self):
        """Test string input against production endpoint."""
        chat = MiaChat()
        
        result = chat.invoke("Tell me a short joke.")
        
        assert result.content is not None
        assert len(result.content) > 0
    
    def test_production_temperature_variation(self):
        """Test temperature parameter against production endpoint."""
        # Test with low temperature
        low_temp_chat = MiaChat(temperature=0.0)
        result_low = low_temp_chat.invoke("What is the capital of France?")
        
        # Test with high temperature
        high_temp_chat = MiaChat(temperature=0.9)
        result_high = high_temp_chat.invoke("What is the capital of France?")
        
        assert result_low.content is not None
        assert result_high.content is not None
        # Both should mention Paris
        assert "Paris" in result_low.content or "paris" in result_low.content.lower()
        assert "Paris" in result_high.content or "paris" in result_high.content.lower()
    
    def test_production_max_tokens(self):
        """Test max_tokens parameter against production endpoint."""
        chat = MiaChat(max_tokens=10)
        
        result = chat.invoke("Write a detailed explanation of quantum physics.")
        
        assert result.content is not None
        # Response should be limited by max_tokens
        assert len(result.content.split()) <= 15  # Allow some flexibility
    
    def test_production_streaming(self):
        """Test streaming against production endpoint."""
        chat = MiaChat(streaming=True)
        messages = [HumanMessage(content="Write a short story about a cat.")]
        
        full_response = ""
        for chunk in chat.stream(messages):
            # Support both ChatGenerationChunk and AIMessageChunk
            content = getattr(chunk, "content", None)
            if content is None and hasattr(chunk, "message"):
                content = getattr(chunk.message, "content", "")
            if content:
                full_response += content
        
        assert full_response is not None
        assert len(full_response) > 0
    
    def test_production_error_handling(self):
        """Test error handling against production endpoint."""
        chat = MiaChat()
        
        # Test with invalid input
        with pytest.raises(ValueError):
            chat.invoke(12345)
    
    def test_production_usage_metadata(self):
        """Test that usage metadata is properly returned."""
        chat = MiaChat()
        messages = [HumanMessage(content="Hello")]
        
        result = chat.invoke(messages)
        
        assert hasattr(result, 'usage_metadata')
        metadata = result.usage_metadata
        
        # Check that metadata contains expected fields
        assert metadata is not None
        # Note: Some APIs might not return all fields, so we check what's available
        if 'input_tokens' in metadata:
            assert isinstance(metadata['input_tokens'], int)
        if 'output_tokens' in metadata:
            assert isinstance(metadata['output_tokens'], int)
        if 'total_tokens' in metadata:
            assert isinstance(metadata['total_tokens'], int)
    
    def test_production_response_metadata(self):
        """Test that response metadata is properly returned."""
        chat = MiaChat()
        messages = [HumanMessage(content="Hello")]
        
        result = chat.invoke(messages)
        
        # Check that response_metadata exists
        assert hasattr(result, 'response_metadata')
        # The response_metadata should be a dict
        assert isinstance(result.response_metadata, dict)
    
    @pytest.mark.slow
    def test_production_timeout_handling(self):
        """Test timeout handling against production endpoint."""
        # Test with a very short timeout
        chat = MiaChat(timeout=1)
        messages = [HumanMessage(content="Write a very long story.")]
        
        # This might timeout, which is expected behavior
        try:
            result = chat.invoke(messages)
            assert result.content is not None
        except Exception as e:
            # Timeout is acceptable for this test
            assert "timeout" in str(e).lower() or "timed out" in str(e).lower()
    
    def test_production_dotenv_loading(self):
        """Test that dotenv loading works correctly."""
        # This test verifies that environment variables are properly loaded
        # from .env files when dotenv is available
        if DOTENV_AVAILABLE:
            # Check that we can access the environment variables
            assert os.getenv("INFERENCE_URL") is not None
            assert os.getenv("INFERENCE_KEY") is not None
            assert os.getenv("INFERENCE_MODEL_ID") is not None
        else:
            # If dotenv is not available, we should still have env vars from system
            assert os.getenv("INFERENCE_URL") is not None
            assert os.getenv("INFERENCE_KEY") is not None
            assert os.getenv("INFERENCE_MODEL_ID") is not None


# Skip all tests if environment variables are not set
def pytest_configure(config):
    """Configure pytest to skip production tests if environment variables are missing."""
    # Try to load .env file if dotenv is available
    if DOTENV_AVAILABLE:
        for env_file in [".env", ".env.local", ".env.production"]:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                break
    
    if not all([
        os.getenv("INFERENCE_URL"),
        os.getenv("INFERENCE_KEY"),
        os.getenv("INFERENCE_MODEL_ID")
    ]):
        config.addinivalue_line(
            "markers", "production: marks tests as production endpoint tests"
        )


def pytest_collection_modifyitems(config, items):
    """Mark production tests and skip if environment variables are missing."""
    skip_production = pytest.mark.skip(reason="Production endpoint tests require environment variables")
    
    for item in items:
        if "production" in item.keywords:
            if not all([
                os.getenv("INFERENCE_URL"),
                os.getenv("INFERENCE_KEY"),
                os.getenv("INFERENCE_MODEL_ID")
            ]):
                item.add_marker(skip_production) 