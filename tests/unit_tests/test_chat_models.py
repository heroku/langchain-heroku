"""Test chat model integration."""

from typing import Type
import pytest
from unittest.mock import patch, MagicMock
from langchain_heroku.chat_models import MiaChat

class TestChatHerokuUnit():
    @property
    def chat_model_class(self) -> Type[MiaChat]:
        return MiaChat

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }


def test_chat_heroku_basic_usage():
    # Mock response from Heroku Inference API
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0)
            messages = [
                type("Msg", (), {"role": "user", "content": "Say hello"})(),
            ]
            result = llm._generate(messages)
            assert result.generations[0].message.content == "Hello! How can I help you?"
            assert result.generations[0].message.usage_metadata["total_tokens"] == 12


def test_chat_heroku_invoke_with_string():
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0)
            result = llm.invoke("Say hello")
            assert result.content == "Hello! How can I help you?"
            assert result.usage_metadata["total_tokens"] == 12


def test_chat_heroku_invoke_with_messages():
    from langchain_core.messages import HumanMessage
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0)
            messages = [HumanMessage(content="Say hello")]
            result = llm.invoke(messages)
            assert result.content == "Hello! How can I help you?"
            assert result.usage_metadata["total_tokens"] == 12


def test_chat_heroku_invoke_invalid_input():
    llm = MiaChat(model="bird-brain-001", temperature=0)
    with pytest.raises(ValueError):
        llm.invoke(12345)  # Not a string or list of BaseMessage


def test_chat_heroku_tool_choice_string():
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Tool test"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0, tool_choice="auto")
            llm.invoke("Test tool choice")
            # Check that tool_choice was included in the payload
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["tool_choice"] == "auto"


def test_chat_heroku_tool_choice_dict():
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Tool test"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    tool_choice_dict = {"type": "function", "function": {"name": "get_weather"}}
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0, tool_choice=tool_choice_dict)
            llm.invoke("Test tool choice dict")
            # Check that tool_choice dict was included in the payload
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["tool_choice"] == tool_choice_dict


def test_chat_heroku_top_p():
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Top-p test"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0, top_p=0.95)
            llm.invoke("Test top_p")
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["top_p"] == 0.95


def test_chat_heroku_message_types():
    """Test that all message types are properly mapped to API roles."""
    from langchain_core.messages import (
        HumanMessage, SystemMessage, AIMessage, ToolMessage, FunctionMessage
    )
    
    llm = MiaChat(model="bird-brain-001", temperature=0)
    
    # Test all message types
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What's the weather like?"),
        AIMessage(content="I don't have access to weather data."),
        ToolMessage(content="The weather is sunny", tool_call_id="call_123"),
        FunctionMessage(content="Temperature: 75°F", name="get_weather")
    ]
    
    api_messages = llm._messages_to_api(messages)
    
    # Verify role mapping
    expected_roles = ["system", "user", "assistant", "tool", "tool"]
    actual_roles = [msg["role"] for msg in api_messages]
    assert actual_roles == expected_roles
    
    # Verify content is preserved
    expected_contents = [
        "You are a helpful assistant.",
        "What's the weather like?",
        "I don't have access to weather data.",
        "The weather is sunny",
        "Temperature: 75°F"
    ]
    actual_contents = [msg["content"] for msg in api_messages]
    assert actual_contents == expected_contents


def test_chat_heroku_streaming_parameter():
    """Test that the streaming parameter is properly handled."""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Streaming test"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0, streaming=True)
            llm.invoke("Test streaming")
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["stream"] == True


def test_chat_heroku_tools_parameter():
    """Test that the tools parameter is properly handled."""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Tools test"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}]
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0, tools=tools)
            llm.invoke("Test tools")
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["tools"] == tools


def test_chat_heroku_allow_ignored_params():
    """Test that allow_ignored_params is included in the payload."""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Allow ignored params test"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0)
            llm.invoke("Test allow_ignored_params")
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["allow_ignored_params"] == True


def test_chat_heroku_extended_thinking():
    """Test that the extended_thinking parameter is properly handled."""
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "bird-brain-001",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Extended thinking test"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    extended_thinking_config = {"enabled": True, "budget_tokens": 1024, "include_reasoning": True}
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value.json.return_value = mock_response
            mock_client.post.return_value.raise_for_status.return_value = None
            mock_client_class.return_value.__enter__.return_value = mock_client

            llm = MiaChat(model="bird-brain-001", temperature=0, extended_thinking=extended_thinking_config)
            llm.invoke("Test extended thinking")
            args, kwargs = mock_client.post.call_args
            assert kwargs["json"]["extended_thinking"] == extended_thinking_config


def test_chat_heroku_extended_thinking_streaming():
    """Test that the extended_thinking parameter is properly handled in streaming mode."""
    extended_thinking_config = {"enabled": True, "budget_tokens": 1024, "include_reasoning": True}
    with patch.dict("os.environ", {"INFERENCE_URL": "https://dummy.url", "INFERENCE_KEY": "dummy-key"}):
        with patch("httpx.stream") as mock_stream:
            # Mock the streaming response
            mock_response = MagicMock()
            mock_response.raise_for_status.return_value = None
            mock_response.iter_lines.return_value = [
                b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
                b'data: {"choices": [{"delta": {"content": " world"}}]}',
                b'data: [DONE]'
            ]
            mock_stream.return_value.__enter__.return_value = mock_response

            llm = MiaChat(model="bird-brain-001", temperature=0, extended_thinking=extended_thinking_config, streaming=True)
            messages = [type("Msg", (), {"role": "user", "content": "Say hello"})()]
            
            # Collect all chunks
            chunks = list(llm._stream(messages))
            
            # Check that extended_thinking was included in the payload
            args, kwargs = mock_stream.call_args
            assert kwargs["json"]["extended_thinking"] == extended_thinking_config


# Optionally, add more tests for error handling, streaming, etc.

if __name__ == "__main__":
    import os
    from langchain_heroku.chat_models import MiaChat
    from langchain_core.messages import HumanMessage

    # Example: Set environment variables here for manual testing
    os.environ["INFERENCE_URL"] = os.getenv("INFERENCE_URL", "https://us.inference.heroku.com")
    os.environ["INFERENCE_KEY"] = os.getenv("INFERENCE_KEY", "inf-asdfghj-1234-567-8910-aabbccddee")
    os.environ["INFERENCE_MODEL_ID"] = os.getenv("INFERENCE_MODEL_ID", "claude-3-7-sonnet")

    chat = MiaChat()
    messages = [HumanMessage(content="Hello, world!")]
    try:
        result = chat.invoke(messages)
        print("Type of result:", type(result))
        print("Result repr:", repr(result))
        # Try to print content if possible
        if hasattr(result, 'generations'):
            print("ChatHeroku response:", result.generations[0].message.content)
        elif hasattr(result, 'content'):
            print("ChatHeroku response:", result.content)
        else:
            print("Unknown result structure.")
    except Exception as e:
        print("Error during ChatHeroku call:", e) 