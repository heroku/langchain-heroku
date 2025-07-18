"""Heroku chat models."""

import json
import os
from typing import Any, Dict, Generator, List, Optional

import httpx
import sseclient
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class MiaChat(BaseChatModel):
    """
    Heroku chat model integration using the Inference API v1 /v1/chat/completions endpoint.

    Example setup (environment variables):
        export INFERENCE_URL="https://your-inference-api-url"
        export INFERENCE_KEY="your-heroku-inference-api-key"
        export INFERENCE_MODEL_ID="your-model-id"

    Basic usage:
        from langchain_heroku.chat_models import MiaChat
        from langchain_core.messages import HumanMessage

        chat = MiaChat()
        result = chat([HumanMessage(content="Hello!")])
        print(result.generations[0].message.content)

    Usage with all message types:
        from langchain_core.messages import (
            HumanMessage, SystemMessage, AIMessage, ToolMessage, FunctionMessage
        )

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What's the weather like?"),
            AIMessage(content="I don't have access to weather data."),
            ToolMessage(content="The weather is sunny", tool_call_id="call_123"),
            FunctionMessage(content="Temperature: 75°F", name="get_weather")
        ]

        chat = MiaChat()
        result = chat(messages)

    Usage with explicit parameters:
        chat = MiaChat(
            model="your-model-id",
            api_key="your-heroku-inference-api-key",
            inference_url="https://your-inference-api-url",
            temperature=0.7,
            max_tokens=256,
            top_p=0.95,
            stop=["\n"],
            tools=[{"type": "function", ...}],
            tool_choice="auto",  # or "required", or a dict for a specific tool
            streaming=False,
            extended_thinking={"enabled": True},  # For Claude Sonnet 3.7 & 4
        )
        result = chat([HumanMessage(content="Hello!")])
        print(result.generations[0].message.content)

    Streaming usage:
        chat = MiaChat(streaming=True)
        for chunk in chat.stream([HumanMessage(content="Hello!")]):
            print(chunk.message.content, end="")

    """

    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    api_key: Optional[str] = None
    inference_url: Optional[str] = None
    tools: Optional[List[dict]] = None  # For tool calling
    tool_choice: Optional[Any] = None  # Controls tool selection per Heroku API
    streaming: bool = False
    top_p: Optional[float] = None  # Nucleus sampling parameter
    extended_thinking: Optional[Dict[str, Any]] = None  # Extended thinking for Claude Sonnet 3.7 & 4

    @property
    def _llm_type(self) -> str:
        return "mia"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model}

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, default)

    def _get_api_key(self) -> Optional[str]:
        return self.api_key or self._get_env("INFERENCE_KEY") or self._get_env("HEROKU_API_KEY")

    def _get_inference_url(self) -> Optional[str]:
        return self.inference_url or self._get_env("INFERENCE_URL")

    def _get_model(self) -> Optional[str]:
        return self.model or self._get_env("INFERENCE_MODEL_ID")

    def _messages_to_api(self, messages: List[BaseMessage]) -> List[dict]:
        # Map LangChain message types to API roles with explicit type checking
        api_msgs = []
        for m in messages:
            # Handle specific message types explicitly
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, ToolMessage):
                role = "tool"
            elif isinstance(m, FunctionMessage):
                role = "tool"  # FunctionMessage maps to tool role for backward compatibility
            else:
                # Fallback to role attribute or type for custom message types
                role = getattr(m, "role", None) or getattr(m, "type", "user") or "user"

            content = str(getattr(m, "content", ""))
            api_msgs.append({"role": role, "content": content})
        return api_msgs

    def _api_to_ai_message(self, resp: dict) -> AIMessage:
        choice = resp["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        additional_kwargs = {"tool_calls": tool_calls} if tool_calls else {}
        usage = resp.get("usage", {})
        usage_metadata = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        # Store usage metadata in additional_kwargs since AIMessage doesn't have usage_metadata parameter
        if usage_metadata:
            additional_kwargs["usage_metadata"] = usage_metadata
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            response_metadata=resp,
        )

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        if not self._get_inference_url():
            raise ValueError("INFERENCE_URL must be set via env or init param.")
        if not self._get_api_key():
            raise ValueError("INFERENCE_KEY or HEROKU_API_KEY must be set via env or init param.")
        if not self._get_model():
            raise ValueError("model or INFERENCE_MODEL_ID must be set via env or init param.")

    def _build_payload(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build the API payload for the chat completion request."""
        payload: Dict[str, Any] = {
            "model": self._get_model(),
            "messages": self._messages_to_api(messages),
            "allow_ignored_params": True,
        }

        # Add optional parameters if they are set
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop is not None:
            payload["stop"] = stop
        elif self.stop is not None:
            payload["stop"] = self.stop
        if self.tools:
            payload["tools"] = self.tools
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        if self.streaming:
            payload["stream"] = True
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.extended_thinking is not None:
            payload["extended_thinking"] = self.extended_thinking

        return payload

    def _make_api_request(self, payload: dict) -> dict:
        """Make the API request with retry logic."""
        url = self._get_inference_url()
        api_key = self._get_api_key()
        timeout = self.timeout or 30
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        max_retries = 2
        for _ in range(max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(f"{url}/v1/chat/completions", json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except Exception as e:
                last_exc = e
        else:
            raise RuntimeError(f"Heroku Inference API call failed after {max_retries} retries: {last_exc}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._validate_config()
        payload = self._build_payload(messages, stop)
        data = self._make_api_request(payload)
        ai_msg = self._api_to_ai_message(data)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _build_streaming_payload(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build the API payload for streaming chat completion requests."""
        payload: Dict[str, Any] = {
            "model": self._get_model(),
            "messages": self._messages_to_api(messages),
            "stream": True,
        }

        # Add optional parameters if they are set
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop is not None:
            payload["stop"] = stop
        elif self.stop is not None:
            payload["stop"] = self.stop
        if self.tools:
            payload["tools"] = self.tools
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.extended_thinking is not None:
            payload["extended_thinking"] = self.extended_thinking

        return payload

    def _make_streaming_request(self, payload: Dict[str, Any]) -> sseclient.SSEClient:
        """Make the streaming API request with retry logic."""
        url = self._get_inference_url()
        api_key = self._get_api_key()
        timeout = self.timeout or 30
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        max_retries = 2
        for _ in range(max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(f"{url}/v1/chat/completions", json=payload, headers=headers)
                    response.raise_for_status()

                    # Convert httpx response to work with sseclient
                    def response_generator() -> Generator[bytes, None, None]:
                        for chunk in response.iter_bytes():
                            yield chunk

                    return sseclient.SSEClient(response_generator())
            except Exception as e:
                last_exc = e
        else:
            raise RuntimeError(f"Heroku Inference API stream call failed after {max_retries} retries: {last_exc}")

    def _parse_sse_event(self, event: sseclient.Event) -> Optional[str]:
        """Parse a single SSE event and extract content."""
        try:
            # Handle the special "[DONE]" message
            if event.data == "[DONE]":
                return None

            data = json.loads(event.data)
            # For streaming, use 'delta' instead of 'message'
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            return delta.get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            # Skip malformed JSON lines or missing data
            return None

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Generator[ChatGenerationChunk, None, None]:
        self._validate_config()
        payload = self._build_streaming_payload(messages, stop)

        client = self._make_streaming_request(payload)
        try:
            for event in client.events():
                content = self._parse_sse_event(event)
                if content is not None:
                    ai_msg_chunk = AIMessageChunk(content=content)
                    chunk = ChatGenerationChunk(message=ai_msg_chunk)
                    if run_manager:
                        run_manager.on_llm_new_token(content, chunk=chunk)
                    yield chunk
        finally:
            client.close()
