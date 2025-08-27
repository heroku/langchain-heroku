"""Heroku chat models."""

import json
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Type, Union

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
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_heroku.config import HerokuClientConfig, HerokuConfig
from langchain_heroku.http_client import HerokuHTTPClient
from langchain_heroku.tool_converter import ToolConverter


class ChatHeroku(BaseChatModel):
    """
    Heroku chat model integration using the Inference API v1 /v1/chat/completions endpoint.

    Example setup (environment variables):
        export INFERENCE_URL="https://your-inference-api-url"
        export INFERENCE_KEY="your-heroku-inference-api-key"
        export INFERENCE_MODEL_ID="your-model-id"

    Basic usage:
        from langchain_heroku.chat_models import ChatHeroku
        from langchain_core.messages import HumanMessage

        chat = ChatHeroku()
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

        chat = ChatHeroku()
        result = chat(messages)

    Usage with explicit parameters:
        chat = ChatHeroku(
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
        chat = ChatHeroku(streaming=True)
        for chunk in chat.stream([HumanMessage(content="Hello!")]):
            print(chunk.message.content, end="")

    """

    # Core parameters (Pydantic fields)
    model: Optional[str] = Field(default=None, description="Model identifier")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    timeout: Optional[int] = Field(default=None, description="Request timeout in seconds")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    inference_url: Optional[str] = Field(default=None, description="Inference API URL")
    tools: Optional[List[dict]] = Field(default=None, description="Tools for function calling")
    tool_choice: Optional[Any] = Field(default=None, description="Tool selection strategy")
    streaming: bool = Field(default=False, description="Enable streaming responses")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter")
    extended_thinking: Optional[Dict[str, Any]] = Field(default=None, description="Extended thinking configuration")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ChatHeroku with configuration."""
        super().__init__(**kwargs)
        self._config: Optional[HerokuClientConfig] = None

    @property
    def _llm_type(self) -> str:
        return "heroku"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self.model}

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return HerokuConfig.get_env(key, default)

    def _get_api_key(self) -> Optional[str]:
        return HerokuConfig.get_api_key(self.api_key)

    def _get_inference_url(self) -> Optional[str]:
        return HerokuConfig.get_inference_url(self.inference_url)

    def _get_model(self) -> Optional[str]:
        return HerokuConfig.get_model_id(self.model)

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
        # Add model_name to response_metadata for usage tracking
        response_metadata = resp.copy()
        response_metadata["model_name"] = resp.get("model", self._get_model())
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
            usage_metadata=usage_metadata if usage_metadata else None,
        )

    def _get_config(self) -> HerokuClientConfig:
        """Get cached or create new configuration."""
        if self._config is None:
            self._config = HerokuConfig.create_client_config(
                inference_url=self.inference_url, api_key=self.api_key, model_id=self.model, timeout=self.timeout or 30, max_retries=2
            )
        return self._config

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        # This will raise HerokuConfigurationError if invalid
        self._get_config()

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
        config = self._get_config()
        return HerokuHTTPClient.make_request(
            url=config.inference_url,
            endpoint="v1/chat/completions",
            payload=payload,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

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
        config = self._get_config()
        return HerokuHTTPClient.make_streaming_request(
            url=config.inference_url,
            endpoint="v1/chat/completions",
            payload=payload,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    def _parse_sse_event(self, event: sseclient.Event) -> Optional[Dict[str, Any]]:
        """Parse a single SSE event and extract content and metadata."""
        try:
            # Handle the special "[DONE]" message
            if event.data == "[DONE]":
                return None

            data = json.loads(event.data)
            # For streaming, use 'delta' instead of 'message'
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")

            # Extract usage metadata if available
            usage = data.get("usage", {})
            usage_metadata = None
            if usage:
                usage_metadata = {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }

            return {
                "content": content,
                "usage_metadata": usage_metadata,
                "response_metadata": data,
            }
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
        """Stream chat completions with enhanced error handling and resource management."""
        try:
            self._validate_config()
            payload = self._build_streaming_payload(messages, stop)
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise

        client = None
        try:
            client = self._make_streaming_request(payload)

            for event in client.events():
                try:
                    parsed_event = self._parse_sse_event(event)
                    if parsed_event is not None:
                        content = parsed_event["content"]
                        usage_metadata = parsed_event.get("usage_metadata")
                        response_metadata = parsed_event.get("response_metadata", {})

                        # Add model_name to response_metadata for usage tracking
                        if response_metadata:
                            response_metadata["model_name"] = response_metadata.get("model", self._get_model())

                        ai_msg_chunk = AIMessageChunk(
                            content=content,
                            usage_metadata=usage_metadata,
                            response_metadata={"model_name": response_metadata.get("model", self._get_model())} if response_metadata else {},
                        )
                        chunk = ChatGenerationChunk(message=ai_msg_chunk)
                        if run_manager:
                            run_manager.on_llm_new_token(content, chunk=chunk)
                        yield chunk
                except Exception as e:
                    # Log the error but continue processing if possible
                    if run_manager:
                        run_manager.on_llm_error(e)
                    # Re-raise if it's a critical error
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    # For other errors, we can optionally log and continue or raise
                    raise
        except Exception as e:
            if run_manager:
                run_manager.on_llm_error(e)
            raise
        finally:
            # Ensure resources are always cleaned up
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass  # Ignore errors during cleanup

    def _convert_tools_to_api_format(self, tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]) -> List[Dict[str, Any]]:
        """Convert tools to the API format expected by Heroku Inference API."""
        return ToolConverter.convert_tools(tools)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> Any:  # Returns RunnableBinding
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model.
            tool_choice: The tool to use. If "any" then any tool can be used.
                Can be "auto", "required", or a specific tool name.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            A new ChatHeroku instance with tools bound.
        """
        # Convert tools to API format
        api_tools = self._convert_tools_to_api_format(tools)

        # Use the parent's bind method which returns RunnableBinding
        bound_kwargs = {"tools": api_tools, "tool_choice": tool_choice, **kwargs}
        return self.bind(**bound_kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel], Type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Create a runnable that returns structured output using tool calling.

        Since the Heroku API doesn't support native structured output via response_format,
        this method implements a workaround using tool calls to force the model to
        return structured JSON data.

        Args:
            schema: The schema to use for structured output. Can be:
                - A Pydantic model class
                - A dictionary representing a JSON schema
                - Any type with a model_json_schema method
            include_raw: Whether to include the raw tool call response alongside
                the parsed output. Defaults to False.
            **kwargs: Additional arguments (currently unused but kept for compatibility)

        Returns:
            A Runnable that will return structured output conforming to the schema.

        Example:
            ```python
            from pydantic import BaseModel
            from langchain_heroku import ChatHeroku

            class PersonInfo(BaseModel):
                name: str
                age: int
                occupation: str

            chat = ChatHeroku()
            structured_chat = chat.with_structured_output(PersonInfo)

            result = structured_chat.invoke("John is a 30-year-old software engineer")
            # Returns: PersonInfo(name="John", age=30, occupation="software engineer")
            ```
        """
        # Convert schema to tool format
        tool_name = "extract_data"
        tool_description = "Extract structured data from the input"

        if isinstance(schema, dict):
            # Already a JSON schema
            tool_schema = schema
        elif hasattr(schema, "model_json_schema"):
            # Pydantic model
            tool_schema = schema.model_json_schema()
        elif hasattr(schema, "__annotations__"):
            # Try to create a basic schema from annotations
            tool_schema = self._create_schema_from_annotations(schema)
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

        # Create the tool definition
        structured_tool = {"type": "function", "function": {"name": tool_name, "description": tool_description, "parameters": tool_schema}}

        # Create a model instance bound with the tool
        tool_model = self.bind_tools([structured_tool], tool_choice=tool_name)

        # Create the parser
        if include_raw:
            # Return both raw and parsed output
            def parse_output(ai_message: AIMessage) -> Dict[str, Any]:
                if not ai_message.tool_calls:
                    raise ValueError("No tool calls found in response")

                tool_call = ai_message.tool_calls[0]
                return {"raw": ai_message, "parsed": tool_call["args"]}

            return tool_model | parse_output
        else:
            # Use the built-in parser to extract just the tool arguments
            parser = JsonOutputKeyToolsParser(key_name=tool_name)
            return tool_model | parser

    def _create_schema_from_annotations(self, cls: Type) -> Dict[str, Any]:
        """Create a basic JSON schema from type annotations."""
        if not hasattr(cls, "__annotations__"):
            return {"type": "object", "properties": {}, "required": []}

        properties = {}
        required = []

        for field_name, field_type in cls.__annotations__.items():
            # Basic type mapping
            if field_type is str:
                properties[field_name] = {"type": "string"}
            elif field_type is int:
                properties[field_name] = {"type": "integer"}
            elif field_type is float:
                properties[field_name] = {"type": "number"}
            elif field_type is bool:
                properties[field_name] = {"type": "boolean"}
            elif hasattr(field_type, "__origin__"):
                # Handle generic types like Optional, List, etc.
                origin = getattr(field_type, "__origin__", None)
                if origin is Union:
                    # Handle Optional types
                    args = getattr(field_type, "__args__", ())
                    if len(args) == 2 and type(None) in args:
                        non_none_type = next(arg for arg in args if arg is not type(None))
                        if non_none_type is str:
                            properties[field_name] = {"type": "string"}
                        elif non_none_type is int:
                            properties[field_name] = {"type": "integer"}
                        elif non_none_type is float:
                            properties[field_name] = {"type": "number"}
                        elif non_none_type is bool:
                            properties[field_name] = {"type": "boolean"}
                        # Optional fields are not required
                        continue
                elif origin in (list, List):
                    properties[field_name] = {"type": "array"}
                else:
                    properties[field_name] = {"type": "string"}  # Default
            else:
                properties[field_name] = {"type": "string"}  # Default fallback

            # Add to required list (unless it was optional)
            required.append(field_name)

        return {"type": "object", "properties": properties, "required": required}
