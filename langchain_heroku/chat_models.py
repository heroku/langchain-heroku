"""Heroku chat models."""

import json
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Union

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
from langchain_core.tools import BaseTool

from langchain_heroku.config import HerokuConfig


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

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        HerokuConfig.validate_config(inference_url=self.inference_url, api_key=self.api_key, model_id=self.model)

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
        self._validate_config()
        payload = self._build_streaming_payload(messages, stop)

        client = self._make_streaming_request(payload)
        try:
            for event in client.events():
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
        finally:
            client.close()

    def _convert_tools_to_api_format(self, tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]) -> List[Dict[str, Any]]:
        """Convert tools to the API format expected by Heroku Inference API."""
        api_tools = []
        
        for tool in tools:
            if isinstance(tool, dict):
                # Tool is already in dict format
                api_tools.append(tool)
            elif isinstance(tool, BaseTool):
                # Convert BaseTool to API format
                base_tool_function_def: Dict[str, Any] = {
                    "name": tool.name,
                    "description": tool.description,
                }
                
                # Add parameters if available
                if hasattr(tool, 'args_schema') and tool.args_schema:
                    try:
                        # Get the JSON schema from the Pydantic model
                        if hasattr(tool.args_schema, 'model_json_schema'):
                            schema = tool.args_schema.model_json_schema()
                            base_tool_function_def["parameters"] = schema
                    except Exception:
                        # Fallback if schema extraction fails
                        pass
                
                tool_def = {
                    "type": "function",
                    "function": base_tool_function_def
                }
                        
                api_tools.append(tool_def)
            elif callable(tool):
                # Handle callable functions - extract name and docstring
                func_name = getattr(tool, '__name__', str(tool))
                func_description = getattr(tool, '__doc__', f"Function {func_name}")
                
                callable_function_def: Dict[str, Any] = {
                    "name": func_name,
                    "description": func_description.strip() if func_description else func_name,
                }
                
                # Try to extract parameters from function signature
                try:
                    import inspect
                    sig = inspect.signature(tool)
                    properties: Dict[str, Any] = {}
                    required: List[str] = []
                    
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self':  # Skip self parameter
                            param_info = {"type": "string"}  # Default type
                            
                            # Add to required if no default value
                            if param.default == inspect.Parameter.empty:
                                required.append(param_name)
                                
                            properties[param_name] = param_info
                    
                    if properties:
                        parameters = {
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                        callable_function_def["parameters"] = parameters
                        
                except Exception:
                    # If signature extraction fails, continue without parameters
                    pass
                
                tool_def = {
                    "type": "function",
                    "function": callable_function_def
                }
                    
                api_tools.append(tool_def)
            elif isinstance(tool, type):
                # Handle class types - treat as tool schema
                class_name = getattr(tool, '__name__', str(tool))
                class_description = getattr(tool, '__doc__', f"Tool {class_name}")
                
                class_function_def: Dict[str, Any] = {
                    "name": class_name,
                    "description": class_description.strip() if class_description else class_name,
                }
                
                # Try to extract schema if it's a Pydantic model
                try:
                    if hasattr(tool, 'model_json_schema'):
                        schema = tool.model_json_schema()
                        class_function_def["parameters"] = schema
                except Exception:
                    pass
                
                tool_def = {
                    "type": "function", 
                    "function": class_function_def
                }
                    
                api_tools.append(tool_def)
        
        return api_tools

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str]] = None,
        **kwargs: Any,
    ) -> "ChatHeroku":
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
        
        # Create a new instance with the same parameters but with tools bound
        return self.__class__(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=self.stop,
            api_key=self.api_key,
            inference_url=self.inference_url,
            tools=api_tools,
            tool_choice=tool_choice,
            streaming=self.streaming,
            top_p=self.top_p,
            extended_thinking=self.extended_thinking,
        )
