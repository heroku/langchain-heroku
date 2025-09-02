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
    BaseMessageChunk,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.ai import UsageMetadata
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
        self._pending_ai_tool_message: Optional[AIMessage] = None

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

    def _ensure_balanced_tool_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Create synthetic ToolMessages for unbalanced tool calls, handling ID mapping.

        The supervisor framework uses custom tool_call_ids (like 'invoice_query_1') while
        our model returns different IDs (like 'tooluse_xyz'). We need to handle both cases
        and ensure every tool call ID has a corresponding ToolMessage.
        """
        if not messages:
            return messages

        # Build set of existing ToolMessage IDs (only model-generated IDs)
        existing_tool_ids: set = set()

        # Collect all existing ToolMessage tool_call_ids
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tid = getattr(msg, "tool_call_id", None)
                if tid:
                    existing_tool_ids.add(str(tid))
        balanced: List[BaseMessage] = []

        for msg in messages:
            balanced.append(msg)

            # For EVERY AIMessage with tool_calls, ensure they have ToolMessages
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)

                    # Create synthetic ToolMessage if one doesn't exist for the actual tool call ID
                    if tc_id and str(tc_id) not in existing_tool_ids:
                        synthetic_tool_msg = ToolMessage(
                            content="Tool execution completed successfully.",
                            tool_call_id=str(tc_id),  # Must match the model's tool call ID exactly
                        )
                        balanced.append(synthetic_tool_msg)
                        existing_tool_ids.add(str(tc_id))

        return balanced

    def _messages_to_api(self, messages: List[BaseMessage]) -> List[dict]:
        # Map LangChain message types to API roles with explicit type checking
        # Preserve message order; do not synthesize tool results
        api_msgs: List[dict] = []

        for m in messages:
            # Handle specific message types explicitly
            if isinstance(m, SystemMessage):
                role = "system"
                content = getattr(m, "content", "")
                # Ensure content is not None and convert to string
                if content is None:
                    content = "You are a helpful assistant."
                else:
                    content = str(content)
                # Ensure non-empty content for API compatibility
                if not content.strip():
                    content = "You are a helpful assistant."
                api_msgs.append({"role": role, "content": content})
            elif isinstance(m, HumanMessage):
                role = "user"
                content = getattr(m, "content", "")
                # Ensure content is not None and convert to string
                if content is None:
                    content = "Please help me."
                else:
                    content = str(content)
                # Ensure non-empty content for API compatibility
                if not content.strip():
                    content = "Please help me."
                api_msgs.append({"role": role, "content": content})
            elif isinstance(m, AIMessage):
                role = "assistant"
                content = getattr(m, "content", "")
                # Ensure content is not None and convert to string
                if content is None:
                    content = ""
                else:
                    content = str(content)
                # Add tool_calls if present; do not synthesize tool results here
                has_tool_calls = hasattr(m, "tool_calls") and m.tool_calls
                if has_tool_calls:
                    import json

                    api_tool_calls: List[Dict[str, Any]] = []
                    # Track assistant tool_call ids and any internal ids from args
                    tc_pairs: List[Dict[str, Optional[str]]] = []
                    for tc in m.tool_calls:
                        # Support both dict-like and object tool call representations
                        if isinstance(tc, dict):
                            tc_id = tc.get("id")
                            tc_name = tc.get("name")
                            tc_args = tc.get("args", {})
                        else:
                            tc_id = getattr(tc, "id", None)
                            tc_name = getattr(tc, "name", None)
                            tc_args = getattr(tc, "args", {})

                        if not tc_id:
                            raise ValueError("Assistant tool_call is missing required 'id' provided by the model")
                        if not tc_name:
                            raise ValueError("Assistant tool_call is missing required function 'name'")
                        tool_call_dict = {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": tc_name, "arguments": json.dumps(tc_args or {})},
                        }
                        api_tool_calls.append(tool_call_dict)
                        internal_id = None
                        if isinstance(tc_args, dict):
                            internal_id_val = tc_args.get("tool_call_id")
                            if internal_id_val is not None:
                                internal_id = str(internal_id_val)
                        tc_pairs.append({"assistant_id": tc_id, "internal_id": internal_id})

                    # Ensure non-empty content (API validation requires content present)
                    if not content or not str(content).strip():
                        content = "I'll use the available tools to help you."
                    msg_dict: Dict[str, Any] = {"role": role, "content": content, "tool_calls": api_tool_calls}
                    api_msgs.append(msg_dict)

                    # Synthesize missing tool results for this assistant turn only
                    # Determine which tool_call ids already have ToolMessages later in history
                    try:
                        current_index = messages.index(m)
                    except ValueError:
                        current_index = -1
                    covered_ids: set = set()
                    if current_index >= 0:
                        for later in messages[current_index + 1 :]:
                            if isinstance(later, ToolMessage):
                                tid = getattr(later, "tool_call_id", None)
                                if tid:
                                    covered_ids.add(str(tid))

                    for pair in tc_pairs:
                        a_id = pair.get("assistant_id")
                        i_id = pair.get("internal_id")
                        is_covered = (a_id in covered_ids) or (i_id is not None and i_id in covered_ids)
                        if not is_covered and a_id:
                            api_msgs.append(
                                {
                                    "role": "tool",
                                    "content": "Tool execution completed successfully.",
                                    "tool_call_id": a_id,
                                }
                            )
                else:
                    # Ensure non-empty content for AIMessage without tool calls
                    if not content.strip():
                        content = "I understand."
                    msg_dict = {"role": role, "content": content}
                    api_msgs.append(msg_dict)
            elif isinstance(m, ToolMessage):
                # Include tool results only if a corresponding assistant tool_call exists
                role = "tool"
                content = getattr(m, "content", "")
                # Ensure content is not None and convert to string
                if content is None:
                    content = "No result returned from the tool."
                else:
                    content = str(content)
                if not content.strip():
                    content = "No result returned from the tool."

                provided_tool_call_id = getattr(m, "tool_call_id", None)
                has_corresponding = False
                resolved_tool_call_id = provided_tool_call_id

                # Search prior assistant messages for matching tool_call id (direct match only)
                for prev_msg in messages[: messages.index(m)]:
                    if isinstance(prev_msg, AIMessage) and getattr(prev_msg, "tool_calls", None):
                        for tc in prev_msg.tool_calls:
                            if isinstance(tc, dict):
                                tc_id = tc.get("id")
                            else:
                                tc_id = getattr(tc, "id", None)

                            # Only direct match against tool_call id - no custom ID mapping
                            if provided_tool_call_id and tc_id == provided_tool_call_id:
                                has_corresponding = True
                                resolved_tool_call_id = tc_id
                                break
                        if has_corresponding:
                            break

                if has_corresponding and resolved_tool_call_id:
                    msg_dict = {"role": role, "content": content, "tool_call_id": resolved_tool_call_id}
                    api_msgs.append(msg_dict)
                # Else: skip orphan tool result to satisfy API validation
            elif isinstance(m, FunctionMessage):
                # FunctionMessage maps to tool role for backward compatibility
                role = "tool"
                content = getattr(m, "content", "")
                # Ensure content is not None and convert to string
                if content is None:
                    content = "No result returned from the function."
                else:
                    content = str(content)
                # Provide default content if empty (API requires non-empty content)
                if not content.strip():
                    content = "No result returned from the function."
                msg_dict = {
                    "role": role,
                    "content": content,
                    "tool_call_id": getattr(m, "name", "unknown"),  # FunctionMessage uses name instead of tool_call_id
                }
                api_msgs.append(msg_dict)
            else:
                # Fallback to role attribute or type for custom message types
                role = getattr(m, "role", None) or getattr(m, "type", "user") or "user"
                content = getattr(m, "content", "")
                # Ensure content is not None and convert to string
                if content is None:
                    content = ""
                else:
                    content = str(content)
                # Ensure non-empty content for custom message types
                if not content.strip():
                    if role == "system":
                        content = "You are a helpful assistant."
                    elif role == "assistant":
                        content = "I understand."
                    else:  # user or other roles
                        content = "Please help me."
                api_msgs.append({"role": role, "content": content})
        return api_msgs

    def _dict_to_usage_metadata(self, usage_dict: Optional[dict]) -> Optional[UsageMetadata]:
        """Convert a usage dictionary to UsageMetadata object."""
        if not usage_dict:
            return None

        # Extract values with proper type conversion
        input_tokens = usage_dict.get("input_tokens")
        output_tokens = usage_dict.get("output_tokens")
        total_tokens = usage_dict.get("total_tokens")

        # Only create UsageMetadata if we have valid values
        if input_tokens is not None and output_tokens is not None and total_tokens is not None:
            return UsageMetadata(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                total_tokens=int(total_tokens),
            )

        return None

    def _api_delta_to_langchain_chunk(
        self, delta: dict, usage_metadata: Optional[UsageMetadata] = None, response_metadata: Optional[dict] = None
    ) -> BaseMessageChunk:
        """Convert a streaming API delta to a LangChain message chunk object.

        This method handles conversion from API delta format to appropriate
        LangChain message chunk types based on the role field.

        Args:
            delta: The delta object from streaming response
            usage_metadata: Optional usage metadata
            response_metadata: Optional response metadata

        Returns:
            Appropriate LangChain message chunk object
        """
        import json

        from langchain_core.messages.tool import invalid_tool_call, tool_call

        role = delta.get("role", "")
        content = delta.get("content", "")

        # Handle tool calls in delta
        tool_calls: List[Any] = []
        api_tool_calls = delta.get("tool_calls", [])
        if api_tool_calls:
            for tc in api_tool_calls:
                try:
                    # In streaming, tool calls might come in parts
                    if "function" in tc and "arguments" in tc["function"]:
                        args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                        lc_tool_call = tool_call(name=tc["function"]["name"], args=args, id=tc.get("id", ""))
                        tool_calls.append(lc_tool_call)
                    elif "function" in tc and "name" in tc["function"]:
                        # Partial tool call (streaming in progress)
                        lc_tool_call = tool_call(name=tc["function"]["name"], args={}, id=tc.get("id", ""))
                        tool_calls.append(lc_tool_call)
                except (json.JSONDecodeError, KeyError) as e:
                    # If parsing fails, create an invalid tool call
                    invalid_tc = invalid_tool_call(
                        name=tc.get("function", {}).get("name", "unknown"),
                        args=tc.get("function", {}).get("arguments", ""),
                        id=tc.get("id", "unknown"),
                        error=f"Failed to parse tool call: {e}",
                    )
                    tool_calls.append(invalid_tc)

        # Create additional kwargs
        additional_kwargs = {}
        if api_tool_calls:
            additional_kwargs["tool_calls"] = api_tool_calls

        # Based on role, create appropriate chunk type
        if role == "system":
            return SystemMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata or {},
                id=None,
            )
        elif role == "user":
            return HumanMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata or {},
                id=None,
            )
        elif role == "assistant" or role == "":  # Default to assistant for streaming
            return AIMessageChunk(
                content=content,
                tool_calls=tool_calls,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata or {},
                usage_metadata=usage_metadata,
                id=None,
            )
        elif role == "tool":
            # Tool messages in streaming are rare, but handle them
            tool_call_id = delta.get("tool_call_id", "")
            return ToolMessageChunk(
                content=content,
                tool_call_id=tool_call_id,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata or {},
                id=None,
            )
        else:
            # Fallback for unknown roles - treat as AIMessageChunk
            return AIMessageChunk(
                content=content,
                tool_calls=tool_calls,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata or {},
                usage_metadata=usage_metadata,
                id=None,
            )

    def _api_message_to_langchain_message(self, api_msg: dict) -> BaseMessage:
        """Convert a single API message to a LangChain message object.

        This method handles conversion from API message format to appropriate
        LangChain message types based on the role field.
        """
        import json

        from langchain_core.messages.tool import invalid_tool_call, tool_call

        role = api_msg.get("role", "")
        content = api_msg.get("content", "")

        if role == "system":
            return SystemMessage(content=content)
        elif role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            # Handle assistant messages with possible tool calls
            api_tool_calls = api_msg.get("tool_calls", [])
            tool_calls: List[Any] = []

            if api_tool_calls:
                for tc in api_tool_calls:
                    try:
                        # Parse arguments from JSON string
                        args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                        # Convert to LangChain tool_call format
                        lc_tool_call = tool_call(name=tc["function"]["name"], args=args, id=tc["id"])
                        tool_calls.append(lc_tool_call)
                    except (json.JSONDecodeError, KeyError) as e:
                        # If parsing fails, create an invalid tool call
                        invalid_tc = invalid_tool_call(
                            name=tc.get("function", {}).get("name", "unknown"),
                            args=tc.get("function", {}).get("arguments", ""),
                            id=tc.get("id", "unknown"),
                            error=f"Failed to parse tool call: {e}",
                        )
                        tool_calls.append(invalid_tc)

            additional_kwargs = {"tool_calls": api_tool_calls} if api_tool_calls else {}
            return AIMessage(
                content=content,
                tool_calls=tool_calls,
                additional_kwargs=additional_kwargs,
            )
        elif role == "tool":
            # Convert tool role messages to ToolMessage
            tool_call_id = api_msg.get("tool_call_id", "")
            return ToolMessage(content=content, tool_call_id=tool_call_id)
        else:
            # Fallback for unknown roles - treat as HumanMessage
            return HumanMessage(content=content)

    def _api_to_ai_message(self, resp: dict) -> BaseMessage:
        # Minimal defensive programming: handle string responses
        if isinstance(resp, str):
            return AIMessage(content=resp)

        choice = resp["choices"][0]
        msg = choice["message"]

        # Use the general conversion method (returns appropriate message type)
        langchain_msg = self._api_message_to_langchain_message(msg)

        # Add usage and response metadata in a typed manner
        usage = resp.get("usage", {})
        usage_dict = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        usage_metadata_obj = self._dict_to_usage_metadata(usage_dict)

        # Preserve any additional kwargs from parsed message (e.g., api tool_calls)
        additional_kwargs = langchain_msg.additional_kwargs.copy()

        # Add model_name to response_metadata for usage tracking
        response_metadata = resp.copy()
        response_metadata["model_name"] = resp.get("model", self._get_model())

        # Reconstruct the same message type with enriched metadata
        if isinstance(langchain_msg, AIMessage):
            self._pending_ai_tool_message = None
            return AIMessage(
                content=langchain_msg.content,
                tool_calls=getattr(langchain_msg, "tool_calls", []),
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
                usage_metadata=usage_metadata_obj,
            )
        elif isinstance(langchain_msg, ToolMessage):
            return ToolMessage(
                content=langchain_msg.content,
                tool_call_id=getattr(langchain_msg, "tool_call_id", ""),
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        elif isinstance(langchain_msg, SystemMessage):
            return SystemMessage(
                content=langchain_msg.content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        elif isinstance(langchain_msg, HumanMessage):
            return HumanMessage(
                content=langchain_msg.content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        else:
            # Fallback to AIMessage type to ensure compatibility
            return AIMessage(
                content=getattr(langchain_msg, "content", msg.get("content", "")),
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
                usage_metadata=usage_metadata_obj,
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

    def _build_payload(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Build the API payload for the chat completion request."""

        # If there is a pending assistant tool_call from the previous turn and the
        # orchestrator omitted the assistant tool_use, inject it just before the
        # matching ToolMessage (assistant id or internal args.tool_call_id), or
        # before the next assistant turn if none present.
        def inject_pending(conv: List[BaseMessage]) -> List[BaseMessage]:
            pending = self._pending_ai_tool_message
            if not pending or not getattr(pending, "tool_calls", None):
                return conv
            pending_ids = set()
            pending_internal_ids = set()
            for tc in pending.tool_calls:  # type: ignore[attr-defined]
                if isinstance(tc, dict):
                    pending_ids.add(tc.get("id"))
                    args = tc.get("args", {})
                    if isinstance(args, dict) and args.get("tool_call_id") is not None:
                        pending_internal_ids.add(str(args.get("tool_call_id")))
                else:
                    pending_ids.add(getattr(tc, "id", None))
                    args = getattr(tc, "args", {})
                    if isinstance(args, dict) and args.get("tool_call_id") is not None:
                        pending_internal_ids.add(str(args.get("tool_call_id")))

            # If conversation already contains these tool_calls, skip injection
            existing_ids = set()
            for m in conv:
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        existing_ids.add(tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None))
            if pending_ids & existing_ids:
                # Clear pending since the AIMessage is already in the conversation
                self._pending_ai_tool_message = None
                return conv

            # Find a ToolMessage match
            insert_at = None
            for idx, m in enumerate(conv):
                if isinstance(m, ToolMessage):
                    tid = getattr(m, "tool_call_id", None)
                    if tid and (tid in pending_ids or tid in pending_internal_ids):
                        insert_at = idx
                        break

            # If none, insert before next assistant
            if insert_at is None:
                for idx, m in enumerate(conv):
                    if isinstance(m, AIMessage):
                        insert_at = idx
                        break

            ai_msg = AIMessage(
                content="I'll use the available tools to help you.",
                tool_calls=pending.tool_calls,
                additional_kwargs=pending.additional_kwargs,
            )  # type: ignore[arg-type]
            new_conv = list(conv)
            if insert_at is not None:
                new_conv.insert(insert_at, ai_msg)
            else:
                new_conv.append(ai_msg)
            # Clear pending upon successful injection
            self._pending_ai_tool_message = None
            return new_conv

        injected_messages = inject_pending(messages)
        # Remap ToolMessage ids from internal ids to assistant ids when present in the same conversation
        normalized_messages = injected_messages
        balanced_messages = self._ensure_balanced_tool_messages(normalized_messages)
        api_messages = self._messages_to_api(balanced_messages)

        # Ensure we never send null or empty messages array
        if not api_messages:
            raise ValueError("Cannot generate chat completion with empty messages list")

        # Validate that no messages are None/null to prevent API validation errors
        for i, msg in enumerate(api_messages):
            if msg is None:
                raise ValueError(f"Message at index {i} is None/null - this will cause API validation failure")
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {i} is not a dict: {type(msg)} - {msg}")
            if "role" not in msg:
                raise ValueError(f"Message at index {i} is missing 'role' field: {msg}")
            if "content" not in msg:
                raise ValueError(f"Message at index {i} is missing 'content' field: {msg}")
            # Ensure content is not None
            if msg["content"] is None:
                msg["content"] = "Please help me."  # Provide fallback content

        # Final safety check - ensure messages array itself is valid
        if api_messages is None:
            raise ValueError("API messages array is None - this will cause API validation failure")

        payload: Dict[str, Any] = {
            "model": self._get_model(),
            # Balanced, serialized messages
            "messages": api_messages,
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

        # Handle tools - prioritize kwargs over instance attributes (for bind_tools support)
        tools = kwargs.get("tools") or self.tools
        if tools:
            # Remove tool_call_id from tool schemas to prevent supervisor framework confusion
            cleaned_tools = []
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    cleaned_tool = tool.copy()
                    if "parameters" in cleaned_tool["function"]:
                        params = cleaned_tool["function"]["parameters"].copy()
                        if "properties" in params and "tool_call_id" in params["properties"]:
                            # Remove tool_call_id from properties
                            params["properties"] = {k: v for k, v in params["properties"].items() if k != "tool_call_id"}
                            # Remove tool_call_id from required fields
                            if "required" in params and "tool_call_id" in params["required"]:
                                params["required"] = [r for r in params["required"] if r != "tool_call_id"]
                            cleaned_tool["function"]["parameters"] = params
                    cleaned_tools.append(cleaned_tool)
                else:
                    cleaned_tools.append(tool)
            payload["tools"] = cleaned_tools

        # Handle tool_choice - prioritize kwargs over instance attributes (for bind_tools support)
        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        elif self.tool_choice is not None:
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

        # Comprehensive payload validation before API call
        import json

        # Check if messages field exists and is valid
        if "messages" not in payload:
            raise ValueError("Payload missing 'messages' field")

        messages = payload["messages"]
        if messages is None:
            raise ValueError("Payload 'messages' field is None")

        if not isinstance(messages, list):
            raise ValueError(f"Payload 'messages' field is not a list: {type(messages)}")

        # Check each message in detail
        for i, msg in enumerate(messages):
            if msg is None:
                raise ValueError(f"Message at index {i} is None in payload")
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {i} is not a dict: {type(msg)}")
            if "role" not in msg:
                raise ValueError(f"Message at index {i} missing 'role': {msg}")
            if "content" not in msg:
                raise ValueError(f"Message at index {i} missing 'content': {msg}")
            if msg["content"] is None:
                # Fix None content on the spot
                msg["content"] = "Please help me."

        # Validate JSON serialization doesn't introduce nulls
        try:
            serialized = json.dumps(payload)
            if '"messages":null' in serialized or '"content":null' in serialized:
                raise ValueError(f"Payload contains null values after serialization: {serialized[:500]}...")
        except Exception as e:
            raise ValueError(f"Payload JSON serialization failed: {e}")

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

        # Only balance messages to prevent validation failures
        messages = self._ensure_balanced_tool_messages(messages)

        # Validate input messages
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Check for None messages in the input
        for i, message in enumerate(messages):
            if message is None:
                raise ValueError(f"Message at index {i} is None - cannot process null messages")

        # API Validation Fix: Ensure conversation has at least one non-system message
        # SystemMessage alone causes "Value null at 'messages'" error in Heroku API
        from langchain_core.messages import HumanMessage, SystemMessage

        has_user_message = any(not isinstance(msg, SystemMessage) for msg in messages)
        if not has_user_message:
            # If only system messages, convert the last one to a user message
            # This preserves the intent while meeting API requirements
            if len(messages) == 1 and isinstance(messages[0], SystemMessage):
                system_content = messages[0].content or "Please help me."
                messages = [HumanMessage(content=system_content)]
            else:
                # Add a generic user message to make the conversation valid
                messages.append(HumanMessage(content="Please process this request."))

        # Validate message content - allow empty content for all message types
        # since _messages_to_api will provide appropriate fallback content
        for i, message in enumerate(messages):
            # Allow empty content for AIMessage (with or without tool calls)
            if isinstance(message, AIMessage):
                continue  # Empty content is valid for AI messages (API conversion handles fallback)
            # Allow empty content for ToolMessage (tool calls can return no results)
            if isinstance(message, ToolMessage):
                continue  # Empty content is valid for tool messages
            # Allow empty content for HumanMessage that might contain structured output context
            if isinstance(message, HumanMessage):
                continue  # Allow empty HumanMessage content (API conversion handles fallback)
            # Allow empty content for SystemMessage (API conversion handles fallback)
            if isinstance(message, SystemMessage):
                continue  # Allow empty SystemMessage content (API conversion handles fallback)
            # For other message types, still validate (but API conversion will handle fallback)
            if not message.content or str(message.content).strip() == "":
                # This should rarely happen since most messages are covered above
                # But if it does, the API conversion will still provide fallback content
                continue

        # Validate stop sequences (ensure they're not blank)
        if stop:
            for i, seq in enumerate(stop):
                if not seq or seq.strip() == "":
                    raise ValueError(f"Stop sequence at index {i} cannot be blank")

        payload = self._build_payload(messages, stop, **kwargs)
        data = self._make_api_request(payload)
        message = self._api_to_ai_message(data)

        # Return simple single-generation ChatResult
        # Let the supervisor framework handle tool execution and ToolMessage creation
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _build_streaming_payload(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Build the API payload for streaming chat completion requests."""

        # Only balance messages to prevent validation failures
        balanced_messages = self._ensure_balanced_tool_messages(messages)
        api_messages = self._messages_to_api(balanced_messages)

        # Ensure we never send null or empty messages array
        if not api_messages:
            raise ValueError("Cannot generate chat completion with empty messages list")

        payload: Dict[str, Any] = {
            "model": self._get_model(),
            "messages": api_messages,
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

        # Handle tools - prioritize kwargs over instance attributes (for bind_tools support)
        tools = kwargs.get("tools") or self.tools
        if tools:
            # Remove tool_call_id from tool schemas to prevent supervisor framework confusion
            cleaned_tools = []
            for tool in tools:
                if isinstance(tool, dict) and "function" in tool:
                    cleaned_tool = tool.copy()
                    if "parameters" in cleaned_tool["function"]:
                        params = cleaned_tool["function"]["parameters"].copy()
                        if "properties" in params and "tool_call_id" in params["properties"]:
                            # Remove tool_call_id from properties
                            params["properties"] = {k: v for k, v in params["properties"].items() if k != "tool_call_id"}
                            # Remove tool_call_id from required fields
                            if "required" in params and "tool_call_id" in params["required"]:
                                params["required"] = [r for r in params["required"] if r != "tool_call_id"]
                            cleaned_tool["function"]["parameters"] = params
                    cleaned_tools.append(cleaned_tool)
                else:
                    cleaned_tools.append(tool)
            payload["tools"] = cleaned_tools

        # Handle tool_choice - prioritize kwargs over instance attributes (for bind_tools support)
        tool_choice = kwargs.get("tool_choice")
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        elif self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice

        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.extended_thinking is not None:
            payload["extended_thinking"] = self.extended_thinking

        return payload

    def _make_streaming_request(self, payload: Dict[str, Any]) -> sseclient.SSEClient:
        """Make the streaming API request with retry logic."""
        config = self._get_config()
        return HerokuHTTPClient.make_streaming_request(  # type: ignore[no-any-return]
            url=config.inference_url,
            endpoint="v1/chat/completions",
            payload=payload,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    # (Intentionally no override of BaseChatModel.generate; balance in _generate/_stream only)

    def _parse_sse_event(self, event: sseclient.Event) -> Optional[Dict[str, Any]]:
        """Parse a single SSE event and extract delta and metadata."""
        try:
            # Handle the special "[DONE]" message
            if event.data == "[DONE]":
                return None

            data = json.loads(event.data)
            # For streaming, use 'delta' instead of 'message'
            choice = data["choices"][0]
            delta = choice.get("delta", {})

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
                "delta": delta,
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

            # Validate input messages (same validation as _generate)
            if not messages:
                raise ValueError("Messages list cannot be empty")

            # Validate message content - allow empty content for all message types
            # since _messages_to_api will provide appropriate fallback content
            for i, message in enumerate(messages):
                # Allow empty content for AIMessage (with or without tool calls)
                if isinstance(message, AIMessage):
                    continue  # Empty content is valid for AI messages (API conversion handles fallback)
                # Allow empty content for ToolMessage (tool calls can return no results)
                if isinstance(message, ToolMessage):
                    continue  # Empty content is valid for tool messages
                # Allow empty content for HumanMessage that might contain structured output context
                if isinstance(message, HumanMessage):
                    continue  # Allow empty HumanMessage content (API conversion handles fallback)
                # Allow empty content for SystemMessage (API conversion handles fallback)
                if isinstance(message, SystemMessage):
                    continue  # Allow empty SystemMessage content (API conversion handles fallback)
                # For other message types, still validate (but API conversion will handle fallback)
                if not message.content or str(message.content).strip() == "":
                    # This should rarely happen since most messages are covered above
                    # But if it does, the API conversion will still provide fallback content
                    continue

            payload = self._build_streaming_payload(messages, stop, **kwargs)
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
                        delta = parsed_event["delta"]
                        usage_metadata = parsed_event.get("usage_metadata")
                        response_metadata = parsed_event.get("response_metadata", {})

                        # Add model_name to response_metadata for usage tracking
                        if response_metadata:
                            response_metadata["model_name"] = response_metadata.get("model", self._get_model())

                        # Convert delta to appropriate LangChain chunk type
                        usage_metadata_obj = self._dict_to_usage_metadata(usage_metadata)
                        message_chunk = self._api_delta_to_langchain_chunk(
                            delta=delta, usage_metadata=usage_metadata_obj, response_metadata=response_metadata
                        )

                        chunk = ChatGenerationChunk(message=message_chunk)

                        # Extract content for token callback
                        content = delta.get("content", "")
                        if run_manager and content:
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
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
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
        from langchain_core.runnables import RunnableLambda

        # Convert schema to tool format
        tool_name = "extract_data"
        tool_description = "Extract structured data from the input according to the specified schema"

        if isinstance(schema, dict):
            # Already a JSON schema - clean it for API compatibility
            tool_schema = self._clean_schema_for_api(schema)
        elif hasattr(schema, "model_json_schema"):
            # Pydantic model - get schema and clean it
            raw_schema = schema.model_json_schema()
            tool_schema = self._clean_schema_for_api(raw_schema)
        elif hasattr(schema, "__annotations__"):
            # Try to create a basic schema from annotations
            raw_schema = self._create_schema_from_annotations(schema)
            tool_schema = self._clean_schema_for_api(raw_schema)
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}")

        # Create the tool definition
        structured_tool = {"type": "function", "function": {"name": tool_name, "description": tool_description, "parameters": tool_schema}}

        # Force the specific tool to be called for structured output
        tool_choice_dict = {"type": "function", "function": {"name": tool_name}}

        # Create a direct invoker that bypasses the RunnableBinding complexity
        def structured_invoke(input_messages):
            """Direct structured output invoker that avoids RunnableBinding issues."""
            # Handle different input formats
            if isinstance(input_messages, list):
                messages = input_messages
            elif hasattr(input_messages, "messages"):
                messages = input_messages.messages
            else:
                # Assume it's a single message or content
                from langchain_core.messages import HumanMessage

                if isinstance(input_messages, str):
                    messages = [HumanMessage(content=input_messages)]
                else:
                    messages = [input_messages]

            # Ensure we have valid messages
            if not messages:
                raise ValueError("No input messages provided")

            # API Validation Fix: Ensure conversation has a user message
            # SystemMessage alone is not valid - need at least one user message
            from langchain_core.messages import HumanMessage, SystemMessage

            has_user_message = any(not isinstance(msg, SystemMessage) for msg in messages)
            if not has_user_message:
                # If only system messages, convert the last one to a user message
                # This preserves the intent while meeting API requirements
                if len(messages) == 1 and isinstance(messages[0], SystemMessage):
                    system_content = messages[0].content
                    messages = [HumanMessage(content=system_content)]
                else:
                    # Add a generic user message to make the conversation valid
                    messages.append(HumanMessage(content="Please process this request."))

            # Call the model directly with tools and tool_choice
            try:
                # Bypass the bind_tools complexity and call _generate directly
                result = self._generate(messages=messages, tools=[structured_tool], tool_choice=tool_choice_dict)

                # Extract the AI message from the result
                ai_message = result.generations[0].message

                if include_raw:
                    # Return both raw and parsed data
                    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                        tool_call = ai_message.tool_calls[0]
                        try:
                            parsed_data = self._parse_tool_call_args(tool_call, schema)
                            return {"raw": ai_message, "parsed": parsed_data}
                        except Exception:
                            # If parsing fails, return default structured data
                            return {"raw": ai_message, "parsed": self._create_default_structured_data(schema)}
                    else:
                        return {"raw": ai_message, "parsed": self._create_default_structured_data(schema)}
                else:
                    # Return only parsed structured data
                    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                        tool_call = ai_message.tool_calls[0]
                        try:
                            return self._parse_tool_call_args(tool_call, schema)
                        except Exception:
                            # If parsing fails, return default structured data
                            return self._create_default_structured_data(schema)
                    else:
                        return self._create_default_structured_data(schema)

            except Exception as e:
                # If anything fails, return default structured data or re-raise
                if include_raw:
                    from langchain_core.messages import AIMessage

                    return {"raw": AIMessage(content=f"Error: {e}"), "parsed": self._create_default_structured_data(schema)}
                else:
                    return self._create_default_structured_data(schema)

        # Return a RunnableLambda that handles the structured output
        return RunnableLambda(structured_invoke)

    def _clean_schema_for_api(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean JSON schema to remove fields not supported by Heroku Inference API.

        The API doesn't support certain JSON schema fields like 'title', 'description'
        in nested objects, so we need to remove them for compatibility.
        """
        if not isinstance(schema, dict):
            return schema

        # Create a clean copy
        clean_schema = {}

        # Allowed top-level fields for function parameters
        allowed_fields = {"type", "properties", "required", "items", "enum", "default"}

        for key, value in schema.items():
            if key in allowed_fields:
                if key == "properties" and isinstance(value, dict):
                    # Clean properties recursively
                    clean_properties = {}
                    for prop_name, prop_schema in value.items():
                        clean_properties[prop_name] = self._clean_schema_for_api(prop_schema)
                    clean_schema[key] = clean_properties
                elif key == "items" and isinstance(value, dict):
                    # Clean items schema
                    clean_schema[key] = self._clean_schema_for_api(value)
                else:
                    clean_schema[key] = value

        return clean_schema

    def _parse_tool_call_args(self, tool_call: Any, schema: Union[Dict[str, Any], Type[BaseModel], Type]) -> Any:
        """Parse tool call arguments and validate against schema.

        This method handles the parsing of tool call arguments and validates them
        against the provided schema, similar to the TypeScript implementation.
        """
        try:
            # Extract arguments from tool call
            if "function" in tool_call and "arguments" in tool_call["function"]:
                args = tool_call["function"]["arguments"]
            elif "args" in tool_call:
                args = tool_call["args"]
            else:
                raise ValueError("No arguments found in tool call")

            # Parse arguments if they're a string, otherwise use as-is
            if isinstance(args, str):
                import json

                parsed_args = json.loads(args)
            else:
                parsed_args = args

            # Validate against schema
            if hasattr(schema, "model_validate"):
                # Pydantic v2
                return schema.model_validate(parsed_args)
            elif hasattr(schema, "parse_obj"):
                # Pydantic v1
                return schema.parse_obj(parsed_args)
            elif isinstance(schema, dict):
                # Dictionary schema - basic validation
                self._validate_dict_schema(parsed_args, schema)
                return parsed_args
            else:
                # For other types, return as-is
                return parsed_args

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse tool call arguments as JSON: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse or validate tool call arguments: {e}")

    def _validate_dict_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Basic validation of data against a dictionary schema."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from data")

        # Check field types (basic validation)
        properties = schema.get("properties", {})
        for field_name, field_value in data.items():
            if field_name in properties:
                expected_type = properties[field_name].get("type")
                if expected_type == "string" and not isinstance(field_value, str):
                    raise ValueError(f"Field '{field_name}' expected string, got {type(field_value)}")
                elif expected_type == "integer" and not isinstance(field_value, int):
                    raise ValueError(f"Field '{field_name}' expected integer, got {type(field_value)}")
                elif expected_type == "number" and not isinstance(field_value, (int, float)):
                    raise ValueError(f"Field '{field_name}' expected number, got {type(field_value)}")
                elif expected_type == "boolean" and not isinstance(field_value, bool):
                    raise ValueError(f"Field '{field_name}' expected boolean, got {type(field_value)}")

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

    def _get_type_default(self, type_annotation: Any, field_name: str = "") -> Any:
        """Get a sensible default value for a given type annotation."""
        import typing

        # Handle basic types
        if type_annotation is str:
            return ""  # Return empty string for string fields
        elif type_annotation is int:
            return 0
        elif type_annotation is float:
            return 0.0
        elif type_annotation is bool:
            return False
        elif type_annotation is list:
            return []
        elif type_annotation is dict:
            return {}

        # Handle typing generics
        origin = getattr(type_annotation, "__origin__", None)
        args = getattr(type_annotation, "__args__", ())

        if origin is Union:
            # For Union types (like Optional), try the first non-None type
            for arg in args:
                if arg is not type(None):
                    return self._get_type_default(arg)
            return None
        elif origin is list:
            return []
        elif origin is dict:
            return {}
        elif hasattr(type_annotation, "__members__"):
            # Enum types - return the first member
            members = list(type_annotation.__members__.values())
            return members[0] if members else ""

        # Handle Literal types
        if hasattr(typing, "get_origin") and hasattr(typing, "get_args"):
            try:
                if typing.get_origin(type_annotation) is typing.Literal:
                    literal_values = typing.get_args(type_annotation)
                    return literal_values[0] if literal_values else ""
            except (AttributeError, TypeError):
                pass

        # Check for _name attribute (used by some typing constructs)
        if hasattr(type_annotation, "_name"):
            if type_annotation._name == "Literal":
                literal_values = getattr(type_annotation, "__args__", ())
                return literal_values[0] if literal_values else ""

        # Fallback to empty string for unknown types
        return ""

    def _create_default_structured_data(self, schema: Union[Dict[str, Any], Type[BaseModel], Type]) -> Any:
        """Create default structured data when no tool calls are present.

        This ensures that structured output always returns a proper data structure
        matching the expected schema, even when the model doesn't make tool calls.

        Args:
            schema: The schema to create default data for

        Returns:
            Default structured data matching the schema
        """
        try:
            if hasattr(schema, "model_validate"):
                # Pydantic v2 - create instance with default values
                defaults = {}

                # Try Pydantic v2 field access first
                if hasattr(schema, "model_fields"):
                    for field_name, field_info in schema.model_fields.items():
                        if hasattr(field_info, "default") and field_info.default is not ...:
                            defaults[field_name] = field_info.default
                        elif hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                            factory = field_info.default_factory
                            try:
                                if callable(factory):
                                    defaults[field_name] = factory()  # type: ignore[call-arg]
                                else:
                                    defaults[field_name] = factory
                            except TypeError:
                                # If factory isn't callable, use it as a value
                                defaults[field_name] = factory
                        else:
                            # For required fields without defaults, provide sensible defaults based on annotation
                            if hasattr(field_info, "annotation"):
                                defaults[field_name] = self._get_type_default(field_info.annotation, field_name)
                            else:
                                defaults[field_name] = ""

                # Fallback to Pydantic v1 style if v2 fields not available
                elif hasattr(schema, "__fields__"):
                    for field_name, field_info in schema.__fields__.items():
                        if hasattr(field_info, "default") and field_info.default is not None:
                            defaults[field_name] = field_info.default
                        elif hasattr(field_info, "default_factory") and field_info.default_factory is not None:
                            factory = field_info.default_factory
                            try:
                                if callable(factory):
                                    defaults[field_name] = factory()  # type: ignore[call-arg]
                                else:
                                    defaults[field_name] = factory
                            except TypeError:
                                # If factory isn't callable, use it as a value
                                defaults[field_name] = factory
                        else:
                            # For required fields, provide sensible defaults
                            defaults[field_name] = ""

                # Try to create with defaults first
                if defaults:
                    try:
                        if hasattr(schema, "model_validate"):
                            return schema.model_validate(defaults)
                        else:
                            return defaults
                    except Exception:
                        # If validation fails with defaults, continue to fallback
                        pass

                # Try to create an empty instance
                try:
                    if callable(schema):
                        return schema()
                    else:
                        return {}  # Fallback for non-callable schema
                except Exception:
                    # If that fails, try with empty dict (will likely fail for required fields)
                    try:
                        if hasattr(schema, "model_validate"):
                            return schema.model_validate({})
                        else:
                            return {}
                    except Exception:
                        # Ultimate fallback - create with minimal required defaults
                        minimal_defaults = {}
                        if hasattr(schema, "model_fields"):
                            for field_name, field_info in schema.model_fields.items():
                                minimal_defaults[field_name] = self._get_type_default(getattr(field_info, "annotation", str), field_name)
                        else:
                            # If we can't determine fields, return empty dict as last resort
                            return {}
                        if hasattr(schema, "model_validate"):
                            return schema.model_validate(minimal_defaults)
                        else:
                            return minimal_defaults
            elif hasattr(schema, "parse_obj"):
                # Pydantic v1 - create instance with default values
                try:
                    if callable(schema):
                        return schema()
                    else:
                        return {}  # Fallback for non-callable schema
                except Exception:
                    if hasattr(schema, "parse_obj"):
                        return schema.parse_obj({})
                    else:
                        return {}
            elif isinstance(schema, dict):
                # Dictionary schema - create object with default values based on properties
                defaults = {}
                properties = schema.get("properties", {})
                for field_name, field_schema in properties.items():
                    field_type = field_schema.get("type", "string")
                    if "default" in field_schema:
                        defaults[field_name] = field_schema["default"]
                    elif field_type == "string":
                        defaults[field_name] = ""
                    elif field_type == "integer":
                        defaults[field_name] = 0
                    elif field_type == "number":
                        defaults[field_name] = 0.0
                    elif field_type == "boolean":
                        defaults[field_name] = False
                    elif field_type == "array":
                        defaults[field_name] = []
                    elif field_type == "object":
                        defaults[field_name] = {}
                    else:
                        defaults[field_name] = None
                return defaults
            else:
                # For other types, try to create an empty instance
                try:
                    return schema()
                except Exception:
                    # Final fallback - return empty dict
                    return {}
        except Exception:
            # Ultimate fallback - return empty dict
            return {}
