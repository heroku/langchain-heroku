"""Heroku chat models."""

from typing import Any, Dict, Iterator, List, Optional
import os
import httpx

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

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

    Usage with explicit parameters:
        chat = ChatHeroku(
            model_name="your-model-id",
            api_key="your-heroku-inference-api-key",
            inference_url="https://your-inference-api-url",
            temperature=0.7,
            max_tokens=256,
            stop=["\n"],
            tool_schemas=[{"type": "function", ...}],
            stream=False,
        )
        result = chat([HumanMessage(content="Hello!")])
        print(result.generations[0].message.content)

    Streaming usage:
        chat = ChatHeroku(stream=True)
        for chunk in chat.stream([HumanMessage(content="Hello!")]):
            print(chunk.message.content, end="")

    """

    model_name: Optional[str] = Field(default=None, alias="model")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    api_key: Optional[str] = None
    inference_url: Optional[str] = None
    tool_schemas: Optional[List[dict]] = None  # For tool calling
    stream: bool = False

    @property
    def _llm_type(self) -> str:
        return "chat-heroku"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, default)

    def _get_api_key(self) -> str:
        return self.api_key or self._get_env("INFERENCE_KEY") or self._get_env("HEROKU_API_KEY")

    def _get_inference_url(self) -> str:
        return self.inference_url or self._get_env("INFERENCE_URL")

    def _get_model(self) -> str:
        return self.model_name or self._get_env("INFERENCE_MODEL_ID")

    def _messages_to_api(self, messages: List[BaseMessage]) -> List[dict]:
        # Map LangChain message roles to API roles
        role_map = {"human": "user", "ai": "assistant", "system": "system", "user": "user", "assistant": "assistant", "tool": "tool"}
        api_msgs = []
        for m in messages:
            role = getattr(m, "role", None) or role_map.get(getattr(m, "type", ""), "user")
            content = getattr(m, "content", "")
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
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            response_metadata=resp,
            usage_metadata=usage_metadata,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        url = self._get_inference_url()
        if not url:
            raise ValueError("INFERENCE_URL must be set via env or init param.")
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("INFERENCE_KEY or HEROKU_API_KEY must be set via env or init param.")
        model = self._get_model()
        if not model:
            raise ValueError("model or INFERENCE_MODEL_ID must be set via env or init param.")
        payload = {
            "model": model,
            "messages": self._messages_to_api(messages),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop is not None:
            payload["stop"] = stop
        elif self.stop is not None:
            payload["stop"] = self.stop
        if self.tool_schemas:
            payload["tools"] = self.tool_schemas
        if self.stream:
            payload["stream"] = True
        timeout = self.timeout or 60
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        for _ in range(self.max_retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(f"{url}/v1/chat/completions", json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    break
            except Exception as e:
                last_exc = e
        else:
            raise RuntimeError(f"Heroku Inference API call failed after {self.max_retries} retries: {last_exc}")
        ai_msg = self._api_to_ai_message(data)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        url = self._get_inference_url()
        if not url:
            raise ValueError("INFERENCE_URL must be set via env or init param.")
        api_key = self._get_api_key()
        if not api_key:
            raise ValueError("INFERENCE_KEY or HEROKU_API_KEY must be set via env or init param.")
        model = self._get_model()
        if not model:
            raise ValueError("model or INFERENCE_MODEL_ID must be set via env or init param.")
        payload = {
            "model": model,
            "messages": self._messages_to_api(messages),
            "stream": True,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if stop is not None:
            payload["stop"] = stop
        elif self.stop is not None:
            payload["stop"] = self.stop
        if self.tool_schemas:
            payload["tools"] = self.tool_schemas
        timeout = self.timeout or 60
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        for _ in range(self.max_retries):
            try:
                with httpx.stream("POST", f"{url}/v1/chat/completions", json=payload, headers=headers, timeout=timeout) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or line.strip() == b"":
                            continue
                        # Heroku Inference API streams JSON objects per line
                        data = httpx.Response(200, content=line).json()
                        ai_msg = self._api_to_ai_message(data)
                        chunk = ChatGenerationChunk(message=AIMessageChunk(content=ai_msg.content))
                        if run_manager:
                            run_manager.on_llm_new_token(ai_msg.content, chunk=chunk)
                        yield chunk
                break
            except Exception as e:
                last_exc = e
        else:
            raise RuntimeError(f"Heroku Inference API stream call failed after {self.max_retries} retries: {last_exc}")

    # TODO: Implement if ChatHeroku supports async streaming. Otherwise delete.
    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:

    # TODO: Implement if ChatHeroku supports async generation. Otherwise delete.
    # async def _agenerate(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> ChatResult:
