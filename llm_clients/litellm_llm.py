import json
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import litellm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from utils.conversation_utils import build_langchain_messages
from utils.debug import debug_print

from .llm_interface import JudgeLLM, Role

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class LiteLLMLLM(JudgeLLM):
    """LiteLLM-backed client for hosted and provider-prefixed model strings.

    This is the adapter used by the eval harness. It keeps VERA-MH's simulation
    and judging logic unchanged while letting callers use the same model strings
    and API bases used by other benchmarks, such as ``hosted_vllm/<model>`` and
    ``vertex_ai/claude-sonnet-4-5@20250929``.
    """

    LITELLM_PROVIDER_PREFIXES = (
        "vertex_ai/",
        "anthropic/",
        "openai/",
        "openrouter/",
        "gemini/",
        "azure/",
        "bedrock/",
        "hosted_vllm/",
        "ollama/",
        "fireworks_ai/",
        "together_ai/",
        "groq/",
        "deepseek/",
        "cohere/",
        "mistral/",
    )

    def _no_retry_substrings(self) -> tuple[str, ...]:
        return (
            "insufficient_quota",
            "billing_hard_limit",
            "invalid_api_key",
            "authentication_error",
            "permission_denied",
            "credit balance is too low",
        )

    @classmethod
    def should_handle_model(cls, model_name: str) -> bool:
        """Return True for model strings intended for LiteLLM routing."""
        model_lower = model_name.lower()
        return model_lower.startswith(cls.LITELLM_PROVIDER_PREFIXES)

    @classmethod
    def resolve_model(cls, model_name: str) -> str:
        """Resolve bare model names to the hosted vLLM provider namespace."""
        if model_name.lower().startswith(cls.LITELLM_PROVIDER_PREFIXES):
            return model_name
        return f"hosted_vllm/{model_name}"

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        first_message = kwargs.pop("first_message", None)
        start_prompt = kwargs.pop("start_prompt", None)
        max_llm_retries = kwargs.pop("max_llm_retries", 3)
        retry_base_delay_seconds = kwargs.pop("retry_base_delay_seconds", 0.75)
        retry_max_delay_seconds = kwargs.pop("retry_max_delay_seconds", 8.0)
        super().__init__(
            name,
            role,
            system_prompt,
            first_message=first_message,
            start_prompt=start_prompt,
            max_llm_retries=max_llm_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_max_delay_seconds=retry_max_delay_seconds,
        )

        if not model_name:
            raise ValueError("LiteLLMLLM requires model_name")

        self.model_name = model_name
        self.resolved_model = self.resolve_model(model_name)
        self.api_base = kwargs.pop("api_base", None)
        self.api_key = kwargs.pop("api_key", None)
        self.timeout = kwargs.pop("timeout", kwargs.pop("timeout_seconds", None))
        self.model_params = kwargs

        # The eval harness historically passes max_completion_tokens. LiteLLM and
        # OpenAI-compatible servers consistently accept max_tokens.
        if (
            "max_tokens" not in self.model_params
            and "max_completion_tokens" in self.model_params
        ):
            self.model_params["max_tokens"] = self.model_params.pop(
                "max_completion_tokens"
            )
        self.temperature = self.model_params.get("temperature")
        self.max_tokens = self.model_params.get("max_tokens")

        print("Creating LiteLLM LLM with parameters:")
        print(f"  Model: {self.resolved_model}")
        print(f"  API base: {self.api_base or 'provider default'}")
        print(f"  Temperature: {self.model_params.get('temperature', 'default')}")
        print(f"  Max tokens: {self.model_params.get('max_tokens', 'default')}")

    async def start_conversation(self) -> str:
        if self.first_message is not None:
            self._set_response_metadata("litellm", static_first_message=True)
            return self.first_message
        return await self.generate_response(self.get_initial_prompt_turns())

    def _message_to_dict(self, message: BaseMessage) -> dict[str, str]:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            msg_type = getattr(message, "type", "")
            if msg_type == "system":
                role = "system"
            elif msg_type == "ai":
                role = "assistant"
            else:
                role = "user"
        return {"role": role, "content": message.text}

    def _build_messages(
        self, conversation_history: Optional[List[Dict[str, Any]]]
    ) -> list[dict[str, str]]:
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.extend(build_langchain_messages(self.role, conversation_history))
        return [self._message_to_dict(message) for message in messages]

    def _completion_kwargs(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.resolved_model,
            "messages": messages,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        kwargs.update(self.model_params)
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    @staticmethod
    def _response_to_dict(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump()
        raise TypeError(f"Unsupported LiteLLM response type: {type(response)!r}")

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks).strip()
        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if isinstance(text, str):
                return text
        return str(content)

    @classmethod
    def _extract_text(cls, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return cls._normalize_content(message.get("content"))

        output_items = payload.get("output")
        if isinstance(output_items, list):
            chunks = []
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                content_items = item.get("content")
                if not isinstance(content_items, list):
                    continue
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()

        raise RuntimeError("LiteLLM response missing choices/output text")

    @staticmethod
    def split_thinking(text: str) -> tuple[str, str]:
        """Return response text and reasoning content."""
        match = _THINK_RE.search(text)
        if match is not None:
            return text[match.end() :].strip(), match.group(1).strip()
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            return parts[1].strip(), parts[0].replace("<think>", "").strip()
        return text.strip(), ""

    def _record_metadata(
        self,
        payload: dict[str, Any],
        response_time_seconds: float,
        *,
        structured_output: bool = False,
    ) -> None:
        choices = payload.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        finish_reason = (
            first_choice.get("finish_reason")
            if isinstance(first_choice, dict)
            else None
        )
        self._set_response_metadata(
            "litellm",
            response_id=payload.get("id"),
            model=payload.get("model", self.resolved_model),
            usage=payload.get("usage") or {},
            response_time_seconds=response_time_seconds,
            finish_reason=finish_reason,
            structured_output=structured_output,
        )

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not conversation_history or len(conversation_history) == 0:
            return await self.start_conversation()

        messages = self._build_messages(conversation_history)

        debug_print(f"\n[DEBUG {self.name} - {self.role.value}] LiteLLM messages:")
        for i, message in enumerate(messages):
            preview = message["content"][:100]
            suffix = "..." if len(message["content"]) > 100 else ""
            debug_print(f"  {i + 1}. {message['role']}: {preview}{suffix}")

        async def _invoke() -> str:
            start_time = time.time()
            response = await litellm.acompletion(**self._completion_kwargs(messages))
            response_time_seconds = round(time.time() - start_time, 3)
            payload = self._response_to_dict(response)
            self._record_metadata(payload, response_time_seconds)
            text = self._extract_text(payload)
            response_text, reasoning_content = self.split_thinking(text)
            if reasoning_content:
                self._last_response_metadata["reasoning_content"] = reasoning_content
            return response_text

        return await self._run_with_retry(_invoke, provider="litellm")

    @staticmethod
    def _response_format(response_model: Type[T]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        }

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match is None:
                raise
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Structured LiteLLM response was not a JSON object")
        return parsed

    @staticmethod
    def _unwrap_structured_payload(
        parsed: dict[str, Any], response_model: Type[T]
    ) -> dict[str, Any]:
        model_fields = set(response_model.model_fields)
        if model_fields.issubset(parsed):
            return parsed

        if len(parsed) == 1:
            nested = next(iter(parsed.values()))
            if isinstance(nested, dict) and model_fields.issubset(nested):
                return nested

        return parsed

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": message or ""})

        async def _invoke() -> T:
            start_time = time.time()
            response = await litellm.acompletion(
                **self._completion_kwargs(
                    messages,
                    response_format=self._response_format(response_model),
                )
            )
            response_time_seconds = round(time.time() - start_time, 3)
            payload = self._response_to_dict(response)
            self._record_metadata(
                payload, response_time_seconds, structured_output=True
            )
            text = self._extract_text(payload)
            parsed = self._parse_json_object(text)
            parsed = self._unwrap_structured_payload(parsed, response_model)
            return response_model.model_validate(parsed)

        return await self._run_with_retry(_invoke, provider="litellm")

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def _post_process_response(self, response: str) -> str:
        response_text, reasoning_content = self.split_thinking(response)
        if reasoning_content:
            self._last_response_metadata["reasoning_content"] = reasoning_content
        return response_text
