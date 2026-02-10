import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from utils.conversation_utils import build_langchain_messages

from .config import Config
from .llm_interface import LLMInterface, Role


class EndpointLLM(LLMInterface):
    """Chat-only LLM that calls a custom POST /api/chat endpoint.

    The API manages conversation history server-side via conversation_id.
    This implementation does not support structured output and cannot be used
    as a judge. For judge operations, use Claude, OpenAI, Gemini, or Azure.
    """

    def __init__(
        self,
        name: str,
        role: Role,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, role, system_prompt)

        cfg = Config.get_endpoint_config()
        self._base_url = (base_url or cfg["base_url"]).rstrip("/")
        self._api_key = api_key or cfg["api_key"]

        if model_name and model_name.lower().startswith("endpoint-"):
            self._api_model = model_name[len("endpoint-") :].strip() or cfg["model"]
        else:
            self._api_model = cfg["model"]
        self.model_name = model_name or "endpoint"
        self.temperature = kwargs.pop("temperature", None)
        self.max_tokens = kwargs.pop("max_tokens", None)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying llm object.

        This allows accessing attributes like temperature, max_tokens, etc.
        directly on the LLM instance, which will be forwarded to the
        underlying LangChain model (self.llm).
        """
        # Check if self.llm exists by looking in __dict__ to avoid recursion
        # Only delegate if self.llm exists and has the attribute
        if "llm" in self.__dict__ and hasattr(self.llm, name):
            return getattr(self.llm, name)
        # If the attribute doesn't exist on self.llm, raise AttributeError
        return getattr(self, name, None)

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response via POST /api/chat with server-side conversation_id.

        The API does not accept a system role; the system prompt is folded into
        the first user message as \"System: ...\".
        """
        messages = build_langchain_messages(self.role, conversation_history)
        last_message = messages[-1].text

        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self._api_model,
            "messages": [
                {
                    "role": "user",
                    "content": last_message,
                },
            ],
            "stream": False,
            "conversation_id": self.conversation_id,
        }

        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    headers=headers,
                    json=body,
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(
                            f"Endpoint returned {resp.status}: {text[:500]}"
                        )
                    resp_data = await resp.json()
            end_time = time.time()

            msg_data = resp_data.get("message") or {}
            msg_text: str = msg_data.get("content", "")

            self.last_response_metadata = {
                "conversation_id": resp_data.get("conversation_id"),
                "model": resp_data.get("model", self._api_model),
                "provider": "endpoint",
                "role": self.role.value,
                "timestamp": datetime.now().isoformat(),
                "response_time_seconds": round(end_time - start_time, 3),
                "total_duration": resp_data.get("total_duration"),
                "load_duration": resp_data.get("load_duration"),
                "prompt_eval_count": resp_data.get("prompt_eval_count"),
                "prompt_eval_duration": resp_data.get("prompt_eval_duration"),
                "eval_count": resp_data.get("eval_count"),
                "eval_duration": resp_data.get("eval_duration"),
            }
            self.ensure_conversation_id()
            return msg_text
        except Exception as e:
            self.last_response_metadata = {
                "model": self._api_model,
                "provider": "endpoint",
                "role": self.role.value,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
            self.ensure_conversation_id()
            return f"Error generating response: {str(e)}"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
