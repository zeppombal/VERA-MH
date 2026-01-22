import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_ollama import OllamaLLM as LangChainOllamaLLM

from utils.conversation_utils import format_conversation_as_string

from .config import Config
from .llm_interface import LLMInterface


class OllamaLLM(LLMInterface):
    """General Ollama implementation using LangChain.

    This implementation can work with any model hosted on Ollama, not just
    Llama models. Examples: phi4, mistral, codellama, etc.

    Note: This implementation does not support structured output generation
    and therefore cannot be used as a judge. For judge operations, use
    Claude, OpenAI, or Gemini models.
    """

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, system_prompt)

        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_ollama_config()["model"]

        # Get default config and allow kwargs to override
        llm_params = {
            "model": self.model_name,
            "base_url": base_url
            or Config.get_ollama_config().get("base_url", "http://localhost:11434"),
        }

        # Store max_tokens if provided (for logging), but map to num_predict for Ollama
        # Ollama uses 'num_predict' instead of 'max_tokens'
        if "max_tokens" in kwargs:
            llm_params["num_predict"] = kwargs.pop("max_tokens")
            self.max_tokens = llm_params["num_predict"]
        elif "num_predict" in kwargs:
            self.max_tokens = kwargs.get("num_predict")
        else:
            self.max_tokens = None

        # Store temperature for logging
        self.temperature = kwargs.get("temperature", None)

        # Override with any remaining provided kwargs
        llm_params.update(kwargs)
        self.llm = LangChainOllamaLLM(**llm_params)

        # If max_tokens wasn't set, try to get it from the llm object
        if self.max_tokens is None:
            self.max_tokens = getattr(self.llm, "num_predict", None)
        # If temperature wasn't set, try to get it from the llm object
        if self.temperature is None:
            self.temperature = getattr(self.llm, "temperature", None)

        # Store metadata from last response
        self.last_response_metadata: Dict[str, Any] = {}

    async def generate_response(
        self,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response based on conversation history.

        Args:
            conversation_history: Optional list of previous conversation turns
        """
        try:
            # Build full message using utility function
            full_message = format_conversation_as_string(
                conversation_history=conversation_history,
                system_prompt=self.system_prompt,
            )

            start_time = time.time()
            # Use ainvoke for async call - BaseLLM.ainvoke returns a string directly
            response_text = await self.llm.ainvoke(full_message)
            end_time = time.time()

            # Extract metadata from response
            # Note: Ollama's BaseLLM.ainvoke returns a string, not an object
            # with metadata. For simplicity, we'll just track basic response info
            self.last_response_metadata = {
                "response_id": None,
                "model": self.model_name,
                "provider": "ollama",
                "timestamp": datetime.now().isoformat(),
                "response_time_seconds": round(end_time - start_time, 3),
                "usage": {},
            }

            return response_text
        except Exception as e:
            # Store error metadata
            self.last_response_metadata = {
                "response_id": None,
                "model": self.model_name,
                "provider": "ollama",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "usage": {},
            }
            return f"Error generating response: {str(e)}"

    def get_last_response_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last response."""
        return self.last_response_metadata.copy()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
