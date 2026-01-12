from typing import Any, Dict, List, Optional

from langchain_community.llms import Ollama

from .config import Config
from .llm_interface import LLMInterface


class LlamaLLM(LLMInterface):
    """Llama implementation using LangChain with Ollama.

    Note: This implementation does not support structured output generation
    and therefore cannot be used as a judge. For judge operations, use
    Claude, OpenAI, or Gemini models.
    """

    def __init__(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, system_prompt)

        # Use provided model name or fall back to config default
        self.model_name = model_name or Config.get_llama_config()["model"]

        # Get default config and allow kwargs to override
        config = Config.get_llama_config()
        llm_params = {
            "model": self.model_name,
            "temperature": config.get("temperature", 0.7),
            "base_url": config.get(
                "base_url", "http://localhost:11434"
            ),  # Default Ollama URL
        }

        # Override with any provided kwargs
        llm_params.update(kwargs)
        self.llm = Ollama(**llm_params)

    async def generate_response(
        self,
        message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate a response to the given message asynchronously.

        Args:
            message: The current message to respond to
            conversation_history: Optional list of previous conversation turns
        """
        try:
            # Build full message with system prompt and history
            full_message = ""

            if self.system_prompt:
                full_message = f"System: {self.system_prompt}\n\n"

            # Add conversation history if provided
            if conversation_history:
                for turn in conversation_history:
                    speaker = turn.get("speaker")
                    text = turn.get("response")
                    if speaker == "persona":
                        full_message += f"Human: {text}\n\n"
                    elif speaker in ["chatbot", "agent"]:
                        full_message += f"Assistant: {text}\n\n"

            # Add current message
            if message:
                full_message += f"Human: {message}\n\nAssistant:"

            # Ollama doesn't have native async support in langchain-community
            # So we'll use the synchronous version
            response = self.llm.invoke(full_message)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
