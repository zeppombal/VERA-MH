from typing import Optional, Type, TypeVar

from langchain_community.llms import Ollama
from pydantic import BaseModel

from .config import Config
from .llm_interface import LLMInterface

T = TypeVar("T", bound=BaseModel)


class LlamaLLM(LLMInterface):
    """Llama implementation using LangChain with Ollama."""

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

    async def generate_response(self, message: Optional[str] = None) -> str:
        """Generate a response to the given message asynchronously."""
        try:
            # Format the message with system prompt if available
            full_message = message
            if self.system_prompt:
                full_message = (
                    f"System: {self.system_prompt}\n\nHuman: {message}\n\nAssistant:"
                )

            # Ollama doesn't have native async support in langchain-community
            # So we'll use the synchronous version
            response = self.llm.invoke(full_message)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    async def generate_structured_response(
        self, message: Optional[str], response_model: Type[T]
    ) -> T:
        """Generate a structured response using Pydantic model.

        Note: Ollama through LangChain community doesn't have native structured
        output support like Claude/OpenAI/Gemini. This implementation will attempt
        to use with_structured_output but may have limited support.

        Args:
            message: The prompt message
            response_model: Pydantic model class to structure the response

        Returns:
            Instance of the response_model with structured data

        Raises:
            NotImplementedError: If Ollama doesn't support structured output
        """
        try:
            # Try to use with_structured_output if available
            structured_llm = self.llm.with_structured_output(response_model)

            full_message = message
            if self.system_prompt:
                full_message = (
                    f"System: {self.system_prompt}\n\nHuman: {message}\n\nAssistant:"
                )

            response = await structured_llm.ainvoke(full_message)

            if not isinstance(response, response_model):
                raise ValueError(
                    f"Response is not an instance of {response_model.__name__}"
                )

            return response  # type: ignore[return-value]
        except AttributeError:
            raise NotImplementedError(
                "Ollama LLM does not support structured output through LangChain. "
                "Consider using Claude, OpenAI, or Gemini models for structured output."
            )
        except Exception as e:
            raise RuntimeError(f"Error generating structured response: {str(e)}") from e

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt
