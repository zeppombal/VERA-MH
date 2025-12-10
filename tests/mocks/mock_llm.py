from typing import Any, Dict, Optional

from llm_clients.llm_interface import LLMInterface


class MockLLM(LLMInterface):
    """Mock LLM for testing without API calls."""

    def __init__(
        self,
        name: str = "mock-llm",
        responses: list[str] | None = None,
        system_prompt: Optional[str] = None,
        simulate_error: bool = False,
        model_name: str = "mock-model",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        super().__init__(name, system_prompt)
        self.responses = responses or ["Mock response"]
        self.response_index = 0
        self.calls: list[str] = []
        self.simulate_error = simulate_error
        self.last_response_metadata: Dict[str, Any] = {}
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_response(self, message: Optional[str] = None) -> str:
        """Return predetermined responses in sequence.

        Returns:
            Response text string
        """
        self.calls.append(message)

        if self.simulate_error:
            self.last_response_metadata = {
                "provider": "mock",
                "model": self.name,
                "error": "Simulated API error",
            }
            raise Exception("Simulated API error")

        if self.response_index >= len(self.responses):
            response = f"Mock response {self.response_index + 1}"
        else:
            response = self.responses[self.response_index]
            self.response_index += 1

        self.last_response_metadata = {
            "provider": "mock",
            "model": self.name,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

        return response

    def get_last_response_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last response."""
        return self.last_response_metadata.copy()

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set or update the system prompt."""
        self.system_prompt = system_prompt

    def reset(self) -> None:
        """Reset for reuse in multiple tests."""
        self.response_index = 0
        self.calls = []
        self.last_response_metadata = {}
