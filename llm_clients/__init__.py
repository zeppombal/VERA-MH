"""
LLM Clients Package - Shared LLM abstraction layer
Provides unified interface for different LLM providers
  - OpenAI (gpt-*)
  - Claude (claude-*)
  - Gemini (gemini-*)
  - Azure (azure-*)
  - Ollama (ollama-*)
  - Custom endpoint (endpoint, endpoint-*)
  - LiteLLM provider-prefixed models (vertex_ai/*, hosted_vllm/*, etc.)
"""

from .config import Config
from .litellm_llm import LiteLLMLLM
from .llm_factory import LLMFactory
from .llm_interface import JudgeLLM, LLMGenerationFailed, LLMInterface, Role

__all__ = [
    "LLMInterface",
    "JudgeLLM",
    "LiteLLMLLM",
    "LLMFactory",
    "Config",
    "Role",
    "LLMGenerationFailed",
]
