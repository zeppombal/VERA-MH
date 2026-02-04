"""Shared pytest fixtures for LLM client tests.

This module provides reusable pytest fixtures that reduce code duplication across
LLM client test files. These fixtures handle common setup patterns like:

- Mock response creation with provider-specific metadata
- Mock LLM instances with configured ainvoke behavior
- API key patching for different providers (Claude, OpenAI, Gemini, Azure)
- Standard test data (conversation histories, messages, LLM kwargs)
"""

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_clients import Role

# ============================================================================
# Mock Response Factories
# ============================================================================


@pytest.fixture
def mock_response_factory():
    """Factory fixture for creating mock LLM responses with metadata.

    Returns a function that creates configured mock responses for different providers.

    Usage:
        mock_resp = mock_response_factory(
            text="Response text",
            response_id="msg_123",
            provider="claude",
            metadata={"usage": {"input_tokens": 10}}
        )
    """

    def _create_mock_response(
        text: str = "Test response",
        response_id: Optional[str] = "test_id",
        provider: str = "claude",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MagicMock:
        """Create a mock response object configured for a specific provider.

        Args:
            text: Response text content
            response_id: Response ID (can be None)
            provider: Provider name ("claude", "openai", "gemini", "ollama", "azure")
            metadata: Provider-specific metadata dict

        Returns:
            Configured MagicMock response object
        """
        mock_response = MagicMock()
        mock_response.text = text
        mock_response.id = response_id

        if metadata is None:
            metadata = {}

        # Configure provider-specific metadata structure
        if provider == "claude":
            mock_response.response_metadata = {
                "model": metadata.get("model", "claude-sonnet-4-5-20250929"),
                **metadata,
            }
        elif provider == "openai":
            mock_response.response_metadata = {
                "model_name": metadata.get("model_name", "gpt-4"),
                **metadata,
            }
            mock_response.additional_kwargs = metadata.get("additional_kwargs", {})
            if "usage_metadata" in metadata:
                mock_response.usage_metadata = metadata["usage_metadata"]
        elif provider == "gemini":
            # Gemini has special metadata object with model_name attribute
            mock_metadata_obj = MagicMock()
            mock_metadata_obj.model_name = metadata.get("model_name", "gemini-1.5-pro")

            # Build complete metadata dict including all custom fields
            metadata_dict = {"model_name": mock_metadata_obj.model_name}
            # Add all other metadata fields
            for key, value in metadata.items():
                if key != "model_name":
                    metadata_dict[key] = value

            # Add dictionary access for usage_metadata and other fields
            mock_metadata_obj.__getitem__ = lambda self, key: metadata_dict.get(key)
            mock_metadata_obj.__contains__ = lambda self, key: key in metadata_dict
            mock_metadata_obj.get = lambda key, default=None: metadata_dict.get(
                key, default
            )

            mock_response.response_metadata = mock_metadata_obj
        elif provider == "azure":
            mock_response.response_metadata = {
                "model_name": metadata.get("model_name", "gpt-4"),
                **metadata,
            }
            mock_response.additional_kwargs = metadata.get("additional_kwargs", {})
            if "usage_metadata" in metadata:
                mock_response.usage_metadata = metadata["usage_metadata"]
        elif provider == "ollama":
            # Ollama responses are simpler - just text strings
            mock_response.response_metadata = metadata
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        return mock_response

    return _create_mock_response


@pytest.fixture
def mock_llm_factory():
    """Factory fixture for creating mock LLM instances.

    Returns a function that creates configured mock LLM instances
    with AsyncMock ainvoke method.

    Usage:
        mock_llm = mock_llm_factory(
            response="Test response",
            model="claude-sonnet-4-5-20250929"
        )
    """

    def _create_mock_llm(
        response: Any = "Test response",
        model: Optional[str] = None,
        side_effect: Optional[Exception] = None,
    ) -> MagicMock:
        """Create a mock LLM instance.

        Args:
            response: Response to return from ainvoke (can be string or mock object)
            model: Model name to set on mock (optional)
            side_effect: Exception to raise from ainvoke (optional)

        Returns:
            Configured MagicMock LLM instance
        """
        mock_llm = MagicMock()

        if model:
            mock_llm.model = model

        if side_effect:
            mock_llm.ainvoke = AsyncMock(side_effect=side_effect)
        else:
            mock_llm.ainvoke = AsyncMock(return_value=response)

        return mock_llm

    return _create_mock_llm


# ============================================================================
# Conversation History Fixtures
# ============================================================================


@pytest.fixture
def sample_conversation_history():
    """Reusable multi-turn conversation history for testing.

    Returns a 3-turn conversation suitable for testing both:
    - Standard conversation history handling
    - Persona role message type flipping

    The conversation alternates between PERSONA and PROVIDER speakers,
    which allows testing of role-based message transformations.
    """
    return [
        {
            "turn": 1,
            "speaker": Role.PERSONA,
            "input": "Start",
            "response": "Hello",
            "early_termination": False,
            "logging": {},
        },
        {
            "turn": 2,
            "speaker": Role.PROVIDER,
            "input": "Hello",
            "response": "Hi there",
            "early_termination": False,
            "logging": {},
        },
        {
            "turn": 3,
            "speaker": Role.PERSONA,
            "input": "Hi there",
            "response": "How are you?",
            "early_termination": False,
            "logging": {},
        },
    ]


# ============================================================================
# Provider-Specific API Key Patches
# ============================================================================


def _patch_api_credentials(
    monkeypatch, env_vars: Dict[str, str], config_attrs: Dict[str, str]
):
    """Helper to patch API credentials in both environment and Config class.

    Args:
        monkeypatch: Pytest monkeypatch fixture
        env_vars: Dict of {ENV_VAR_NAME: value} to set
        config_attrs: Dict of {Config.ATTR_NAME: value} to set
    """
    from llm_clients.config import Config

    for env_var, value in env_vars.items():
        monkeypatch.setenv(env_var, value)

    for attr_name, value in config_attrs.items():
        monkeypatch.setattr(Config, attr_name, value)


@pytest.fixture
def mock_anthropic_api_key(monkeypatch):
    """Patch Anthropic API key for Claude tests."""
    _patch_api_credentials(
        monkeypatch,
        env_vars={"ANTHROPIC_API_KEY": "test-anthropic-key"},
        config_attrs={"ANTHROPIC_API_KEY": "test-anthropic-key"},
    )


@pytest.fixture
def mock_openai_api_key(monkeypatch):
    """Patch OpenAI API key for OpenAI tests."""
    _patch_api_credentials(
        monkeypatch,
        env_vars={"OPENAI_API_KEY": "test-openai-key"},
        config_attrs={"OPENAI_API_KEY": "test-openai-key"},
    )


@pytest.fixture
def mock_google_api_key(monkeypatch):
    """Patch Google API key for Gemini tests."""
    _patch_api_credentials(
        monkeypatch,
        env_vars={"GOOGLE_API_KEY": "test-google-key"},
        config_attrs={"GOOGLE_API_KEY": "test-google-key"},
    )


@pytest.fixture
def mock_azure_credentials(monkeypatch):
    """Patch Azure credentials for Azure tests."""
    _patch_api_credentials(
        monkeypatch,
        env_vars={
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-azure-key",
        },
        config_attrs={
            "AZURE_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_API_KEY": "test-azure-key",
        },
    )


# ============================================================================
# Common Test Data
# ============================================================================


@pytest.fixture
def default_llm_kwargs():
    """Default kwargs for LLM initialization tests."""
    return {
        "temperature": 0.5,
        "max_tokens": 500,
        "top_p": 0.9,
    }
