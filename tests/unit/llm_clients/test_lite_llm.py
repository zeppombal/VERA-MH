from unittest.mock import AsyncMock, patch

import pytest

from judge.response_models import QuestionResponse
from llm_clients import JudgeLLM, LLMFactory, Role
from llm_clients.litellm_llm import LiteLLMLLM


@pytest.mark.unit
class TestLiteLLMLLM:
    def test_resolve_model_routes_bare_models_to_hosted_vllm(self):
        assert LiteLLMLLM.resolve_model("Qwen3.5-35B-A3B") == (
            "hosted_vllm/Qwen3.5-35B-A3B"
        )
        assert (
            LiteLLMLLM.resolve_model("vertex_ai/claude-sonnet-4-5@20250929")
            == "vertex_ai/claude-sonnet-4-5@20250929"
        )

    def test_factory_uses_litellm_for_provider_prefixed_models(self):
        llm = LLMFactory.create_llm(
            model_name="vertex_ai/claude-sonnet-4-5@20250929",
            name="Test LiteLLM",
            role=Role.PROVIDER,
            api_base="https://example.test/v1/models/claude",
        )

        assert isinstance(llm, LiteLLMLLM)
        assert isinstance(llm, JudgeLLM)
        assert llm.resolved_model == "vertex_ai/claude-sonnet-4-5@20250929"

    def test_factory_routes_bare_provider_model_with_api_base_to_hosted_vllm(self):
        llm = LLMFactory.create_llm(
            model_name="Qwen3.5-35B-A3B",
            name="Test Hosted vLLM",
            role=Role.PROVIDER,
            api_base="http://node:8000/v1",
        )

        assert isinstance(llm, LiteLLMLLM)
        assert llm.resolved_model == "hosted_vllm/Qwen3.5-35B-A3B"
        assert llm.api_base == "http://node:8000/v1"

    @pytest.mark.asyncio
    async def test_generate_response_forwards_litellm_kwargs_and_strips_thinking(self):
        mock_response = {
            "id": "chatcmpl-1",
            "model": "served-model",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "choices": [
                {
                    "message": {
                        "content": "<think>private reasoning</think>\nVisible answer"
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        with patch(
            "llm_clients.litellm_llm.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_completion:
            llm = LiteLLMLLM(
                name="Provider",
                role=Role.PROVIDER,
                model_name="local-model",
                api_base="http://node:8000/v1",
                max_completion_tokens=123,
                reasoning_effort="low",
                temperature=0,
            )
            response = await llm.generate_response(
                [{"turn": 0, "response": "Start the conversation"}]
            )

        assert response == "Visible answer"
        assert llm.max_tokens == 123
        assert llm.temperature == 0
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "hosted_vllm/local-model"
        assert kwargs["api_base"] == "http://node:8000/v1"
        assert kwargs["max_tokens"] == 123
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["temperature"] == 0
        assert kwargs["messages"] == [
            {"role": "user", "content": "Start the conversation"}
        ]

        metadata = llm.last_response_metadata
        assert metadata["provider"] == "litellm"
        assert metadata["model"] == "served-model"
        assert metadata["reasoning_content"] == "private reasoning"
        assert metadata["usage"]["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_generate_response_with_empty_conversation_history(self):
        mock_response = {
            "id": "chatcmpl-empty",
            "model": "served-model",
            "choices": [
                {
                    "message": {"content": "Opening response"},
                    "finish_reason": "stop",
                }
            ],
        }

        with patch(
            "llm_clients.litellm_llm.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_completion:
            llm = LiteLLMLLM(
                name="Persona",
                role=Role.PERSONA,
                model_name="local-model",
                system_prompt="You are a persona.",
                start_prompt="Begin as the persona.",
            )
            response = await llm.generate_response([])

        assert response == "Opening response"
        assert mock_completion.call_args.kwargs["messages"] == [
            {"role": "system", "content": "You are a persona."},
            {"role": "user", "content": "Begin as the persona."},
        ]

    @pytest.mark.asyncio
    async def test_generate_structured_response_parses_pydantic_model(self):
        mock_response = {
            "id": "judge-1",
            "model": "judge-model",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answer": "Yes", '
                            '"reasoning": "The transcript shows the behavior."}'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        with patch(
            "llm_clients.litellm_llm.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_completion:
            llm = LiteLLMLLM(
                name="Judge",
                role=Role.JUDGE,
                model_name="vertex_ai/claude-sonnet-4-5@20250929",
                system_prompt="Judge this transcript.",
                api_base="https://example.test/v1/models/claude",
            )
            response = await llm.generate_structured_response(
                "Does the chatbot detect risk?", QuestionResponse
            )

        assert response == QuestionResponse(
            answer="Yes", reasoning="The transcript shows the behavior."
        )
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "vertex_ai/claude-sonnet-4-5@20250929"
        assert kwargs["api_base"] == "https://example.test/v1/models/claude"
        assert kwargs["messages"][0] == {
            "role": "system",
            "content": "Judge this transcript.",
        }
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["response_format"]["json_schema"]["name"] == "QuestionResponse"

        metadata = llm.last_response_metadata
        assert metadata["structured_output"] is True
        assert metadata["provider"] == "litellm"

    @pytest.mark.asyncio
    async def test_generate_structured_response_unwraps_placeholder_object(self):
        mock_response = {
            "id": "judge-placeholder",
            "model": "judge-model",
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"$PARAMETER_NAME": {'
                            '"answer": "No", '
                            '"reasoning": "The transcript does not show it."'
                            "}}"
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        with patch(
            "llm_clients.litellm_llm.litellm.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            llm = LiteLLMLLM(
                name="Judge",
                role=Role.JUDGE,
                model_name="vertex_ai/claude-sonnet-4-5@20250929",
            )
            response = await llm.generate_structured_response(
                "Does the chatbot detect risk?", QuestionResponse
            )

        assert response == QuestionResponse(
            answer="No", reasoning="The transcript does not show it."
        )

    def test_post_process_response_strips_thinking(self):
        llm = LiteLLMLLM(name="Provider", role=Role.PROVIDER, model_name="local-model")

        response = llm._post_process_response("<think>hidden</think>\nShown")

        assert response == "Shown"
        assert llm.last_response_metadata["reasoning_content"] == "hidden"
