from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.litellm import LiteLLMModel

from ..conftest import TestEnv


class VanillaResponse(BaseModel):
    original_text: str
    corrected_text: str
    metadata: dict[str, Any] = {}


def test_api_key_arg(env: TestEnv):
    env.set("LITELLM_API_KEY", "via-env-var")
    env.set("LITELLM_BASE_URL", "default-base-url")
    m = LiteLLMModel("openai/gpt-4o-ga", api_key="via-arg", base_url="via-arg-base-url")
    assert m.configs["api_key"] == "via-arg"
    assert m.configs["base_url"] == "via-arg-base-url"


def test_api_key_env_var(env: TestEnv):
    env.set("LITELLM_API_KEY", "via-env-var")
    env.set("LITELLM_BASE_URL", "default-base-url")
    m = LiteLLMModel("openai/gpt-4o-ga")
    assert m.configs["api_key"] == "via-env-var"
    assert m.configs["base_url"] == "default-base-url"


def test_api_key_not_set(env: TestEnv):
    env.remove("LITELLM_API_KEY")
    with pytest.raises(UserError, match="API key must be provided or set in the LITELLM_API_KEY environment variable"):
        LiteLLMModel("openai/gpt-4o-ga")


def test_base_url_not_set(env: TestEnv):
    env.set("LITELLM_API_KEY", "via-env-var")
    env.remove("LITELLM_BASE_URL")
    with pytest.raises(
        UserError, match="Base URL must be provided or set in the LITELLM_BASE_URL environment variable"
    ):
        LiteLLMModel("openai/gpt-4o-ga")


def test_agent_simple(env: TestEnv):
    with override_allow_model_requests(True):
        env.set("LITELLM_API_KEY", "via-env-var")
        env.set("LITELLM_BASE_URL", "default-base-url")
        support_agent = Agent(
            "litellm:openai/gpt-4o-ga",
            result_type=VanillaResponse,
            system_prompt="You are a excellent English teacher. "
            "Now you have to correct the following text of your student.",
        )
        result = support_agent.run_sync("Here is your student's text: 'I are a excellent English teacher.'")
        assert isinstance(result.data, VanillaResponse)
        print(result.data)


@pytest.mark.asyncio
async def test_agent_async(env: TestEnv):
    with override_allow_model_requests(True):
        env.set("LITELLM_API_KEY", "via-env-var")
        env.set("LITELLM_BASE_URL", "default-base-url")
        support_agent = Agent(
            "litellm:openai/gpt-4o-ga",
            result_type=VanillaResponse,
            system_prompt="You are a excellent English teacher. "
            "Now you have to correct the following text of your student.",
        )
        result = await support_agent.run("Here is your student's text: 'I are a excellent English teacher.'")
        assert isinstance(result.data, VanillaResponse)
        print(result.data)


@pytest.mark.asyncio
async def test_agent_streaming(env: TestEnv):
    with override_allow_model_requests(True):
        env.set("LITELLM_API_KEY", "via-env-var")
        env.set("LITELLM_BASE_URL", "default-base-url")
        support_agent = Agent(
            "litellm:openai/gpt-4o-ga",
            result_type=VanillaResponse,
            system_prompt="You are a excellent English teacher. "
            "Now you have to correct the following text of your student.",
        )

        user_text = "Here is your student's text: 'I are a excellent English teacher.'"
        chunk_count = 0
        async with support_agent.run_stream(user_text) as result:
            async for message in result.stream():
                assert isinstance(message, VanillaResponse)
                print(message)
                chunk_count += 1

        print(chunk_count)
        assert chunk_count > 1


@pytest.mark.asyncio
async def test_agent_stream_text(env: TestEnv):
    with override_allow_model_requests(True):
        env.set("LITELLM_API_KEY", "via-env-var")
        env.set("LITELLM_BASE_URL", "default-base-url")
        support_agent = Agent(
            "litellm:openai/gpt-4o-ga",
            result_type=str,
            system_prompt="You are a excellent English teacher. "
            "Now you have to correct the following text of your student.",
        )

        user_text = "Here is your student's text: 'I are a excellent English teacher.'"
        async with support_agent.run_stream(user_text) as result:
            async for message in result.stream_text():
                assert isinstance(message, str)
                print(message)


@pytest.mark.asyncio
async def test_agent_stream_structured(env: TestEnv):
    with override_allow_model_requests(True):
        env.set("LITELLM_API_KEY", "via-env-var")
        env.set("LITELLM_BASE_URL", "default-base-url")
        support_agent = Agent(
            "litellm:openai/gpt-4o-ga",
            result_type=VanillaResponse,
            system_prompt="You are a excellent English teacher. "
            "Now you have to correct the following text of your student.",
        )

        user_text = "Here is your student's text: 'I are a excellent English teacher.'"
        async with support_agent.run_stream(user_text) as result:
            async for message, last in result.stream_structured(debounce_by=0.01):
                try:
                    response = await result.validate_structured_result(message, allow_partial=not last)
                except ValidationError as exc:
                    if all(e["type"] == "missing" for e in exc.errors()):
                        continue
                    else:
                        raise

                assert isinstance(response, VanillaResponse)
                print(response)
