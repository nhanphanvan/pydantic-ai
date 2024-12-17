import os
from typing import Any, overload, Literal, assert_never
from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import litellm

from .. import result, UnexpectedModelBehavior, _utils
from ..result import Cost
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..messages import (
    ArgsJson,
    Message,
    ModelResponse,
    ModelResponsePart,
    RetryPrompt,
    SystemPrompt,
    TextPart,
    ToolCallPart,
    ToolReturn,
    UserPrompt,
)
from . import (
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamStructuredResponse,
    StreamTextResponse,
    check_allow_model_requests,
)

LiteLLMModelName = str


@dataclass(init=False)
class LiteLLMModel(Model):
    """
    A model that uses LiteLLM.

    Internally, this uses the [LiteLLM Python](https://github.com/BerriAI/litellm) to interact with the API.

    [LiteLLM SDK Documentation](https://docs.litellm.ai/docs/)
    """

    model_name: LiteLLMModelName

    def __init__(
        self,
        model_name: LiteLLMModelName,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the LiteLLM model.

        Args:
            model_name: The name of the LiteLLM model.
            base_url: The base URL of the LiteLLM API.
            api_key: The API key for the LiteLLM API.
        """
        self.model_name = model_name
        self.configs = {
            'base_url': os.getenv("LITELLM_BASE_URL") if base_url is None else base_url,
            'api_key': os.getenv("LITELLM_API_KEY") if api_key is None else api_key,
        }

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """ Create an agent model for the LiteLLM model. """
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return LiteLLMAgentModel(
            model_name=self.model_name,
            configs=self.configs,
            allow_text_result=allow_text_result,
            tools=tools,
        )

    def name(self) -> str:
        return f"litellm:{self.model_name}"

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> dict[str, Any]:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.parameters_json_schema,
            },
        }


@dataclass
class LiteLLMAgentModel(AgentModel):
    """Implementation of `AgentModel` for LiteLLM."""

    model_name: LiteLLMModelName
    configs: dict[str, Any]
    allow_text_result: bool
    tools: list[dict[str, Any]]

    async def request(
        self, messages: list[Message], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, result.Cost]:
        response = await self._completions_create(messages, False, model_settings)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[Message], model_settings: ModelSettings | None
    ) -> AsyncIterator[EitherStreamedResponse]:
        response = await self._completions_create(messages, True, model_settings)
        yield await self._process_streamed_response(response)

    @overload
    async def _completions_create(
        self, messages: list[Message], stream: Literal[False], model_settings: ModelSettings | None
    ) -> litellm.ModelResponse:
        pass

    @overload
    async def _completions_create(
        self, messages: list[Message], stream: Literal[True], model_settings: ModelSettings | None
    ) -> litellm.CustomStreamWrapper:
        pass

    async def _completions_create(
        self, messages: list[Message], stream: bool, model_settings: ModelSettings | None
    ) -> litellm.ModelResponse | litellm.CustomStreamWrapper:
        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        openai_messages = [self._map_message(m) for m in messages]

        model_settings = model_settings or {}

        return await litellm.acompletion(
            model=self.model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=True if self.tools else None,
            tools=self.tools or None,
            tool_choice=tool_choice or None,
            stream=stream,
            stream_options={'include_usage': True} if stream else None,
            max_tokens=model_settings.get('max_tokens', None),
            temperature=model_settings.get('temperature', None),
            top_p=model_settings.get('top_p', None),
            timeout=model_settings.get('timeout', None),
            **self.configs,
        )

    @staticmethod
    def _map_message(message: Message) -> dict[str, Any]:
        """Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
        if isinstance(message, SystemPrompt):
            return dict(role='system', content=message.content)
        elif isinstance(message, UserPrompt):
            return dict(role='user', content=message.content)
        elif isinstance(message, ToolReturn):
            return dict(
                role='tool',
                tool_call_id=message.tool_call_id,
                content=message.model_response_str(),
            )
        elif isinstance(message, RetryPrompt):
            if message.tool_name is None:
                return dict(role='user', content=message.model_response())
            else:
                return dict(
                    role='tool',
                    tool_call_id=message.tool_call_id,
                    content=message.model_response(),
                )
        elif isinstance(message, ModelResponse):
            texts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for item in message.parts:
                if isinstance(item, TextPart):
                    texts.append(item.content)
                elif isinstance(item, ToolCallPart):
                    tool_calls.append(_map_tool_call(item))
                else:
                    assert_never(item)
            message_param = dict(role='assistant')
            if texts:
                # Note: model responses from this model should only have one text item, so the following
                # shouldn't merge multiple texts into one unless you switch models between runs:
                message_param['content'] = '\n\n'.join(texts)
            if tool_calls:
                message_param['tool_calls'] = tool_calls
            return message_param
        else:
            assert_never(message)

    @staticmethod
    def _process_response(response: litellm.ModelResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        if choice.message.content is not None:
            items.append(TextPart(choice.message.content))
        if choice.message.tool_calls is not None:
            for c in choice.message.tool_calls:
                items.append(ToolCallPart.from_json(c.function.name, c.function.arguments, c.id))
        return ModelResponse(items, timestamp=timestamp)

    @staticmethod
    async def _process_streamed_response(response: litellm.CustomStreamWrapper) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        timestamp: datetime | None = None
        start_cost = Cost()
        # the first chunk may contain enough information so we iterate until we get either `tool_calls` or `content`
        while True:
            try:
                chunk = await response.__anext__()
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e

            timestamp = timestamp or datetime.fromtimestamp(chunk.created, tz=timezone.utc)
            start_cost += _map_cost(chunk)

            if chunk.choices:
                delta = chunk.choices[0].delta

                if delta.content is not None:
                    return LiteLLMStreamTextResponse(delta.content, response, timestamp, start_cost)
                elif delta.tool_calls is not None:
                    return LiteLLMStreamStructuredResponse(
                        response,
                        {c.index: c for c in delta.tool_calls},
                        timestamp,
                        start_cost,
                    )
                # else continue until we get either delta.content or delta.tool_calls


@dataclass
class LiteLLMStreamTextResponse(StreamTextResponse):
    """Implementation of `StreamTextResponse` for LiteLLM models."""

    _first: str | None
    _response: litellm.CustomStreamWrapper
    _timestamp: datetime
    _cost: result.Cost
    _buffer: list[str] = field(default_factory=list, init=False)

    async def __anext__(self) -> None:
        if self._first is not None:
            self._buffer.append(self._first)
            self._first = None
            return None

        chunk = await self._response.__anext__()
        self._cost += _map_cost(chunk)
        try:
            choice = chunk.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        # we don't raise StopAsyncIteration on the last chunk because usage comes after this
        if choice.finish_reason is None:
            assert (
                choice.delta.content is not None or chunk.get("usage") is not None
            ), f"Expected delta with content, invalid chunk: {chunk!r}"
        if choice.delta.content is not None:
            self._buffer.append(choice.delta.content)

    def get(self, *, final: bool = False) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class LiteLLMStreamStructuredResponse(StreamStructuredResponse):
    """Implementation of `StreamStructuredResponse` for LiteLLM models."""

    _response: litellm.CustomStreamWrapper
    _delta_tool_calls: dict[int, litellm.types.utils.ChatCompletionDeltaToolCall]
    _timestamp: datetime
    _cost: result.Cost

    async def __anext__(self) -> None:
        chunk = await self._response.__anext__()
        self._cost += _map_cost(chunk)
        try:
            choice = chunk.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        if choice.finish_reason is not None:
            raise StopAsyncIteration()

        assert choice.delta.content is None, f'Expected tool calls, got content instead, invalid chunk: {chunk!r}'

        for new in choice.delta.tool_calls or []:
            if current := self._delta_tool_calls.get(new.index):
                if current.function is None:
                    current.function = new.function
                elif new.function is not None:
                    current.function.name = _utils.add_optional(current.function.name, new.function.name)
                    current.function.arguments = _utils.add_optional(current.function.arguments, new.function.arguments)
            else:
                self._delta_tool_calls[new.index] = new

    def get(self, *, final: bool = False) -> ModelResponse:
        items: list[ModelResponsePart] = []
        for c in self._delta_tool_calls.values():
            if f := c.function:
                if f.name is not None and f.arguments is not None:
                    items.append(ToolCallPart.from_json(f.name, f.arguments, c.id))

        return ModelResponse(items, timestamp=self._timestamp)

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


def _map_tool_call(t: ToolCallPart) -> dict[str, Any]:
    assert isinstance(t.args, ArgsJson), f'Expected ArgsJson, got {t.args}'
    return dict(
        id=t.tool_call_id,
        type='function',
        function={'name': t.tool_name, 'arguments': t.args.args_json},
    )


def _map_cost(response: litellm.ModelResponse) -> result.Cost:
    usage = response.get('usage')
    if usage is None:
        return result.Cost()
    else:
        details: dict[str, int] = {}
        if usage.completion_tokens_details is not None:
            details.update(usage.completion_tokens_details)
        if usage.prompt_tokens_details is not None:
            details.update(usage.prompt_tokens_details)
        details = {k: v for k, v in details.items() if v is not None}
        return result.Cost(
            request_tokens=usage.prompt_tokens,
            response_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            details=details,
        )
