"""LLM access via OpenRouter.

OpenRouter speaks the OpenAI wire format, so this module uses the `openai` SDK
with a redirected base URL rather than a vendor SDK. That is not only
convenience: the deployment target for this system is eventually a UAE
in-country host (CBUAE's sovereign financial cloud), and keeping a single
OpenAI-compatible seam means changing providers is a base-URL and model-string
change rather than a rewrite of the pipeline.

Two call shapes are exposed, matching the two things the pipeline needs:
`complete` for the bounded tool loop, and `complete_structured` for the final
assessment, whose shape is enforced by the API rather than requested in prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .config import Settings
from .obs.trace import log
from .resilience import CircuitBreaker, call_with_resilience

T = TypeVar("T", bound=BaseModel)

_llm_breaker = CircuitBreaker(service="openrouter.chat", failure_threshold=5)

# JSON Schema keywords that strict structured-output mode does not accept.
# Pydantic emits them from Field(ge=..., max_length=...); they are stripped for
# the wire and re-enforced by validating the parsed result against the model.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "pattern", "format",
        "minItems", "maxItems", "uniqueItems", "multipleOf", "default",
    }
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    raw_message: dict[str, Any] = field(default_factory=dict)


def to_strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Render a Pydantic model as a strict-mode JSON Schema.

    Strict mode requires every property to be listed in `required` and every
    object to forbid extra properties. Optionality is therefore expressed by the
    model emitting an empty list, not by omitting the field.
    """
    schema = model.model_json_schema()
    _strictify(schema)
    return schema


def _strictify(node: Any) -> None:
    if isinstance(node, dict):
        for keyword in _UNSUPPORTED_KEYWORDS & node.keys():
            node.pop(keyword, None)
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties") or {}
            node["required"] = list(properties.keys())
            node["additionalProperties"] = False
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for value in node:
            _strictify(value)


class LLM:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.client = AsyncOpenAI(
            # A missing key must not prevent construction. The service should
            # boot and report "degraded" on /health with the reason, rather than
            # crash-looping with an opaque SDK error — a container that reports
            # why it is unhealthy is far more operable than one that dies.
            api_key=settings.openrouter_api_key or "MISSING",
            base_url=settings.openrouter_base_url,
            timeout=120.0,
            max_retries=0,  # retries are owned by call_with_resilience
            default_headers={
                # OpenRouter uses these for attribution only; harmless elsewhere.
                "HTTP-Referer": "https://github.com/mal/sharia-compliance-agent",
                "X-Title": "Mal Shari'ah Compliance Agent",
            },
        )

    def _extra_body(self) -> dict[str, Any]:
        """Gateway-specific request extras.

        Reasoning effort is a capability of some models and not others — Haiku
        4.5 predates it. `SCA_MODEL_EFFORT=none` omits the field so a cheaper
        model can be swapped in without a 400.
        """
        if self._s.model_effort == "none":
            return {}
        return {"reasoning": {"effort": self._s.model_effort}}

    async def complete(
        self,
        *,
        messages: list[dict],
        system: str | None = None,
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> LLMResponse:
        payload: list[dict] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        async def call():
            kwargs: dict[str, Any] = {
                "model": model or self._s.model,
                "messages": payload,
                "max_tokens": max_tokens,
                "extra_body": self._extra_body(),
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return await self.client.chat.completions.create(**kwargs)

        response = await call_with_resilience(
            call,
            service="openrouter.chat",
            breaker=_llm_breaker,
            timeout=120.0,
            attempts=2,
        )
        return _to_llm_response(response)

    async def complete_structured(
        self,
        *,
        output_model: type[T],
        messages: list[dict],
        system: str | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> tuple[T | None, Usage]:
        """Constrain generation to a schema, then validate the result anyway.

        The API guarantees shape; it does not guarantee the value constraints
        that were stripped from the schema (ranges, lengths). Validating through
        the Pydantic model restores those, and a failure here is a real signal
        rather than a formatting accident — the pipeline escalates on it.
        """
        payload: list[dict] = []
        if system:
            payload.append({"role": "system", "content": system})
        payload.extend(messages)

        schema = to_strict_schema(output_model)

        async def call():
            return await self.client.chat.completions.create(
                model=model or self._s.model,
                messages=payload,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
                extra_body=self._extra_body(),
            )

        response = await call_with_resilience(
            call,
            service="openrouter.chat",
            breaker=_llm_breaker,
            timeout=120.0,
            attempts=2,
        )
        parsed = _to_llm_response(response)

        if not parsed.text.strip():
            log("llm.structured_empty", finish_reason=parsed.finish_reason)
            return None, parsed.usage
        try:
            return output_model.model_validate_json(parsed.text), parsed.usage
        except (ValidationError, json.JSONDecodeError) as exc:
            log("llm.structured_invalid", error=str(exc)[:300])
            return None, parsed.usage

    async def close(self) -> None:
        await self.client.close()


def _to_llm_response(response: Any) -> LLMResponse:
    choice = response.choices[0]
    message = choice.message
    calls: list[ToolCall] = []
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            log("llm.tool_args_unparseable", name=call.function.name)
            arguments = {}
        calls.append(ToolCall(id=call.id, name=call.function.name, arguments=arguments))

    usage = getattr(response, "usage", None)
    return LLMResponse(
        text=message.content or "",
        tool_calls=calls,
        finish_reason=choice.finish_reason or "stop",
        usage=Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        ),
        raw_message=message.model_dump(exclude_none=True),
    )


def breaker_state() -> dict[str, str]:
    return {_llm_breaker.service: _llm_breaker.state}
