"""The thin layer between a provider and the caching client. Supports ADR-003/007 (LLM steps).

A `Transport` sends one prompt and returns the raw JSON text the model produced plus its usage.
`CachingClient` is the `LLMClient` the rest of the code uses: it answers from the response cache
when it can, otherwise checks the spend cap, calls the transport, stores the answer, and parses it
into the requested schema. An answer that does not parse is raised as `LLMOutputError` and not
cached. Every call logs the model, prompt version, token counts, latency and whether the cache
answered.
"""

import time
from typing import Protocol

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from noterecall.llm.cache import ResponseCache, Stored, cache_key
from noterecall.llm.client import LLMOutputError, LLMResult, Prompt, Usage
from noterecall.llm.cost import CostMeter

log = structlog.get_logger()


class RawCompletion(BaseModel):
    """What a provider returns for one prompt."""

    model_config = ConfigDict(frozen=True)

    text: str
    usage: Usage
    latency_s: float
    attempts: int = 1


class Transport(Protocol):
    """Sends one prompt; `schema_json` is the JSON Schema the answer must follow."""

    def complete_text(
        self, prompt: Prompt, schema_name: str, schema_json: dict[str, object]
    ) -> RawCompletion: ...


class CachingClient:
    """An `LLMClient` with a response cache and a spend cap in front of a `Transport`."""

    def __init__(
        self, transport: Transport, cache: ResponseCache, meter: CostMeter, effort: str
    ) -> None:
        self._transport = transport
        self._cache = cache
        self._meter = meter
        self._effort = effort

    def complete_json[S: BaseModel](self, prompt: Prompt, schema: type[S]) -> LLMResult[S]:
        key = cache_key(prompt, prompt.reasoning_effort or self._effort)
        stored = self._cache.get(key)
        if stored is not None:
            parsed = self._parse(stored.text, schema)
            self._log(prompt, stored.usage, stored.latency_s, from_cache=True, attempts=0)
            return LLMResult(
                parsed=parsed,
                usage=stored.usage,
                latency_s=0.0,
                from_cache=True,
                api_latency_s=stored.latency_s,
            )
        self._meter.check()
        start = time.perf_counter()
        raw = self._transport.complete_text(prompt, schema.__name__, schema.model_json_schema())
        self._meter.add(prompt.model, raw.usage)  # an unusable answer was still paid for
        parsed = self._parse(raw.text, schema)  # raises before anything is stored
        self._cache.put(key, prompt, Stored(raw.text, raw.usage, raw.latency_s))
        self._log(prompt, raw.usage, raw.latency_s, from_cache=False, attempts=raw.attempts)
        return LLMResult(
            parsed=parsed,
            usage=raw.usage,
            latency_s=time.perf_counter() - start,
            from_cache=False,
            api_latency_s=raw.latency_s,
        )

    @staticmethod
    def _parse[S: BaseModel](text: str, schema: type[S]) -> S:
        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMOutputError(f"the answer does not match {schema.__name__}: {exc}") from exc

    @staticmethod
    def _log(
        prompt: Prompt, usage: Usage, latency_s: float, *, from_cache: bool, attempts: int
    ) -> None:
        log.info(
            "llm.call",
            prompt=prompt.name,
            prompt_version=prompt.version,
            model=prompt.model,
            input_tokens=usage.input_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            latency_s=round(latency_s, 3),
            from_cache=from_cache,
            attempts=attempts,
        )
