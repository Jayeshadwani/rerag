"""The OpenAI transport (proposal 004). Implements the real side of `LLMClient`.

One call is one `responses.create` request: the instructions as the system text, and a single user
message that is the shared prefix (the notebook) followed by the variable tail (the chunk), so
OpenAI's automatic prefix cache can reuse the notebook across the chunks of that notebook.
`prompt_cache_key` is a hash of the shared prefix, which helps requests with the same prefix land
together. The answer is requested as strict JSON Schema and returned as text for the caller to
parse. `store` is off, so OpenAI does not keep the request. Network errors, timeouts, rate limits
and server errors are retried with exponential backoff; anything else, a refusal, an incomplete
answer or a missing usage record is raised, never replaced by a default. The API key is read from
the environment by the SDK and is scrubbed from every message this module raises or logs.
"""

import hashlib
import os
import time
from collections.abc import Callable
from typing import Protocol

import openai
import structlog

from noterecall.llm.client import LLMCallError, LLMOutputError, Prompt, Usage
from noterecall.llm.transport import RawCompletion

log = structlog.get_logger()
MAX_BACKOFF_S = 30.0
_RETRYABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class _InputDetails(Protocol):
    cached_tokens: int
    cache_write_tokens: int


class _OutputDetails(Protocol):
    reasoning_tokens: int


class _UsageLike(Protocol):
    input_tokens: int
    input_tokens_details: _InputDetails
    output_tokens: int
    output_tokens_details: _OutputDetails


class _IncompleteLike(Protocol):
    reason: str | None


class _ResponseLike(Protocol):
    status: str | None
    incomplete_details: _IncompleteLike | None
    output_text: str
    usage: _UsageLike | None


class ResponsesLike(Protocol):
    """The slice of `openai.OpenAI().responses` this module uses."""

    def create(  # noqa: PLR0913 - mirrors the request fields
        self,
        *,
        model: str,
        instructions: str,
        input: str,
        max_output_tokens: int,
        reasoning: dict[str, str],
        text: dict[str, object],
        prompt_cache_key: str,
        store: bool,
    ) -> _ResponseLike: ...


def scrub(message: str, secret: str | None) -> str:
    """Remove the API key (and anything shaped like a partial key echo) from a message."""
    if secret:
        message = message.replace(secret, "<api-key>")
    return message


class OpenAITransport:
    """Sends prompts to the OpenAI Responses API."""

    def __init__(
        self,
        responses: ResponsesLike,
        effort: str,
        max_retries: int,
        sleep: Callable[[float], None] = time.sleep,
        secret: str | None = None,
    ) -> None:
        self._responses = responses
        self._effort = effort
        self._max_retries = max_retries
        self._sleep = sleep
        self._secret = secret

    def complete_text(
        self, prompt: Prompt, schema_name: str, schema_json: dict[str, object]
    ) -> RawCompletion:
        start = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._send(prompt, schema_name, schema_json)
            except (
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
            ) as exc:
                if attempts > self._max_retries:
                    detail = scrub(str(exc), self._secret)
                    raise LLMCallError(
                        f"{type(exc).__name__} after {attempts} attempts: {detail}"
                    ) from None
                delay = min(2.0 ** (attempts - 1), MAX_BACKOFF_S)
                log.warning("llm.retry", error=type(exc).__name__, attempt=attempts, delay_s=delay)
                self._sleep(delay)
                continue
            except openai.AuthenticationError:
                raise LLMCallError("authentication failed: check OPENAI_API_KEY") from None
            except openai.APIStatusError as exc:
                detail = scrub(str(exc), self._secret)
                raise LLMCallError(
                    f"{type(exc).__name__} (HTTP {exc.status_code}): {detail}"
                ) from None
            return self._result(response, time.perf_counter() - start, attempts)

    def _send(
        self, prompt: Prompt, schema_name: str, schema_json: dict[str, object]
    ) -> _ResponseLike:
        shared_key = hashlib.sha256(prompt.shared.encode()).hexdigest()[:16]
        return self._responses.create(
            model=prompt.model,
            instructions=prompt.instructions,
            input=(prompt.shared + "\n\n" + prompt.variable) if prompt.shared else prompt.variable,
            max_output_tokens=prompt.max_output_tokens,
            reasoning={"effort": self._effort},
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema_json,
                    "strict": True,
                }
            },
            prompt_cache_key=shared_key,
            store=False,
        )

    @staticmethod
    def _result(response: _ResponseLike, latency_s: float, attempts: int) -> RawCompletion:
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            raise LLMOutputError(f"the answer was cut off ({reason}); raise the output token cap")
        if response.status not in (None, "completed"):
            raise LLMOutputError(f"the response status was {response.status!r}")
        if not response.output_text:
            raise LLMOutputError("the response had no text (a refusal or an empty answer)")
        if response.usage is None:
            raise LLMOutputError("the response carried no usage record")
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            cached_tokens=u.input_tokens_details.cached_tokens,
            cache_write_tokens=u.input_tokens_details.cache_write_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=u.output_tokens_details.reasoning_tokens,
        )
        return RawCompletion(
            text=response.output_text, usage=usage, latency_s=latency_s, attempts=attempts
        )


def create_transport(effort: str, timeout_s: float, max_retries: int) -> OpenAITransport:
    """A transport on the real SDK. The key comes from `OPENAI_API_KEY`; SDK retries are off
    because this module retries (and logs) itself."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise LLMCallError("OPENAI_API_KEY is not set in the environment")
    client = openai.OpenAI(api_key=key, timeout=timeout_s, max_retries=0)
    return OpenAITransport(client.responses, effort, max_retries, secret=key)  # type: ignore[arg-type]
