"""The LLM interface every provider implements. Supports ADR-003 and ADR-007 (the LLM steps).

`LLMClient.complete_json(prompt, schema)` sends one prompt and returns the answer parsed into a
pydantic schema, or raises: an answer that does not parse is never turned into a default. A
`Prompt` is split into `instructions`, a `shared` prefix and a `variable` tail, in that order, so a
provider that caches a common prefix (OpenAI does, automatically) can reuse the notebook across all
the chunks of that notebook. `Usage` carries what the provider reports, reasoning tokens apart.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class LLMError(Exception):
    """Base class for LLM failures."""


class LLMOutputError(LLMError):
    """The provider answered, but not with a usable result (refusal, truncation, bad JSON)."""


class LLMCallError(LLMError):
    """The call failed after the retries (network, rate limit, server error, bad request)."""


class BudgetExceededError(LLMError):
    """The money already spent has reached the cap, so no further call is made."""


class Prompt(BaseModel):
    """One request: text in prefix order, the model and the cap on output tokens."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    model: str
    instructions: str
    shared: str
    variable: str
    max_output_tokens: int


class Usage(BaseModel):
    """Token counts from the provider. `reasoning_tokens` are part of `output_tokens`."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def plus(self, other: "Usage") -> "Usage":
        """Field-wise sum."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class LLMResult[T: BaseModel](BaseModel):
    """A parsed answer and how it was obtained. `from_cache` means no API call was made."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    parsed: T
    usage: Usage
    latency_s: float
    from_cache: bool


class LLMClient(Protocol):
    """Sends a prompt, returns the answer parsed into `schema` (or raises an `LLMError`)."""

    def complete_json[S: BaseModel](self, prompt: Prompt, schema: type[S]) -> LLMResult[S]: ...
