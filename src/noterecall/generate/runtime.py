"""Building the real LLM client for answering. Implements the spend rules of proposal 004.

The client reads `OPENAI_API_KEY` from the environment (via `llm/openai_client.py`), caches every
answer in the response cache and stops at `llm.max_usd`, counting what the cache ledger already
holds. Tests replace `create_transport` with the fake transport.
"""

import tiktoken
from pydantic import BaseModel

from noterecall.chunking.tokens import Tokenizer
from noterecall.config import Settings
from noterecall.generate.answer import Generator
from noterecall.generate.verify import Verifier
from noterecall.llm.cache import ResponseCache, cache_key
from noterecall.llm.client import LLMCallError, LLMClient, LLMOutputError, LLMResult, Prompt
from noterecall.llm.cost import CostMeter, ledger_cost
from noterecall.llm.openai_client import create_transport
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient


def spent_so_far(settings: Settings, cache: ResponseCache) -> float:
    """Dollars already recorded in the response cache's ledger."""
    return ledger_cost(cache.spend(), settings.llm.prices)


def build_client(settings: Settings, cache: ResponseCache) -> CachingClient:
    """A caching client on the real transport, within the spend cap."""
    llm = settings.llm
    meter = CostMeter(llm.prices, llm.max_usd, spent_so_far(settings, cache))
    transport = create_transport(llm.reasoning_effort, llm.timeout_s, llm.max_retries)
    return CachingClient(transport, cache, meter, llm.reasoning_effort)


def build_generator(settings: Settings, client: LLMClient) -> Generator:
    """The answer generator with the pinned model and output cap."""
    return Generator(
        client,
        load_prompt("answer"),
        settings.llm.model,
        settings.generation.max_output_tokens,
    )


class NoCalls:
    """An `LLMClient` that refuses every call: used to build prompts for an estimate."""

    def complete_json[S: BaseModel](self, prompt: Prompt, schema: type[S]) -> LLMResult[S]:
        raise LLMCallError("this client never calls the API (dry run)")


def model_tokenizer(model: str) -> Tokenizer:
    """The tokenizer tiktoken maps to the model (an approximation of the provider's count)."""
    return Tokenizer(tiktoken.encoding_for_model(model))


def build_verifier(settings: Settings, client: LLMClient) -> Verifier:
    """The stage-3 verifier with the pre-registered effort and output cap (proposal 007)."""
    gen = settings.generation
    return Verifier(
        client,
        load_prompt("verify"),
        settings.llm.model,
        (gen.verify_effort, gen.verify_max_output_tokens),
    )


class CacheOnly:
    """An `LLMClient` that answers only from the response cache and refuses everything else.

    Used by the dry run to read the answers that already exist without ever calling the API."""

    def __init__(self, cache: ResponseCache, effort: str) -> None:
        self._cache = cache
        self._effort = effort

    def complete_json[S: BaseModel](self, prompt: Prompt, schema: type[S]) -> LLMResult[S]:
        stored = self._cache.get(cache_key(prompt, prompt.reasoning_effort or self._effort))
        if stored is None:
            raise LLMCallError("not in the response cache (and this client never calls the API)")
        try:
            parsed = schema.model_validate_json(stored.text)
        except ValueError as exc:
            raise LLMOutputError(f"a cached answer no longer parses: {exc}") from exc
        return LLMResult(
            parsed=parsed,
            usage=stored.usage,
            latency_s=0.0,
            from_cache=True,
            api_latency_s=stored.latency_s,
        )
