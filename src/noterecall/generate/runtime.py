"""Building the real LLM client for answering. Implements the spend rules of proposal 004.

The client reads `OPENAI_API_KEY` from the environment (via `llm/openai_client.py`), caches every
answer in the response cache and stops at `llm.max_usd`, counting what the cache ledger already
holds. Tests replace `create_transport` with the fake transport.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel

from noterecall.config import Settings
from noterecall.generate.answer import Generator
from noterecall.llm.cache import ResponseCache
from noterecall.llm.client import LLMCallError, LLMClient, LLMResult, Prompt
from noterecall.llm.cost import CostMeter, ledger_cost
from noterecall.llm.openai_client import create_transport
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient

if TYPE_CHECKING:
    from noterecall.chunking.tokens import Tokenizer


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


def model_tokenizer(model: str) -> "Tokenizer":
    """The tokenizer tiktoken maps to the model (an approximation of the provider's count).

    Imported here so the served API, which never counts tokens, does not need tiktoken."""
    import tiktoken  # noqa: PLC0415 - not installed in the API image

    from noterecall.chunking.tokens import Tokenizer  # noqa: PLC0415 - imports tiktoken

    return Tokenizer(tiktoken.encoding_for_model(model))
