"""Token and dollar estimate for answering a set of questions, before any call (`--dry-run`).

Input tokens are counted on the real first-attempt prompts with the tokenizer tiktoken maps to the
model (close to, not equal to, the provider's count). Output tokens are an assumption the caller
states. The upper bound lets every question make two calls (one retry) that both use their whole
output cap.
"""

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from noterecall.config import PriceConfig
from noterecall.llm.client import Prompt
from noterecall.llm.cost import PER


class GenerationEstimate(BaseModel):
    """What answering the questions would cost."""

    model_config = ConfigDict(frozen=True)

    calls: int
    input_tokens: int
    assumed_output_tokens_per_call: int
    output_cap_per_call: int
    cost_expected: float
    cost_upper_bound: float


def estimate_generation(
    prompts: Sequence[Prompt],
    count: Callable[[str], int],
    price: PriceConfig,
    assumed_output_tokens: int,
) -> GenerationEstimate:
    """Cost of one call per prompt (expected) and of two capped calls per prompt (bound)."""
    inputs = [count(p.instructions + "\n" + p.shared + "\n" + p.variable) for p in prompts]
    total_in = sum(inputs)
    cap = prompts[0].max_output_tokens if prompts else 0
    expected = (total_in * price.input + len(prompts) * assumed_output_tokens * price.output) / PER
    bound = 2 * (total_in * price.input + len(prompts) * cap * price.output) / PER
    return GenerationEstimate(
        calls=len(prompts),
        input_tokens=total_in,
        assumed_output_tokens_per_call=assumed_output_tokens,
        output_cap_per_call=cap,
        cost_expected=expected,
        cost_upper_bound=bound,
    )
