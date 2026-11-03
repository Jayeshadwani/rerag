"""Token and dollar estimate for a set of LLM jobs, before any call is made (`--dry-run`).

Tokens are counted with the tokenizer `tiktoken` maps to the model (an approximation of what the
provider will bill). For each notebook the first job is priced as ordinary input and, in the
cached case, every later job's shared prefix is priced at the cached-input price. Three costs are
given: the cached case, the case where caching does not happen at all, and an upper bound that also
charges every call the full output-token cap. Output tokens in the first two are an assumption the
caller states. Jobs already in the response cache are left out.
"""

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from noterecall.config import PriceConfig
from noterecall.enrich.run import Job
from noterecall.llm.cost import PER

Counter = Callable[[str], int]


class StepEstimate(BaseModel):
    """Estimate for one step (blurbs, or the clean)."""

    model_config = ConfigDict(frozen=True)

    jobs_total: int
    jobs_to_run: int
    jobs_already_cached: int
    input_tokens: int
    reusable_prefix_tokens: int
    assumed_output_tokens_per_call: int
    output_cap_per_call: int
    cost_with_prefix_caching: float
    cost_without_caching: float
    cost_upper_bound: float


def estimate_step[K](
    groups: Sequence[Sequence[Job[K]]],
    is_cached: Callable[[Job[K]], bool],
    count: Counter,
    price: PriceConfig,
    assumed_output_tokens: int,
) -> StepEstimate:
    """Estimate the jobs that are not yet in the response cache."""
    total = sum(len(g) for g in groups)
    todo = [[j for j in g if not is_cached(j)] for g in groups]
    to_run = sum(len(g) for g in todo)
    input_tokens = reusable = 0
    for group in todo:
        for position, job in enumerate(group):
            p = job.prompt
            input_tokens += count(p.instructions + "\n" + p.shared + "\n" + p.variable)
            if position > 0 and p.shared:
                reusable += count(p.instructions + "\n" + p.shared)
    cap = groups[0][0].prompt.max_output_tokens if total else 0
    out_assumed = to_run * assumed_output_tokens
    cached_input = (input_tokens - reusable) * price.input + reusable * price.cached_input
    plain_input = input_tokens * price.input
    return StepEstimate(
        jobs_total=total,
        jobs_to_run=to_run,
        jobs_already_cached=total - to_run,
        input_tokens=input_tokens,
        reusable_prefix_tokens=reusable,
        assumed_output_tokens_per_call=assumed_output_tokens,
        output_cap_per_call=cap,
        cost_with_prefix_caching=(cached_input + out_assumed * price.output) / PER,
        cost_without_caching=(plain_input + out_assumed * price.output) / PER,
        cost_upper_bound=(plain_input + to_run * cap * price.output) / PER,
    )
