"""The `noterecall enrich` steps: plan, estimate, and (only when told to) run. ADR-003, ADR-007.

Each step first builds its jobs and prices them from token counts without calling anything. A
real run is a separate decision: it needs the caller to say so explicitly, and it stops at the
money cap set in config. Answers go into the response cache, so a repeated or resumed run pays
only for what is missing. The cleaned notebooks and the blurbs are read back from that cache.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import tiktoken

from noterecall.chunking.tokens import Tokenizer
from noterecall.config import Settings
from noterecall.enrich.contextual import Blurb
from noterecall.enrich.estimate import StepEstimate, estimate_step
from noterecall.enrich.lookup import BlurbSetup, blurb_job_groups
from noterecall.enrich.run import (
    Job,
    RunSummary,
    clean_groups,
    read_cleaned,
    run_groups,
)
from noterecall.eval.run_retrieval import Notebooks, build_chunks
from noterecall.ingest.llm_clean import CleanedCell, CleanReport, apply_all
from noterecall.llm.cache import ResponseCache, cache_key
from noterecall.llm.cost import CostMeter, ledger_cost
from noterecall.llm.openai_client import create_transport
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient

# Stated assumptions of the estimate (not measurements): a blurb is one or two sentences plus the
# JSON wrapper; a cleaned cell is about as long as the cell it replaces, plus the wrapper.
ASSUMED_BLURB_OUTPUT_TOKENS = 80
ASSUMED_CLEAN_WRAPPER_TOKENS = 15


@dataclass(frozen=True)
class Plan[K]:
    """The jobs of one step and what they are expected to cost."""

    name: str
    model: str
    groups: list[list[Job[K]]]
    schema: type[Blurb] | type[CleanedCell]
    estimate: StepEstimate
    assumption: str


def model_tokenizer(model: str) -> Tokenizer:
    """The tokenizer tiktoken maps to the model (an approximation of the provider's count)."""
    return Tokenizer(tiktoken.encoding_for_model(model))


def blurb_setup(settings: Settings) -> BlurbSetup:
    """The blurb prompt template with its model, effort and output cap."""
    llm = settings.llm
    return BlurbSetup(
        template=load_prompt("contextual_blurb"),
        model=llm.blurb_model,
        effort=llm.reasoning_effort,
        max_output_tokens=llm.max_output_tokens_blurb,
    )


def _estimate[K](
    settings: Settings, cache: ResponseCache, groups: list[list[Job[K]]], model: str, assumed: int
) -> StepEstimate:
    effort = settings.llm.reasoning_effort
    tokenizer = model_tokenizer(model)
    return estimate_step(
        groups,
        lambda job: cache.get(cache_key(job.prompt, effort)) is not None,
        tokenizer.count,
        settings.llm.prices[model],
        assumed,
    )


def plan_blurbs(
    settings: Settings, corpus: Notebooks, cache: ResponseCache, tokenizer: Tokenizer
) -> Plan[str]:
    """Blurb jobs for every chunk of the corpus at the decided chunk cap."""
    decided = settings.retrieval.decided
    if decided is None:
        raise EnrichError("config has no [retrieval.decided]; the chunk cap comes from M4")
    setup = blurb_setup(settings)
    chunks = build_chunks(corpus, decided.chunk_cap, "structural", tokenizer)
    groups = blurb_job_groups(setup, dict(corpus), chunks)
    estimate = _estimate(settings, cache, groups, setup.model, ASSUMED_BLURB_OUTPUT_TOKENS)
    text = f"{ASSUMED_BLURB_OUTPUT_TOKENS} output tokens per call (one or two sentences)"
    return Plan("blurbs", setup.model, groups, Blurb, estimate, text)


def plan_clean(
    settings: Settings, corpus: Notebooks, cache: ResponseCache
) -> Plan[tuple[str, int]]:
    """Clean jobs for every non-empty markdown cell."""
    llm = settings.llm
    groups = clean_groups(
        corpus, load_prompt("clean_markdown"), llm.clean_model, llm.max_output_tokens_clean
    )
    tokenizer = model_tokenizer(llm.clean_model)
    jobs = [job for group in groups for job in group]
    mean = math.ceil(sum(tokenizer.count(j.prompt.variable) for j in jobs) / max(len(jobs), 1))
    assumed = mean + ASSUMED_CLEAN_WRAPPER_TOKENS
    estimate = _estimate(settings, cache, groups, llm.clean_model, assumed)
    text = (
        f"{assumed} output tokens per call (the average cell, {mean} tokens, plus "
        f"{ASSUMED_CLEAN_WRAPPER_TOKENS} for the JSON wrapper)"
    )
    return Plan("clean", llm.clean_model, groups, CleanedCell, estimate, text)


class EnrichError(Exception):
    """Raised when a step cannot be planned or run."""


def format_plan[K](plan: Plan[K], settings: Settings, spent_usd: float) -> str:
    """The dry-run text: what would be sent, the assumptions, three cost figures, the cap."""
    e = plan.estimate
    llm = settings.llm
    lines = [
        f"enrich {plan.name}: model {plan.model}, reasoning effort {llm.reasoning_effort}",
        f"  jobs: {e.jobs_total} total, {e.jobs_already_cached} already in the cache, "
        f"{e.jobs_to_run} to run",
        f"  input tokens to send: {e.input_tokens} "
        f"({e.reusable_prefix_tokens} of them a shared notebook prefix after the first call)",
        f"  assumption: {plan.assumption}",
        f"  output cap per call: {e.output_cap_per_call} tokens (includes reasoning)",
        f"  estimated cost, prefix caching works: ${e.cost_with_prefix_caching:.4f}",
        f"  estimated cost, no caching at all:    ${e.cost_without_caching:.4f}",
        f"  upper bound, every call hits its cap: ${e.cost_upper_bound:.4f}",
        f"  spent so far (cache ledger): ${spent_usd:.4f} of the ${llm.max_usd:.2f} cap",
        "  token counts use tiktoken's encoding for the model: close to, not exactly, the "
        "provider's",
    ]
    return "\n".join(lines)


def spent_so_far(settings: Settings, cache: ResponseCache) -> float:
    """Dollars already recorded in the cache ledger."""
    return ledger_cost(cache.spend(), settings.llm.prices)


def execute[K](plan: Plan[K], settings: Settings, cache: ResponseCache) -> RunSummary:
    """Run the plan's jobs against the real API, within the cap. Needs `OPENAI_API_KEY`."""
    llm = settings.llm
    meter = CostMeter(llm.prices, llm.max_usd, spent_so_far(settings, cache))
    transport = create_transport(llm.reasoning_effort, llm.timeout_s, llm.max_retries)
    client = CachingClient(transport, cache, meter, llm.reasoning_effort)
    _answers, summary = run_groups(plan.groups, client, plan.schema, llm.concurrency)
    return summary


def format_run(summary: RunSummary, spent_usd: float, max_usd: float) -> str:
    """The text printed after a real run."""
    failures = ", ".join(f"{n} x {why}" for why, n in sorted(summary.failures.items())) or "none"
    return "\n".join(
        [
            f"  ran {summary.jobs} jobs: {summary.from_api} answered by the API, "
            f"{summary.from_cache} from the cache, unusable answers: {failures}",
            f"  spent so far (cache ledger): ${spent_usd:.4f} of the ${max_usd:.2f} cap",
        ]
    )


def describe_or_run[K](
    plan: Plan[K],
    settings: Settings,
    cache: ResponseCache,
    *,
    dry_run: bool,
    confirmed: bool,
) -> str:
    """The estimate, and only if `confirmed` (and not `dry_run`) the real run after it."""
    text = format_plan(plan, settings, spent_so_far(settings, cache))
    if dry_run:
        return text
    if not confirmed:
        raise EnrichError(
            f"{text}\n\nno call was made: this would call the API for "
            f"{plan.estimate.jobs_to_run} jobs. Read the estimate, then re-run with --yes."
        )
    summary = execute(plan, settings, cache)
    return text + "\n" + format_run(summary, spent_so_far(settings, cache), settings.llm.max_usd)


def cleaned_notebooks(
    settings: Settings, corpus: Notebooks, cache: ResponseCache
) -> tuple[Notebooks, CleanReport]:
    """The notebooks with verified cleaned markdown cells, answers read from the cache."""
    llm = settings.llm
    groups = clean_groups(
        corpus, load_prompt("clean_markdown"), llm.clean_model, llm.max_output_tokens_clean
    )
    jobs: Sequence[Job[tuple[str, int]]] = [job for group in groups for job in group]
    answers = read_cleaned(jobs, cache, llm.reasoning_effort)
    return apply_all(corpus, answers)
