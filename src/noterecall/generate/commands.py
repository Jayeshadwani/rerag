"""Plain-text output and planning for the M6 commands (`ask`, the dry run, the smoke test).

Kept apart from `cli.py` so the formatting and the estimate are tested without Typer. Nothing here
prints note text except `format_answer`, which shows an answer to the person who asked.
"""

from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from noterecall.config import Settings
from noterecall.eval.run_retrieval import QueryRef
from noterecall.generate.answer import CitedClaim, Generator, Outcome, Response
from noterecall.generate.ask import Asker, AskResult
from noterecall.generate.estimate import GenerationEstimate, estimate_generation
from noterecall.generate.verify import Verifier
from noterecall.llm.client import LLMCallError, LLMOutputError, Prompt
from noterecall.models import Chunk

ASSUMED_OUTPUT_TOKENS = 300  # an assumption of the estimate, not a measurement
ASSUMED_VERIFY_OUTPUT_TOKENS = 1500  # reasoning at `medium` plus the verdict: an assumption

REFUSALS = {
    Outcome.REFUSED_STAGE1: "Not in your notes. (No model call was made: the best match was "
    "too weak.)",
    Outcome.REFUSED_MODEL: "Not in your notes. (The model found nothing in the retrieved notes "
    "that answers it.)",
    Outcome.REFUSED_INVALID: "No answer: the model's citations failed validation twice, so "
    "nothing is shown.",
}


def where(chunk: Chunk) -> str:
    """Notebook, headings and cells of a chunk, as shown next to an answer."""
    path = " > ".join([chunk.notebook_path, *chunk.heading_path])
    cells = sorted({p.cell_index for p in chunk.pieces})
    return f"{path} (cells {', '.join(map(str, cells))})" if cells else path


def format_answer(result: AskResult, chunks: Mapping[str, Chunk]) -> str:
    """The answer with numbered sources, or the reason there is none."""
    response = result.response
    if response.outcome is not Outcome.ANSWERED:
        return REFUSALS[response.outcome]
    order: list[str] = []
    lines = []
    for claim in response.claims:
        marks = []
        for chunk_id in claim.chunk_ids:
            if chunk_id not in order:
                order.append(chunk_id)
            marks.append(f"[{order.index(chunk_id) + 1}]")
        lines.append(f"- {claim.text} {''.join(marks)}")
    sources = [f"[{n}] {where(chunks[cid])}" for n, cid in enumerate(order, start=1)]
    return "\n".join([*lines, "", "Sources:", *sources])


def format_context(result: AskResult, chunks: Mapping[str, Chunk]) -> str:
    """The signals and what was in the context (for `--show-context`)."""
    s = result.signals
    head = f"top cosine {s.top_cosine:.3f}, normalized top BM25 {s.bm25_norm:.3f}"
    rows = [f"C{n}: {where(chunks[cid])}" for n, cid in enumerate(result.context_ids, start=1)]
    return "\n".join([head, *rows])


class GenerationPlan(BaseModel):
    """What a run would do: stage-1 refusals, calls already in the cache, and the estimates."""

    model_config = ConfigDict(frozen=True)

    queries: int
    refused_stage1: int
    answer_cached: int
    answer: GenerationEstimate
    stage3_cached: int = 0
    stage3_assumed: int = 0
    stage3: GenerationEstimate | None = None


def plan_generation(
    asker: Asker,
    queries: Sequence[QueryRef],
    settings: Settings,
    count: Callable[[str], int],
    is_cached: Callable[[Prompt], bool],
) -> GenerationPlan:
    """How many queries stage 1 refuses, how many answers are cached, and the estimate for the
    rest (no model call)."""
    prompts = []
    refused = cached = 0
    for q in queries:
        chunks, signals = asker.context_for(q.text)
        if asker.stage1_refuses(signals):
            refused += 1
            continue
        prompt = asker.generator.prompt_for(q.text, chunks)
        if is_cached(prompt):
            cached += 1
        else:
            prompts.append(prompt)
    price = settings.llm.prices[settings.llm.model]
    estimate = estimate_generation(prompts, count, price, ASSUMED_OUTPUT_TOKENS)
    return GenerationPlan(queries=len(queries), refused_stage1=refused, answer_cached=cached,
                          answer=estimate)  # fmt: skip


def plan_stage3(  # noqa: PLR0913, PLR0917 - asker, cached answerer, verifier, queries, config
    plan: GenerationPlan,
    asker: Asker,
    cached_answerer: Generator,
    verifier: Verifier,
    inputs: tuple[Sequence[QueryRef], Mapping[str, Chunk]],
    settings: Settings,
    hooks: tuple[Callable[[str], int], Callable[[Prompt], bool]],
) -> GenerationPlan:
    """Add the stage-3 estimate. The drafts come from the cache; a draft that is not cached is
    priced as the worst case (one claim citing every chunk in the context)."""
    queries, chunks_by_id = inputs
    count, is_cached = hooks
    prompts = []
    cached = assumed = 0
    for q in queries:
        chunks, signals = asker.context_for(q.text)
        if asker.stage1_refuses(signals):
            continue
        try:
            draft = cached_answerer.answer(q.text, chunks)
        except (LLMCallError, LLMOutputError):
            draft = None
        if draft is not None and draft.outcome is not Outcome.ANSWERED:
            continue  # refused before stage 3: no verifier call
        if draft is None:
            assumed += 1
            ids = tuple(c.id for c in chunks)
            draft = Response(
                outcome=Outcome.ANSWERED, claims=(CitedClaim(text="x", chunk_ids=ids),)
            )
        prompt = verifier.prompt_for(q.text, draft.claims, chunks_by_id)
        if is_cached(prompt):
            cached += 1
        else:
            prompts.append(prompt)
    price = settings.llm.prices[settings.llm.model]
    estimate = estimate_generation(prompts, count, price, ASSUMED_VERIFY_OUTPUT_TOKENS)
    return plan.model_copy(
        update={"stage3": estimate, "stage3_cached": cached, "stage3_assumed": assumed}
    )


def format_plan(plan: GenerationPlan, spent: float, cap: float) -> str:
    """The dry-run text."""
    e = plan.answer
    lines = [
        f"generation: {plan.queries} queries, {plan.refused_stage1} refused by stage 1 (no call), "
        f"{plan.queries - plan.refused_stage1} reach the model",
        f"  answer calls: {plan.answer_cached} already in the cache, {e.calls} to make",
        f"    input tokens: {e.input_tokens}; assumption: {e.assumed_output_tokens_per_call} "
        f"output tokens per call; output cap {e.output_cap_per_call} (includes reasoning)",
        f"    estimated cost ${e.cost_expected:.4f}; upper bound ${e.cost_upper_bound:.4f}",
    ]
    total_expected, total_bound = e.cost_expected, e.cost_upper_bound
    if plan.stage3 is not None:
        v = plan.stage3
        lines += [
            f"  stage 3 calls: {plan.stage3_cached} already in the cache, {v.calls} to make "
            f"({plan.stage3_assumed} priced as a worst-case draft: answer not cached)",
            f"    input tokens: {v.input_tokens}; assumption: {v.assumed_output_tokens_per_call} "
            f"output tokens per call; output cap {v.output_cap_per_call} (includes reasoning)",
            f"    estimated cost ${v.cost_expected:.4f}; upper bound ${v.cost_upper_bound:.4f}",
        ]
        total_expected += v.cost_expected
        total_bound += v.cost_upper_bound
    lines += [
        f"  total estimated cost: ${total_expected:.4f}; upper bound (two capped calls per query "
        f"and stage): ${total_bound:.4f}",
        f"  spent so far (cache ledger): ${spent:.4f} of the ${cap:.2f} cap",
        "  token counts use tiktoken's encoding for the model: close to, not the provider's",
    ]
    return "\n".join(lines)
