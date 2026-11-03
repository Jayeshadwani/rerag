"""Plain-text output and planning for the M6 commands (`ask`, the dry run, the smoke test).

Kept apart from `cli.py` so the formatting and the estimate are tested without Typer. Nothing here
prints note text except `format_answer`, which shows an answer to the person who asked.
"""

from collections.abc import Callable, Mapping, Sequence

from noterecall.config import Settings
from noterecall.eval.run_retrieval import QueryRef
from noterecall.generate.answer import Generator, Outcome
from noterecall.generate.ask import Asker, AskResult
from noterecall.generate.estimate import GenerationEstimate, estimate_generation
from noterecall.models import Chunk

ASSUMED_OUTPUT_TOKENS = 300  # an assumption of the estimate, not a measurement

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


def plan_generation(
    asker: Asker,
    generator: Generator,
    queries: Sequence[QueryRef],
    settings: Settings,
    count: Callable[[str], int],
) -> tuple[int, GenerationEstimate]:
    """How many queries stage 1 refuses, and the estimate for the rest (no model call)."""
    prompts = []
    refused = 0
    for q in queries:
        chunks, signals = asker.context_for(q.text)
        if asker.stage1_refuses(signals):
            refused += 1
        else:
            prompts.append(generator.prompt_for(q.text, chunks))
    price = settings.llm.prices[settings.llm.model]
    return refused, estimate_generation(prompts, count, price, ASSUMED_OUTPUT_TOKENS)


def format_plan(
    refused: int, total: int, estimate: GenerationEstimate, spent: float, cap: float
) -> str:
    """The dry-run text."""
    return "\n".join(
        [
            f"generation: {total} queries, {refused} refused by stage 1 (no call), "
            f"{estimate.calls} go to the model",
            f"  input tokens (first attempts): {estimate.input_tokens}",
            f"  assumption: {estimate.assumed_output_tokens_per_call} output tokens per call",
            f"  output cap per call: {estimate.output_cap_per_call} tokens (includes reasoning)",
            f"  estimated cost: ${estimate.cost_expected:.4f}",
            f"  upper bound (two capped calls per query): ${estimate.cost_upper_bound:.4f}",
            f"  spent so far (cache ledger): ${spent:.4f} of the ${cap:.2f} cap",
            "  token counts use tiktoken's encoding for the model: close to, not exactly, the "
            "provider's",
        ]
    )
