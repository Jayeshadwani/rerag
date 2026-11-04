"""The pre-flight smoke test for the real API (owner's rule: show real usage and cost first).

Step A sends one call built from three **invented** chunks, so no private data is involved.
It checks that the API accepts the settings (reasoning effort, strict JSON schema) and shows the
real token usage and cost. Step B asks three dev queries chosen by a fixed rule: the first two
answerable ones by id, then the first unanswerable one that stage 1 does not refuse (so the
model's own `insufficient` path is exercised), else the next answerable one.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from noterecall.config import PriceConfig
from noterecall.eval.gold import QueryType
from noterecall.eval.run_retrieval import QueryRef
from noterecall.generate.answer import CitedClaim, Generator, Outcome, Response
from noterecall.generate.verify import Verifier
from noterecall.llm.client import Usage
from noterecall.llm.cost import cost_usd
from noterecall.models import Chunk

SYNTHETIC_QUESTION = "How do I average a column for each group in pandas?"
SMOKE_CAP_USD = 0.05


def synthetic_chunks() -> list[Chunk]:
    """Three made-up note chunks (nothing from the owner's notebooks)."""
    texts = [
        ("demo/pandas.ipynb", ("Aggregation",), "Use df.groupby('key')['value'].mean() to get the "
         "mean of a column for each group."),
        ("demo/sklearn.ipynb", ("Splitting",), "train_test_split(X, y, test_size=0.2) holds out a "
         "fifth of the rows for testing."),
        ("demo/plots.ipynb", ("Plotting",), "plt.hist(values, bins=20) draws a histogram."),
    ]  # fmt: skip
    return [
        Chunk(id=f"synthetic-{n}", notebook_path=path, heading_path=headings, pieces=(), text=text,
              tokens=len(text.split()), cell_hashes=())
        for n, (path, headings, text) in enumerate(texts, start=1)
    ]  # fmt: skip


class SmokeRow(BaseModel):
    """One smoke call: ids and numbers only (never query text or answer text)."""

    model_config = ConfigDict(frozen=True)

    label: str
    outcome: str
    attempts: int
    claims: int
    usage: Usage
    cost_usd: float


def smoke_row(label: str, response: Response, price: PriceConfig) -> SmokeRow:
    return SmokeRow(
        label=label,
        outcome=response.outcome.value,
        attempts=response.attempts,
        claims=len(response.claims),
        usage=response.usage,
        cost_usd=cost_usd(response.usage, price),
    )


def run_step_a(generator: Generator, price: PriceConfig) -> SmokeRow:
    """One real call on the invented chunks."""
    response = generator.answer(SYNTHETIC_QUESTION, synthetic_chunks())
    return smoke_row("step A (invented chunks)", response, price)


def verify_row(label: str, response: Response, price: PriceConfig) -> SmokeRow:
    """A stage-3 call: the verifier's usage and cost, and what it did to the draft."""
    note = response.verification
    if note is None:
        raise ValueError("the response has no verification record")
    return SmokeRow(
        label=label,
        outcome=response.outcome.value,
        attempts=note.attempts,
        claims=len(response.claims),
        usage=note.usage,
        cost_usd=cost_usd(note.usage, price),
    )


def run_step_c(verifier: Verifier, price: PriceConfig) -> tuple[SmokeRow, str]:
    """One real stage-3 call on invented chunks: a draft with one claim its chunk states and one
    that contradicts its chunk. Returns the row and what the verifier did (observed)."""
    chunks = {c.id: c for c in synthetic_chunks()}
    draft = Response(
        outcome=Outcome.ANSWERED,
        attempts=1,
        claims=(
            CitedClaim(text="Use df.groupby('key')['value'].mean().", chunk_ids=("synthetic-1",)),
            CitedClaim(text="plt.hist draws a pie chart.", chunk_ids=("synthetic-3",)),
        ),
    )
    result = verifier.verify(SYNTHETIC_QUESTION, draft, chunks)
    note = result.verification
    answers = None if note is None else note.answers_question
    detail = f"answers_question={answers}; kept {len(result.claims)} of {len(draft.claims)} claims"
    return verify_row("step C (stage 3, invented chunks)", result, price), detail


def choose_step_b(items: Sequence[tuple[QueryRef, bool]]) -> list[QueryRef]:
    """Three dev queries by the fixed rule. `items` are (query, refused by stage 1), by id."""
    ordered = sorted(items, key=lambda item: item[0].id)
    answerable = [q for q, _ in ordered if q.type is not QueryType.UNANSWERABLE]
    chosen = answerable[:2]
    unanswerable = [q for q, refused in ordered if q.type is QueryType.UNANSWERABLE and not refused]
    chosen.append(unanswerable[0] if unanswerable else answerable[2])
    return chosen


def format_rows(rows: Sequence[SmokeRow]) -> str:
    """The table shown to the owner."""
    head = "| call | outcome | attempts | input | cached | output | reasoning | cost USD |"
    lines = [head, "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        u = r.usage
        lines.append(
            f"| {r.label} | {r.outcome} | {r.attempts} | {u.input_tokens} | {u.cached_tokens} | "
            f"{u.output_tokens} | {u.reasoning_tokens} | {r.cost_usd:.6f} |"
        )
    total = sum(r.cost_usd for r in rows)
    return "\n".join([*lines, f"total cost: ${total:.6f}"])
