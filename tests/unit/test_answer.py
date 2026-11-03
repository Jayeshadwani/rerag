"""The answer generator: retry with feedback, refusals, scrubbing, accounting (fake model only)."""

import json
import re
from pathlib import Path

from make_fixtures import FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY
from noterecall.config import PriceConfig
from noterecall.generate.answer import (
    Generator,
    Outcome,
    format_notes,
    short_ids,
)
from noterecall.llm.cache import ResponseCache
from noterecall.llm.client import Prompt
from noterecall.llm.cost import CostMeter
from noterecall.llm.fake import FakeTransport, Responder
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient
from noterecall.models import Chunk, Piece, PieceKind, PieceUnit

PRICE = PriceConfig(input=1.0, cached_input=0.1, output=4.0)


def chunk(n: int, text: str = "some note text") -> Chunk:
    piece = Piece(cell_index=n, kind=PieceKind.MARKDOWN, part=0, unit=PieceUnit.CELL, text=text)
    return Chunk(
        id=f"nb.ipynb#{n}",
        notebook_path="nb.ipynb",
        heading_path=("Top", "Sub"),
        pieces=(piece,),
        text=text,
        tokens=3,
        cell_hashes=(),
    )


def make(tmp_path: Path, responder: Responder) -> tuple[Generator, FakeTransport]:
    transport = FakeTransport(responder)
    client = CachingClient(
        transport, ResponseCache(tmp_path / "c.sqlite"), CostMeter({"m": PRICE}, 1.0), "minimal"
    )
    return Generator(client, load_prompt("answer"), "m", 100), transport


def reply(insufficient: bool = False, **claims: list[str]) -> str:
    items = [{"text": f"claim {k}", "citations": v} for k, v in claims.items()]
    return json.dumps({"insufficient": insufficient, "claims": items})


def test_short_ids_follow_rank_order_and_notes_show_where_each_chunk_is_from() -> None:
    context = short_ids([chunk(1, "alpha"), chunk(2, "beta")])
    assert list(context) == ["C1", "C2"]
    assert format_notes(context) == (
        "[C1] nb.ipynb > Top > Sub\nalpha\n\n[C2] nb.ipynb > Top > Sub\nbeta"
    )


def test_a_valid_answer_is_returned_with_chunk_ids_resolved(tmp_path: Path) -> None:
    gen, transport = make(tmp_path, lambda _p: reply(a=["C2"], b=["C1", "C2"]))
    response = gen.answer("what is x?", [chunk(1), chunk(2)])
    assert response.outcome is Outcome.ANSWERED
    assert response.attempts == 1
    assert [c.chunk_ids for c in response.claims] == [("nb.ipynb#2",), ("nb.ipynb#1", "nb.ipynb#2")]
    first = response.first_attempt
    assert first is not None
    assert (first.valid, first.claims, first.uncited_claims, first.unknown_citations) == (
        True, 2, 0, 0
    )  # fmt: skip
    assert len(transport.prompts) == 1


def test_an_invalid_first_answer_is_retried_once_with_the_problems_listed(tmp_path: Path) -> None:
    calls: list[Prompt] = []

    def responder(p: Prompt) -> str:
        calls.append(p)
        return reply(a=["C9"], b=[]) if len(calls) == 1 else reply(a=["C1"])

    gen, _ = make(tmp_path, responder)
    response = gen.answer("q", [chunk(1)])
    assert response.outcome is Outcome.ANSWERED
    assert response.attempts == 2
    assert "problems_with_your_previous_answer" not in calls[0].variable
    retry = calls[1].variable
    assert "unknown_id: claim 1 cites 'C9'" in retry
    assert "empty_citations: claim 2 cites nothing" in retry
    first = response.first_attempt
    assert first is not None
    assert (first.valid, first.claims, first.uncited_claims, first.unknown_citations) == (
        False, 2, 1, 1
    )  # fmt: skip


def test_two_invalid_answers_end_in_a_refusal_never_an_unvalidated_answer(tmp_path: Path) -> None:
    gen, transport = make(tmp_path, lambda _p: reply(a=["C9"]))
    response = gen.answer("q", [chunk(1)])
    assert response.outcome is Outcome.REFUSED_INVALID
    assert response.claims == ()
    assert response.attempts == 2
    assert len(transport.prompts) == 2  # one retry, no more
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


def test_the_model_saying_insufficient_is_a_refusal_after_one_call(tmp_path: Path) -> None:
    gen, transport = make(tmp_path, lambda _p: reply(insufficient=True))
    response = gen.answer("q", [chunk(1)])
    assert (response.outcome, response.attempts, response.claims) == (Outcome.REFUSED_MODEL, 1, ())
    assert len(transport.prompts) == 1


def test_insufficient_together_with_claims_is_retried(tmp_path: Path) -> None:
    replies = iter([reply(insufficient=True, a=["C1"]), reply(insufficient=True)])
    gen, _ = make(tmp_path, lambda _p: next(replies))
    response = gen.answer("q", [chunk(1)])
    assert (response.outcome, response.attempts) == (Outcome.REFUSED_MODEL, 2)


def test_unparsable_output_is_retried_and_counted_as_not_parsed(tmp_path: Path) -> None:
    replies = iter(["not json", reply(a=["C1"])])
    gen, transport = make(tmp_path, lambda _p: next(replies))
    response = gen.answer("q", [chunk(1)])
    assert response.outcome is Outcome.ANSWERED
    assert response.attempts == 2
    assert response.first_attempt is not None
    assert not response.first_attempt.parsed
    assert "unparsable:" in transport.prompts[1].variable


def test_the_question_is_scrubbed_before_it_is_sent(tmp_path: Path) -> None:
    gen, transport = make(tmp_path, lambda _p: reply(a=["C1"]))
    question = f"why does {FAKE_AWS_ACCESS_KEY} or {FAKE_ANTHROPIC_KEY} fail?"
    gen.answer(question, [chunk(1)])
    sent = transport.prompts[0].variable
    assert FAKE_AWS_ACCESS_KEY not in sent
    assert FAKE_ANTHROPIC_KEY not in sent
    assert "[REDACTED" in sent
    assert FAKE_AWS_ACCESS_KEY not in gen.prompt_for(question, [chunk(1)]).variable


def test_a_note_that_quotes_a_placeholder_is_sent_as_written(tmp_path: Path) -> None:
    gen, transport = make(tmp_path, lambda _p: reply(a=["C1"]))
    gen.answer("the real question", [chunk(1, "template says {{question}} and {{feedback}}")])
    sent = transport.prompts[0].variable
    assert "template says {{question}} and {{feedback}}" in sent
    assert sent.count("the real question") == 1


def test_usage_adds_up_over_both_attempts(tmp_path: Path) -> None:
    gen, transport = make(tmp_path, lambda _p: reply(a=["C9"]))
    response = gen.answer("q", [chunk(1)])
    words = sum(
        len((p.instructions + " " + p.shared + " " + p.variable).split()) for p in transport.prompts
    )
    assert response.usage.input_tokens == words  # the fake counts words, over both calls


def test_prompt_for_is_the_first_attempt_prompt_and_the_version_is_the_file_hash(
    tmp_path: Path,
) -> None:
    gen, transport = make(tmp_path, lambda _p: reply(a=["C1"]))
    gen.answer("q", [chunk(1)])
    assert gen.prompt_for("q", [chunk(1)]) == transport.prompts[0]
    assert re.fullmatch(r"[0-9a-f]{12}", gen.prompt_version)
    assert transport.prompts[0].version == gen.prompt_version
