"""Stage 3: the verdict check, how a verdict changes an answer, retries and fail-closed refusal."""

import json
from pathlib import Path

from make_fixtures import FAKE_AWS_ACCESS_KEY
from noterecall.config import PriceConfig
from noterecall.generate.answer import CitedClaim, Outcome, Response
from noterecall.generate.verify import (
    ClaimVerdict,
    Verdict,
    Verifier,
    apply_verdict,
    cited_notes,
    format_claims,
    validate_verdict,
)
from noterecall.llm.cache import ResponseCache
from noterecall.llm.client import Prompt
from noterecall.llm.cost import CostMeter
from noterecall.llm.fake import FakeTransport, Responder
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient
from noterecall.models import Chunk, Piece, PieceKind, PieceUnit

PRICE = PriceConfig(input=1.0, cached_input=0.1, output=4.0)


def chunk(n: int, text: str) -> Chunk:
    piece = Piece(cell_index=n, kind=PieceKind.MARKDOWN, part=0, unit=PieceUnit.CELL, text=text)
    return Chunk(
        id=f"nb.ipynb#{n}",
        notebook_path="nb.ipynb",
        heading_path=("Top",),
        pieces=(piece,),
        text=text,
        tokens=3,
        cell_hashes=(),
    )


CHUNKS = {c.id: c for c in (chunk(1, "alpha is a"), chunk(2, "beta is b"), chunk(3, "gamma is c"))}


def claim(text: str, *numbers: int) -> CitedClaim:
    return CitedClaim(text=text, chunk_ids=tuple(f"nb.ipynb#{n}" for n in numbers))


def drafted(*claims: CitedClaim) -> Response:
    return Response(outcome=Outcome.ANSWERED, claims=claims, attempts=1)


def verdict(answers: bool, *supported: bool) -> str:
    entries = [{"claim": n, "supported": s} for n, s in enumerate(supported, start=1)]
    return json.dumps({"answers_question": answers, "claims": entries})


def make(tmp_path: Path, responder: Responder) -> tuple[Verifier, FakeTransport]:
    transport = FakeTransport(responder)
    client = CachingClient(
        transport, ResponseCache(tmp_path / "c.sqlite"), CostMeter({"m": PRICE}, 1.0), "minimal"
    )
    return Verifier(client, load_prompt("verify"), "m", ("medium", 4000)), transport


# ------------------------------------------------------------------------------- the verdict


def test_a_verdict_must_judge_every_claim_exactly_once() -> None:
    entries = [ClaimVerdict(claim=1, supported=True), ClaimVerdict(claim=2, supported=False)]
    ok = Verdict(answers_question=True, claims=entries)
    assert validate_verdict(ok, 2) == []
    missing = validate_verdict(Verdict(answers_question=True, claims=[]), 2)
    assert [p.detail for p in missing] == ["claim 1 was not judged", "claim 2 was not judged"]
    twice = Verdict(answers_question=True, claims=[ClaimVerdict(claim=1, supported=True)] * 2)
    assert "claim 1 was judged 2 times" in [p.detail for p in validate_verdict(twice, 1)]
    extra = Verdict(answers_question=True, claims=[ClaimVerdict(claim=3, supported=True)])
    details = [p.detail for p in validate_verdict(extra, 1)]
    assert details == ["claim 3 does not exist (the draft has 1)", "claim 1 was not judged"]


def test_a_verdict_refuses_drops_or_keeps() -> None:
    claims = (claim("a", 1), claim("b", 2), claim("c", 3))

    def v(answers: bool, *flags: bool) -> Verdict:
        entries = [ClaimVerdict(claim=n, supported=f) for n, f in enumerate(flags, 1)]
        return Verdict(answers_question=answers, claims=entries)

    assert apply_verdict(v(False, True, True, True), claims) == (Outcome.REFUSED_VERIFIER, ())
    assert apply_verdict(v(True, True, False, True), claims) == (
        Outcome.ANSWERED,
        (claims[0], claims[2]),
    )
    assert apply_verdict(v(True, False, False, False), claims) == (Outcome.REFUSED_UNSUPPORTED, ())
    assert apply_verdict(v(True, True, True, True), claims) == (Outcome.ANSWERED, claims)


def test_the_verifier_sees_only_the_cited_chunks_relabelled_by_first_citation() -> None:
    claims = (claim("x", 3), claim("y", 3, 1))
    numbering, notes = cited_notes(claims, CHUNKS)
    assert numbering == {"nb.ipynb#3": "N1", "nb.ipynb#1": "N2"}
    assert notes == "[N1] nb.ipynb > Top\ngamma is c\n\n[N2] nb.ipynb > Top\nalpha is a"
    assert "beta" not in notes  # chunk 2 was in the answer context but never cited
    assert format_claims(claims, numbering) == "1. x (cites: N1)\n2. y (cites: N1, N2)"


# ------------------------------------------------------------------------------ the verifier


def test_stage_three_keeps_supported_claims_and_records_what_it_did(tmp_path: Path) -> None:
    verifier, transport = make(tmp_path, lambda _p: verdict(True, True, False, True))
    response = verifier.verify("q?", drafted(claim("a", 1), claim("b", 2), claim("c", 3)), CHUNKS)
    assert response.outcome is Outcome.ANSWERED
    assert [c.text for c in response.claims] == ["a", "c"]
    note = response.verification
    assert note is not None
    assert (note.answers_question, note.claims_in, note.claims_dropped, note.attempts) == (
        True, 3, 1, 1
    )  # fmt: skip
    assert note.usage.input_tokens > 0
    assert len(transport.prompts) == 1


def test_it_asks_with_the_higher_effort_and_its_own_cap(tmp_path: Path) -> None:
    verifier, transport = make(tmp_path, lambda _p: verdict(True, True))
    verifier.verify("q?", drafted(claim("a", 1)), CHUNKS)
    prompt = transport.prompts[0]
    assert (prompt.name, prompt.reasoning_effort, prompt.max_output_tokens) == (
        "verify",
        "medium",
        4000,
    )
    assert verifier.prompt_version == prompt.version
    assert "<question>\nq?\n</question>" in prompt.variable
    assert "1. a (cites: N1)" in prompt.variable
    assert "alpha is a" in prompt.variable
    assert "beta is b" not in prompt.variable


def test_a_question_the_notes_do_not_answer_is_refused_by_the_verifier(tmp_path: Path) -> None:
    verifier, _ = make(tmp_path, lambda _p: verdict(False, True))
    response = verifier.verify("q?", drafted(claim("a", 1)), CHUNKS)
    assert (response.outcome, response.claims) == (Outcome.REFUSED_VERIFIER, ())


def test_no_supported_claim_left_is_a_refusal(tmp_path: Path) -> None:
    verifier, _ = make(tmp_path, lambda _p: verdict(True, False, False))
    response = verifier.verify("q?", drafted(claim("a", 1), claim("b", 2)), CHUNKS)
    assert (response.outcome, response.claims) == (Outcome.REFUSED_UNSUPPORTED, ())
    assert response.verification is not None
    assert response.verification.claims_dropped == 2


def test_a_refused_response_is_passed_through_without_a_call(tmp_path: Path) -> None:
    verifier, transport = make(tmp_path, lambda _p: verdict(True, True))
    refused = Response(outcome=Outcome.REFUSED_MODEL, attempts=1)
    assert verifier.verify("q?", refused, CHUNKS) is refused
    assert transport.prompts == []


def test_an_incomplete_verdict_is_retried_with_the_problems_listed(tmp_path: Path) -> None:
    replies = iter([verdict(True, True), verdict(True, True, True)])  # the first skips claim 2
    verifier, transport = make(tmp_path, lambda _p: next(replies))
    response = verifier.verify("q?", drafted(claim("a", 1), claim("b", 2)), CHUNKS)
    assert response.outcome is Outcome.ANSWERED
    assert response.verification is not None
    assert response.verification.attempts == 2
    assert "claim 2 was not judged" in transport.prompts[1].variable
    assert "problems_with_your_previous_verdict" not in transport.prompts[0].variable


def test_a_verdict_that_stays_invalid_refuses_the_answer_fail_closed(tmp_path: Path) -> None:
    verifier, transport = make(tmp_path, lambda _p: verdict(True, True))  # always skips claim 2
    response = verifier.verify("q?", drafted(claim("a", 1), claim("b", 2)), CHUNKS)
    assert (response.outcome, response.claims) == (Outcome.REFUSED_VERIFIER_INVALID, ())
    assert len(transport.prompts) == 2
    assert response.verification is not None
    assert response.verification.answers_question is None


def test_unparsable_output_is_retried_and_then_refused(tmp_path: Path) -> None:
    verifier, transport = make(tmp_path, lambda _p: "not json")
    response = verifier.verify("q?", drafted(claim("a", 1)), CHUNKS)
    assert response.outcome is Outcome.REFUSED_VERIFIER_INVALID
    assert "unparsable:" in transport.prompts[1].variable


def test_the_question_is_scrubbed_before_it_reaches_the_verifier(tmp_path: Path) -> None:
    verifier, transport = make(tmp_path, lambda _p: verdict(True, True))
    verifier.verify(f"why {FAKE_AWS_ACCESS_KEY}?", drafted(claim("a", 1)), CHUNKS)
    assert FAKE_AWS_ACCESS_KEY not in transport.prompts[0].variable
    assert "[REDACTED" in transport.prompts[0].variable


def test_usage_and_latency_add_up_over_the_attempts(tmp_path: Path) -> None:
    replies = iter([verdict(True), verdict(True, True)])
    verifier, transport = make(tmp_path, lambda _p: next(replies))
    response = verifier.verify("q?", drafted(claim("a", 1)), CHUNKS)
    words = sum(len((p.instructions + " " + p.variable).split()) for p in transport.prompts)
    assert response.verification is not None
    assert response.verification.usage.input_tokens == words


def test_a_prompt_object_records_its_own_effort() -> None:
    template = load_prompt("verify")
    prompt: Prompt = template.render(
        "m", 10, reasoning_effort="high", question="q", claims="c", notes="n", feedback=""
    )
    assert prompt.reasoning_effort == "high"
    assert (
        template.render("m", 10, question="q", claims="c", notes="n", feedback="").reasoning_effort
        is None
    )
