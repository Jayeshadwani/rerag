"""The citation validator: every kind of problem, the valid cases, and an exact property."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from noterecall.generate.citations import (
    Answer,
    Claim,
    ProblemKind,
    validate_answer,
)

IDS = {"C1", "C2", "C3"}


def answer(*claims: tuple[str, list[str]], insufficient: bool = False) -> Answer:
    return Answer(insufficient=insufficient, claims=[Claim(text=t, citations=c) for t, c in claims])


def kinds(problems: list) -> list[ProblemKind]:  # type: ignore[type-arg]
    return [p.kind for p in problems]


def test_a_valid_answer_has_no_problems() -> None:
    assert validate_answer(answer(("a", ["C1"]), ("b", ["C2", "C3"])), IDS) == []


def test_an_unknown_id_is_rejected_and_named() -> None:
    problems = validate_answer(answer(("a", ["C1"]), ("b", ["C9"])), IDS)
    assert kinds(problems) == [ProblemKind.UNKNOWN_ID]
    assert "claim 2" in problems[0].detail
    assert "'C9'" in problems[0].detail
    assert problems[0].message().startswith("unknown_id:")


def test_a_claim_without_citations_is_rejected() -> None:
    problems = validate_answer(answer(("a", ["C1"]), ("b", [])), IDS)
    assert kinds(problems) == [ProblemKind.EMPTY_CITATIONS]
    assert "claim 2" in problems[0].detail


def test_an_answer_with_no_claims_and_not_insufficient_is_rejected() -> None:
    assert kinds(validate_answer(answer(), IDS)) == [ProblemKind.NO_CLAIMS]


def test_insufficient_with_no_claims_is_a_valid_refusal() -> None:
    assert validate_answer(answer(insufficient=True), IDS) == []


def test_insufficient_with_claims_is_rejected() -> None:
    problems = validate_answer(answer(("a", ["C1"]), insufficient=True), IDS)
    assert kinds(problems) == [ProblemKind.CLAIMS_WHEN_INSUFFICIENT]


def test_every_problem_is_reported_not_just_the_first() -> None:
    problems = validate_answer(answer(("a", []), ("b", ["C7", "C1"]), ("c", ["C8"])), IDS)
    assert kinds(problems) == [
        ProblemKind.EMPTY_CITATIONS,
        ProblemKind.UNKNOWN_ID,
        ProblemKind.UNKNOWN_ID,
    ]


def test_a_context_of_no_chunks_accepts_only_insufficient() -> None:
    assert kinds(validate_answer(answer(("a", ["C1"])), set())) == [ProblemKind.UNKNOWN_ID]
    assert validate_answer(answer(insufficient=True), set()) == []


def test_an_empty_claim_text_does_not_parse() -> None:
    with pytest.raises(ValueError, match="at least 1 character"):
        Claim(text="", citations=["C1"])


id_strategy = st.sampled_from(["C1", "C2", "C3", "C4", "X"])


@given(
    st.lists(
        st.tuples(st.text(min_size=1), st.lists(id_strategy, max_size=3)), min_size=1, max_size=4
    )
)
def test_a_non_insufficient_answer_is_valid_exactly_when_all_claims_cite_known_ids(
    claims: list[tuple[str, list[str]]],
) -> None:
    expected = all(c and all(i in IDS for i in c) for _, c in claims)
    assert (validate_answer(answer(*claims), IDS) == []) is expected
