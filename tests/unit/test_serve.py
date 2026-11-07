"""Artifacts, the rewrite, the QA service and the HTTP API, all offline (proposal 009)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from make_fixtures import FAKE_ANTHROPIC_KEY
from noterecall.api import UI_IP_HEADER, UI_SECRET_HEADER, create_app
from noterecall.api_models import AskRequest, AskResponse, RefusalReason, Status, Turn
from noterecall.config import ApiConfig, load_settings
from noterecall.generate.answer import CitedClaim, Outcome, Response
from noterecall.generate.ask import AskResult
from noterecall.generate.rewrite import RewriteError, Rewriter
from noterecall.llm.cache import ResponseCache
from noterecall.llm.client import BudgetExceededError, LLMCallError, LLMClient
from noterecall.llm.cost import CostMeter
from noterecall.llm.fake import FakeTransport
from noterecall.llm.prompts import load_prompt
from noterecall.llm.transport import CachingClient
from noterecall.models import Chunk, Piece, PieceKind, PieceUnit
from noterecall.retrieval.pipeline import AbstentionSignals
from noterecall.serve.artifacts import (
    ArtifactError,
    Manifest,
    load_artifacts,
    write_artifacts,
)
from noterecall.serve.ratelimit import RateLimiter
from noterecall.serve.service import QAService, recent_turns, to_response
from noterecall.srs.clock import FixedClock

SECRET = "ui-secret"
LIMITS = ApiConfig(
    artifacts_dir=Path("x"),
    history_turns=2,
    max_question_chars=50,
    max_turn_chars=20,
    daily_limit=3,
    rewrite_max_output_tokens=50,
)
SIGNALS = AbstentionSignals(top_cosine=0.5, bm25_norm=1.0)


def chunk(cid: str, text: str, cell: int = 2) -> Chunk:
    piece = Piece(cell_index=cell, kind=PieceKind.MARKDOWN, part=0, unit=PieceUnit.CELL, text=text)
    return Chunk(
        id=cid, notebook_path="nb/a.ipynb", heading_path=("H1", "H2"), pieces=(piece,),
        text=text, tokens=3, cell_hashes=("h",),
    )  # fmt: skip


CHUNKS = {"a": chunk("a", "alpha text"), "b": chunk("b", "beta text", cell=5)}


def answered(*claims: tuple[str, tuple[str, ...]]) -> AskResult:
    cited = tuple(CitedClaim(text=t, chunk_ids=ids) for t, ids in claims)
    return AskResult(
        response=Response(outcome=Outcome.ANSWERED, claims=cited),
        signals=SIGNALS,
        context_ids=("a", "b"),
    )


def refused(outcome: Outcome) -> AskResult:
    return AskResult(response=Response(outcome=outcome), signals=SIGNALS, context_ids=("a",))


class StubAsker:
    def __init__(self, result: AskResult) -> None:
        self.result, self.questions = result, []  # type: ignore[var-annotated]

    def ask(self, question: str) -> AskResult:
        self.questions.append(question)
        return self.result


def make_rewriter(tmp_path: Path, reply: str) -> tuple[Rewriter, FakeTransport]:
    settings = load_settings()
    transport = FakeTransport(lambda _p: reply)
    meter = CostMeter(settings.llm.prices, settings.llm.max_usd)
    client: LLMClient = CachingClient(
        transport, ResponseCache(tmp_path / "c.sqlite"), meter, settings.llm.reasoning_effort
    )
    template = load_prompt("rewrite_question")
    return Rewriter(client, template, settings.llm.model, (50, 60, 40)), transport


# --- artifacts ---


def manifest(n: int, dim: int) -> Manifest:
    return Manifest(
        schema_version=1, model="m", revision="r", runtime="onnxruntime",
        chunk_cap=512, header=True, chunks=n, dim=dim,
    )  # fmt: skip


def unit_matrix() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def test_artifacts_round_trip(tmp_path: Path) -> None:
    write_artifacts(tmp_path / "out", list(CHUNKS.values()), unit_matrix(), manifest(2, 2))
    loaded = load_artifacts(tmp_path / "out")
    assert [c.id for c in loaded.chunks] == ["a", "b"]
    assert loaded.chunks[0] == CHUNKS["a"]
    assert np.array_equal(loaded.matrix, unit_matrix())
    assert loaded.manifest.model == "m"


def test_artifacts_reject_disagreeing_sizes(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="disagree"):
        write_artifacts(tmp_path, list(CHUNKS.values()), unit_matrix(), manifest(3, 2))


def test_artifacts_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="build-index"):
        load_artifacts(tmp_path)


def test_artifacts_reject_tampered_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_artifacts(out, list(CHUNKS.values()), unit_matrix(), manifest(2, 2))
    np.save(out / "embeddings.npy", np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    with pytest.raises(ArtifactError, match="unit length"):
        load_artifacts(out)
    np.save(out / "embeddings.npy", np.ones((3, 2), dtype=np.float32))
    with pytest.raises(ArtifactError, match="disagree"):
        load_artifacts(out)
    np.save(out / "embeddings.npy", unit_matrix().astype(np.float64))
    with pytest.raises(ArtifactError, match="float32"):
        load_artifacts(out)
    write_artifacts(out, list(CHUNKS.values()), unit_matrix(), manifest(2, 2))
    line = (out / "chunks.jsonl").read_text().splitlines()[0]
    (out / "chunks.jsonl").write_text(line + "\n" + line + "\n")
    with pytest.raises(ArtifactError, match="share an id"):
        load_artifacts(out)


# --- rewrite ---


def turns(n: int) -> list[Turn]:
    return [Turn(question=f"q{i}", answer=f"a{i}") for i in range(n)]


def test_rewrite_returns_the_standalone_question(tmp_path: Path) -> None:
    rewriter, transport = make_rewriter(tmp_path, json.dumps({"question": " What is X? "}))
    assert rewriter.rewrite("and why?", turns(1)) == "What is X?"
    assert "and why?" in transport.prompts[0].variable
    assert "Question: q0" in transport.prompts[0].variable


def test_rewrite_scrubs_secrets_before_the_call(tmp_path: Path) -> None:
    rewriter, transport = make_rewriter(tmp_path, json.dumps({"question": "ok?"}))
    history = [Turn(question=f"my key {FAKE_ANTHROPIC_KEY}", answer="fine")]
    rewriter.rewrite(f"what about {FAKE_ANTHROPIC_KEY}?", history)
    sent = transport.prompts[0]
    assert FAKE_ANTHROPIC_KEY not in sent.variable + sent.shared + sent.instructions


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (json.dumps({"question": "  "}), "empty"),
        (json.dumps({"question": "x" * 61}), "longer"),
        ("not json", "unusable"),
    ],
)
def test_rewrite_failures_are_loud(tmp_path: Path, reply: str, message: str) -> None:
    rewriter, _ = make_rewriter(tmp_path, reply)
    with pytest.raises(RewriteError, match=message):
        rewriter.rewrite("and why?", turns(1))


def test_rewrite_needs_history(tmp_path: Path) -> None:
    rewriter, transport = make_rewriter(tmp_path, json.dumps({"question": "q?"}))
    with pytest.raises(RewriteError, match="no history"):
        rewriter.rewrite("hi", [])
    assert transport.prompts == []


# --- service ---


def test_recent_turns_keeps_the_last_and_cuts_text() -> None:
    long = [Turn(question="q" * 30, answer="a" * 30), *turns(2)]
    kept = recent_turns(long, 2, 20)
    assert [t.question for t in kept] == ["q0", "q1"]
    assert recent_turns([Turn(question="q" * 30, answer="a" * 30)], 5, 20)[0].question == "q" * 20
    assert recent_turns(turns(3), 0, 20) == []


def test_to_response_maps_context_positions_to_c_labels() -> None:
    result = answered(("one", ("b",)), ("two", ("a", "b")))
    response = to_response("sq", result, CHUNKS)
    assert response.status is Status.ANSWERED
    assert response.refusal_reason is None
    assert [c.citations for c in response.claims] == [["C2"], ["C1", "C2"]]
    assert [s.id for s in response.sources] == ["C1", "C2"]  # cited sources, in label order
    source = response.sources[1]
    assert (source.notebook, source.cell_index, source.text) == ("nb/a.ipynb", 5, "beta text")
    assert source.heading_path == ["H1", "H2"]


def test_to_response_returns_only_cited_sources() -> None:
    response = to_response("sq", answered(("one", ("b",))), CHUNKS)
    assert [s.id for s in response.sources] == ["C2"]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (Outcome.REFUSED_MODEL, RefusalReason.NOT_IN_NOTES),
        (Outcome.REFUSED_INVALID, RefusalReason.INVALID_ANSWER),
        (Outcome.REFUSED_STAGE1, RefusalReason.BELOW_THRESHOLD),
    ],
)
def test_refusals_carry_a_reason_and_no_sources(outcome: Outcome, reason: RefusalReason) -> None:
    response = to_response("sq", refused(outcome), CHUNKS)
    assert response.status is Status.REFUSED
    assert response.refusal_reason is reason
    assert response.claims == []
    assert response.sources == []


def test_no_history_means_no_rewrite_call(tmp_path: Path) -> None:
    rewriter, transport = make_rewriter(tmp_path, json.dumps({"question": "rewritten?"}))
    asker = StubAsker(answered(("one", ("a",))))
    service = QAService(asker, rewriter, CHUNKS, LIMITS)  # type: ignore[arg-type]
    response = service.ask(AskRequest(question="  what is a?  "))
    assert transport.prompts == []
    assert asker.questions == ["what is a?"]
    assert response.standalone_question == "what is a?"


def test_history_triggers_one_rewrite_and_the_asker_gets_only_the_standalone(
    tmp_path: Path,
) -> None:
    rewriter, transport = make_rewriter(tmp_path, json.dumps({"question": "Why is a so?"}))
    asker = StubAsker(answered(("one", ("a",))))
    service = QAService(asker, rewriter, CHUNKS, LIMITS)  # type: ignore[arg-type]
    request = AskRequest(question="why?", history=turns(4))
    response = service.ask(request)
    assert len(transport.prompts) == 1
    assert "Turn 3" not in transport.prompts[0].variable  # history_turns = 2: turns 1..2 only
    assert "Question: q3" in transport.prompts[0].variable
    assert asker.questions == ["Why is a so?"]  # history never reaches retrieval or the answer
    assert response.standalone_question == "Why is a so?"


# --- API ---


class Raising:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def ask(self, request: AskRequest) -> AskResponse:
        raise self.exc


class Fixed:
    def __init__(self) -> None:
        self.requests: list[AskRequest] = []

    def ask(self, request: AskRequest) -> AskResponse:
        self.requests.append(request)
        return to_response("sq", answered(("one", ("a",))), CHUNKS)


def client_for(service: object, clock: FixedClock | None = None) -> TestClient:
    limiter = RateLimiter(LIMITS.daily_limit, clock or FixedClock(datetime(2026, 3, 1, tzinfo=UTC)))
    app = create_app(service, limiter, SECRET, LIMITS, {"chunks": 2})  # type: ignore[arg-type]
    return TestClient(app)


def test_health() -> None:
    assert client_for(Fixed()).get("/health").json() == {"status": "ok", "chunks": 2}


def test_ask_answers() -> None:
    service = Fixed()
    reply = client_for(service).post("/ask", json={"question": " hello ", "history": []})
    body = reply.json()
    assert reply.status_code == 200
    assert body["status"] == "answered"
    assert body["claims"] == [{"text": "one", "citations": ["C1"]}]
    assert body["sources"][0]["id"] == "C1"
    assert service.requests[0].question == "hello"


def test_empty_and_long_questions_are_rejected_before_counting() -> None:
    client = client_for(Fixed())
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "q" * 51}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
    for _ in range(3):  # the rejected ones did not use up the 3 allowed
        assert client.post("/ask", json={"question": "ok"}).status_code == 200


def test_the_limit_returns_429_with_retry_after() -> None:
    client = client_for(Fixed())
    for _ in range(3):
        assert client.post("/ask", json={"question": "ok"}).status_code == 200
    reply = client.post("/ask", json={"question": "ok"})
    assert reply.status_code == 429
    assert reply.json()["detail"] == "daily limit reached"
    assert int(reply.headers["Retry-After"]) == reply.json()["retry_after_s"] == 86400


def test_day_rollover_resets_the_count() -> None:
    clock = FixedClock(datetime(2026, 3, 1, 12, tzinfo=UTC))
    client = client_for(Fixed(), clock)
    for _ in range(4):
        client.post("/ask", json={"question": "ok"})
    assert client.post("/ask", json={"question": "ok"}).status_code == 429
    clock.advance(timedelta(hours=12))
    assert client.post("/ask", json={"question": "ok"}).status_code == 200


def ask_as(client: TestClient, ip: str, secret: str | None) -> int:
    headers = {"X-Forwarded-For": "198.51.100.7", UI_IP_HEADER: ip}
    if secret is not None:
        headers[UI_SECRET_HEADER] = secret
    return client.post("/ask", json={"question": "ok"}, headers=headers).status_code


def test_forwarded_ip_with_the_secret_is_counted_per_end_user() -> None:
    client = client_for(Fixed())
    for _ in range(3):
        assert ask_as(client, "203.0.113.1", SECRET) == 200
    assert ask_as(client, "203.0.113.1", SECRET) == 429
    assert ask_as(client, "203.0.113.2", SECRET) == 200  # another end user behind the same UI


def test_spoofing_the_forwarded_ip_without_the_secret_does_not_help() -> None:
    client = client_for(Fixed())
    codes = [ask_as(client, f"203.0.113.{n}", None) for n in range(1, 6)]
    assert codes == [200, 200, 200, 429, 429]  # all charged to the real caller 198.51.100.7
    wrong = [ask_as(client, f"203.0.113.{n}", "wrong") for n in range(6, 8)]
    assert wrong == [429, 429]


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (RewriteError("x"), 502),
        (BudgetExceededError("x"), 503),
        (LLMCallError("x"), 502),
    ],
)
def test_model_failures_map_to_5xx_without_details(exc: Exception, code: int) -> None:
    reply = client_for(Raising(exc)).post("/ask", json={"question": "ok"})
    assert reply.status_code == code
    assert set(reply.json()) == {"detail", "retry_after_s"}
    assert reply.json()["detail"] != "x"  # the exception text is not passed on
