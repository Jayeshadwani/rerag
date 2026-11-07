"""The Streamlit page, driven with Streamlit's own test harness and a stand-in for the API."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

import streamlit_app

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")
ANSWERED = {
    "status": "answered",
    "refusal_reason": None,
    "standalone_question": "q",
    "claims": [{"text": "Alpha is first.", "citations": ["C1"]}],
    "sources": [
        {
            "id": "C1",
            "notebook": "nb/a.ipynb",
            "heading_path": ["Intro", "Alpha"],
            "cell_index": 3,
            "text": "alpha cell text",
        }
    ],
}
REFUSED = {**ANSWERED, "status": "refused", "refusal_reason": "not_in_notes", "claims": [],
           "sources": []}  # fmt: skip


class FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code, self._body = status, body

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("not json")
        return self._body


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_URL", "https://api.example/")
    monkeypatch.setenv("UI_API_SECRET", "shh")


def patch_post(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[..., FakeResponse]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return handler(url, **kwargs)

    monkeypatch.setattr(requests, "post", post)
    return calls


def ask(at: AppTest, question: str) -> AppTest:
    return at.chat_input[0].set_value(question).run()


def test_answer_shows_claims_tags_and_source_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_post(monkeypatch, lambda *_a, **_k: FakeResponse(200, ANSWERED))
    at = ask(AppTest.from_file(APP).run(), "what is alpha?")
    assert not at.exception
    assert any("Alpha is first." in m.value and "[C1]" in m.value for m in at.markdown)
    assert "[C1] nb/a.ipynb > Intro > Alpha (cell 3)" in [e.label for e in at.expander]
    assert at.expander[0].text[0].value == "alpha cell text"
    assert calls[0]["url"] == "https://api.example/ask"
    assert calls[0]["json"] == {"question": "what is alpha?", "history": []}
    assert calls[0]["headers"]["X-UI-Secret"] == "shh"


def test_history_is_sent_and_capped_at_five_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_post(monkeypatch, lambda *_a, **_k: FakeResponse(200, ANSWERED))
    at = AppTest.from_file(APP).run()
    for n in range(7):
        at = ask(at, f"question {n}")
    history = calls[-1]["json"]["history"]
    assert [t["question"] for t in history] == [f"question {n}" for n in range(1, 6)]
    assert history[0]["answer"] == "Alpha is first. [C1]"
    assert len(at.session_state["turns"]) == 5


def test_refusal_message(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_post(monkeypatch, lambda *_a, **_k: FakeResponse(200, REFUSED))
    at = ask(AppTest.from_file(APP).run(), "what is zeta?")
    assert [i.value for i in at.info] == ["This is not in your notes."]
    assert len(at.expander) == 0


def test_rate_limit_message_and_the_question_is_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"detail": "daily limit reached", "retry_after_s": 60}
    patch_post(monkeypatch, lambda *_a, **_k: FakeResponse(429, body))
    at = ask(AppTest.from_file(APP).run(), "again")
    assert [w.value for w in at.warning] == [streamlit_app.LIMIT]
    assert "daily limit reached" in streamlit_app.LIMIT.lower()
    assert at.session_state["turns"] == []


@pytest.mark.parametrize(
    "outcome",
    [requests.ConnectionError("down"), requests.Timeout("slow"), FakeResponse(503, None)],
)
def test_waking_message(monkeypatch: pytest.MonkeyPatch, outcome: Any) -> None:
    def handler(*_a: Any, **_k: Any) -> FakeResponse:
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome

    patch_post(monkeypatch, handler)
    at = ask(AppTest.from_file(APP).run(), "hello")
    assert [w.value for w in at.warning] == [streamlit_app.WAKING]


def test_api_error_message_does_not_pretend_it_is_waking(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_post(
        monkeypatch,
        lambda *_a, **_k: FakeResponse(502, {"detail": "the language model is unavailable"}),
    )
    at = ask(AppTest.from_file(APP).run(), "hello")
    assert len(at.error) == 1
    assert "the language model is unavailable" in at.error[0].value


def test_missing_configuration_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UI_API_SECRET")
    at = AppTest.from_file(APP).run()
    assert [e.value for e in at.error] == ["Not configured: set UI_API_SECRET."]


def test_history_payload_keeps_the_last_turns_only() -> None:
    turns = [{"question": f"q{i}", "text": f"a{i}"} for i in range(8)]
    payload = streamlit_app.history_payload(turns)
    assert [t["question"] for t in payload] == ["q3", "q4", "q5", "q6", "q7"]


def test_the_visitor_ip_comes_from_the_first_forwarded_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Context:
        headers: ClassVar[dict[str, str]] = {"X-Forwarded-For": "203.0.113.4, 10.1.1.1"}
        ip_address = "10.9.9.9"

    monkeypatch.setattr(st, "context", Context())
    assert streamlit_app.client_ip() == "203.0.113.4"
    Context.headers = {}
    assert streamlit_app.client_ip() == "10.9.9.9"


def test_the_visitor_ip_is_sent_with_the_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_post(monkeypatch, lambda *_a, **_k: FakeResponse(200, ANSWERED))
    streamlit_app.ask_api("q", [], "203.0.113.4")
    assert calls[0]["headers"] == {"X-UI-Secret": "shh", "X-UI-Client-IP": "203.0.113.4"}
    streamlit_app.ask_api("q", [], None)
    assert "X-UI-Client-IP" not in calls[1]["headers"]
