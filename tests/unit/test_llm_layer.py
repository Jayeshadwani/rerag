"""Prompts, cache keys, the spend cap, the caching client and the OpenAI transport (offline)."""

import hashlib
from pathlib import Path

import httpx2
import openai
import pytest
import structlog
from pydantic import BaseModel, ConfigDict

from noterecall.config import PriceConfig
from noterecall.llm.cache import ResponseCache, SpendRow, Stored, cache_key
from noterecall.llm.client import (
    BudgetExceededError,
    LLMCallError,
    LLMOutputError,
    Prompt,
    Usage,
)
from noterecall.llm.cost import CostMeter, cost_usd, ledger_cost
from noterecall.llm.fake import FakeTransport, Responder, count_words
from noterecall.llm.openai_client import OpenAITransport, create_transport, scrub
from noterecall.llm.prompts import PROMPT_DIR, PromptFileError, load_prompt
from noterecall.llm.transport import CachingClient

PRICE = PriceConfig(input=0.25, cached_input=0.025, output=2.0)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context: str


def prompt(**changes: object) -> Prompt:
    base = Prompt(
        name="p",
        version="v1",
        model="m",
        instructions="i",
        shared="s",
        variable="v",
        max_output_tokens=50,
    )
    return base.model_copy(update=changes)


# ------------------------------------------------------------------------------------ prompts


def test_real_prompt_files_load_with_a_content_hash_version() -> None:
    blurb = load_prompt("contextual_blurb")
    raw = (PROMPT_DIR / "contextual_blurb.md").read_bytes()
    assert blurb.version == hashlib.sha256(raw).hexdigest()[:12]
    rendered = blurb.render("m", 50, notebook="NB-TEXT", chunk="CHUNK-TEXT")
    assert "NB-TEXT" in rendered.shared
    assert "CHUNK-TEXT" in rendered.variable
    assert "NB-TEXT" not in rendered.variable
    assert "CHUNK-TEXT" not in rendered.shared
    assert "{{" not in rendered.instructions + rendered.shared + rendered.variable
    assert rendered.instructions.startswith("You help a search index")
    clean = load_prompt("clean_markdown").render("m", 50, cell="CELL-TEXT")
    assert clean.shared == ""
    assert "CELL-TEXT" in clean.variable


def test_editing_a_prompt_changes_its_version(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("## instructions\nbe brief\n", encoding="utf-8")
    first = load_prompt("x", tmp_path).version
    (tmp_path / "x.md").write_text("## instructions\nbe very brief\n", encoding="utf-8")
    assert load_prompt("x", tmp_path).version != first


def test_a_prompt_file_needs_instructions_and_must_exist(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("## variable\nonly a tail\n", encoding="utf-8")
    with pytest.raises(PromptFileError, match="no '## instructions' section"):
        load_prompt("bad", tmp_path)
    with pytest.raises(PromptFileError, match="cannot read the prompt file"):
        load_prompt("missing", tmp_path)


def test_sections_are_split_on_their_headings_only(tmp_path: Path) -> None:
    text = "# title\n## instructions\nA\n## notes\nstill A\n## shared\nB\n## variable\nC\n"
    (tmp_path / "y.md").write_text(text, encoding="utf-8")
    parts = load_prompt("y", tmp_path).render("m", 5)
    assert (parts.instructions, parts.shared, parts.variable) == ("A\n## notes\nstill A", "B", "C")


# --------------------------------------------------------------------------------- cache keys


def test_cache_key_depends_on_every_input() -> None:
    base = cache_key(prompt(), "minimal")
    assert cache_key(prompt(), "minimal") == base
    changed = [
        cache_key(prompt(), "low"),
        cache_key(prompt(name="q"), "minimal"),
        cache_key(prompt(version="v2"), "minimal"),
        cache_key(prompt(model="n"), "minimal"),
        cache_key(prompt(max_output_tokens=51), "minimal"),
        cache_key(prompt(instructions="j"), "minimal"),
        cache_key(prompt(shared="t"), "minimal"),
        cache_key(prompt(variable="w"), "minimal"),
    ]
    assert len({base, *changed}) == 9


def test_cache_round_trip_first_answer_wins_and_spend_is_aggregated(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "sub" / "c.sqlite")
    assert cache.get("k") is None
    first = Stored(
        '{"context": "a"}', Usage(input_tokens=10, cached_tokens=4, output_tokens=3), 0.5
    )
    cache.put("k", prompt(), first)
    cache.put("k", prompt(), Stored("other", Usage(input_tokens=999), 9.0))  # ignored
    assert cache.get("k") == first
    cache.put(
        "k2",
        prompt(),
        Stored("{}", Usage(input_tokens=5, reasoning_tokens=2, output_tokens=2), 0.1),
    )
    cache.put("k3", prompt(name="q", model="n"), Stored("{}", Usage(input_tokens=1), 0.1))
    rows = cache.spend()
    assert rows == [
        SpendRow(
            "p",
            "m",
            2,
            Usage(input_tokens=15, cached_tokens=4, output_tokens=5, reasoning_tokens=2),
        ),
        SpendRow("q", "n", 1, Usage(input_tokens=1)),
    ]
    cache.close()
    assert ResponseCache(tmp_path / "sub" / "c.sqlite").get("k") == first  # persists


# -------------------------------------------------------------------------------------- cost


def test_cost_from_usage_is_hand_computed() -> None:
    usage = Usage(input_tokens=1000, cached_tokens=400, output_tokens=200, reasoning_tokens=50)
    # fresh 600 * 0.25 = 150; cached 400 * 0.025 = 10; output 200 * 2.0 = 400
    # (reasoning tokens are inside the output count) = 560 per million = 0.00056
    assert cost_usd(usage, PRICE) == pytest.approx(0.00056)
    rows = [SpendRow("p", "m", 1, usage), SpendRow("q", "m", 1, usage)]
    assert ledger_cost(rows, {"m": PRICE}) == pytest.approx(0.00112)
    total = Usage(input_tokens=1).plus(
        Usage(cached_tokens=2, cache_write_tokens=3, reasoning_tokens=4)
    )
    assert total == Usage(input_tokens=1, cached_tokens=2, cache_write_tokens=3, reasoning_tokens=4)


def test_meter_stops_new_calls_at_the_cap_and_counts_prior_spend() -> None:
    meter = CostMeter({"m": PRICE}, cap_usd=0.001, already_spent_usd=0.0004)
    meter.check()
    added = meter.add("m", Usage(input_tokens=1000, cached_tokens=400, output_tokens=200))
    assert added == pytest.approx(0.00056)
    assert meter.spent_usd == pytest.approx(0.00096)
    meter.check()  # still under 0.001
    meter.add("m", Usage(input_tokens=200))  # 200 * 0.25 / 1e6 = 0.00005 -> 0.00101 >= 0.001
    with pytest.raises(BudgetExceededError, match=r"spent \$0.0010 of the \$0.00 cap"):
        meter.check()


# ----------------------------------------------------------------------------- caching client


def echo(p: Prompt) -> str:
    return '{"context": "about ' + p.variable + '"}'


def make_client(
    tmp_path: Path,
    responder: Responder = echo,
    cap: float = 1.0,
    fail_after: int | None = None,
) -> tuple[CachingClient, FakeTransport, ResponseCache, CostMeter]:
    transport = FakeTransport(responder, fail_after)
    cache = ResponseCache(tmp_path / "c.sqlite")
    meter = CostMeter({"m": PRICE}, cap)
    return CachingClient(transport, cache, meter, "minimal"), transport, cache, meter


def test_a_miss_calls_the_transport_and_a_repeat_is_answered_from_the_cache(tmp_path: Path) -> None:
    client, transport, _cache, meter = make_client(tmp_path)
    first = client.complete_json(prompt(), Answer)
    assert first.parsed == Answer(context="about v")
    assert not first.from_cache
    assert first.usage == Usage(input_tokens=3, output_tokens=3)  # the fake counts words
    spent = meter.spent_usd
    second = client.complete_json(prompt(), Answer)
    assert second.from_cache
    assert second.parsed == first.parsed
    assert len(transport.prompts) == 1  # no second call
    assert meter.spent_usd == spent  # and nothing more was charged


def test_every_call_logs_what_the_rules_require(tmp_path: Path) -> None:
    client, _t, _c, _m = make_client(tmp_path)
    with structlog.testing.capture_logs() as logs:
        client.complete_json(prompt(), Answer)
        client.complete_json(prompt(), Answer)
    calls = [entry for entry in logs if entry["event"] == "llm.call"]
    assert [c["from_cache"] for c in calls] == [False, True]
    first = calls[0]
    assert first["model"] == "m"
    assert first["prompt_version"] == "v1"
    assert {
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "reasoning_tokens",
        "latency_s",
    } <= set(first)


def test_an_unusable_answer_is_raised_not_cached_and_not_charged(tmp_path: Path) -> None:
    client, transport, cache, meter = make_client(tmp_path, responder=lambda p: '{"wrong": 1}')
    with pytest.raises(LLMOutputError, match="does not match Answer"):
        client.complete_json(prompt(), Answer)
    assert cache.get(cache_key(prompt(), "minimal")) is None
    assert meter.spent_usd == 0.0
    with pytest.raises(LLMOutputError):
        client.complete_json(prompt(), Answer)
    assert len(transport.prompts) == 2  # asked again, because nothing was cached


def test_the_spend_cap_blocks_the_call_but_not_a_cache_hit(tmp_path: Path) -> None:
    client, transport, _c, _m = make_client(tmp_path, cap=1e-9)
    client.complete_json(prompt(), Answer)  # the first call is allowed: nothing spent yet
    with pytest.raises(BudgetExceededError):
        client.complete_json(prompt(variable="other"), Answer)
    assert client.complete_json(prompt(), Answer).from_cache  # cache hits are free
    assert len(transport.prompts) == 1


def test_fake_transport_reports_word_counts_and_can_be_told_to_fail(tmp_path: Path) -> None:
    assert count_words("a b  c") == 3
    client, _t, _c, _m = make_client(tmp_path, fail_after=1)
    client.complete_json(prompt(), Answer)
    with pytest.raises(LLMCallError, match="told to fail"):
        client.complete_json(prompt(variable="x"), Answer)


# ------------------------------------------------------------------------------ OpenAI transport


class FakeUsage:
    def __init__(self, i: int = 120, c: int = 100, w: int = 8, o: int = 30, r: int = 12) -> None:
        self.input_tokens = i
        self.input_tokens_details = type("D", (), {"cached_tokens": c, "cache_write_tokens": w})()
        self.output_tokens = o
        self.output_tokens_details = type("D", (), {"reasoning_tokens": r})()


class FakeResponse:
    def __init__(self, text: str = '{"context": "x"}', status: str | None = "completed",
                 usage: object = "default", reason: str | None = None) -> None:  # fmt: skip
        self.output_text = text
        self.status = status
        self.usage = FakeUsage() if usage == "default" else usage
        self.incomplete_details = type("I", (), {"reason": reason})() if reason else None


class FakeResponses:
    def __init__(self, *outcomes: object) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeResponse:
        self.requests.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, FakeResponse)
        return outcome


def http_error(
    cls: type[openai.APIStatusError], status: int, message: str
) -> openai.APIStatusError:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    return cls(message, response=httpx2.Response(status, request=request), body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(
        request=httpx2.Request("POST", "https://api.openai.com/v1/responses")
    )


def transport(responses: FakeResponses, retries: int = 3, secret: str | None = None):  # type: ignore[no-untyped-def]  # noqa: ANN201
    sleeps: list[float] = []
    return OpenAITransport(responses, "minimal", retries, sleeps.append, secret), sleeps  # type: ignore[arg-type]


def test_request_puts_the_shared_prefix_first_and_asks_for_strict_json() -> None:
    responses = FakeResponses(FakeResponse())
    t, _ = transport(responses)
    raw = t.complete_text(prompt(shared="NOTEBOOK", variable="CHUNK"), "Answer", {"type": "object"})
    (request,) = responses.requests
    assert request["model"] == "m"
    assert request["instructions"] == "i"
    assert request["input"] == "NOTEBOOK\n\nCHUNK"  # prefix first, tail last
    assert request["max_output_tokens"] == 50
    assert request["reasoning"] == {"effort": "minimal"}
    assert request["store"] is False  # the provider is asked not to keep the request
    assert request["prompt_cache_key"] == hashlib.sha256(b"NOTEBOOK").hexdigest()[:16]
    assert request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "Answer",
            "schema": {"type": "object"},
            "strict": True,
        }
    }
    assert raw.text == '{"context": "x"}'
    assert raw.usage == Usage(
        input_tokens=120,
        cached_tokens=100,
        cache_write_tokens=8,
        output_tokens=30,
        reasoning_tokens=12,
    )
    assert raw.attempts == 1


def test_a_prompt_without_a_shared_prefix_sends_only_the_tail() -> None:
    responses = FakeResponses(FakeResponse())
    transport(responses)[0].complete_text(prompt(shared="", variable="ONLY"), "A", {})
    assert responses.requests[0]["input"] == "ONLY"


def test_transient_errors_are_retried_with_exponential_backoff() -> None:
    errors = [
        connection_error(),
        http_error(openai.RateLimitError, 429, "slow down"),
        http_error(openai.InternalServerError, 500, "oops"),
    ]
    responses = FakeResponses(*errors, FakeResponse())
    t, sleeps = transport(responses, retries=3)
    raw = t.complete_text(prompt(), "A", {})
    assert sleeps == [1.0, 2.0, 4.0]  # 2 ** 0, 2 ** 1, 2 ** 2
    assert raw.attempts == 4


def test_retries_are_capped_and_the_final_error_is_raised() -> None:
    responses = FakeResponses(*[http_error(openai.RateLimitError, 429, "no") for _ in range(3)])
    t, sleeps = transport(responses, retries=2)
    with pytest.raises(LLMCallError, match="RateLimitError after 3 attempts"):
        t.complete_text(prompt(), "A", {})
    assert sleeps == [1.0, 2.0]


def test_backoff_never_exceeds_the_ceiling() -> None:
    errors = [connection_error() for _ in range(8)]
    t, sleeps = transport(FakeResponses(*errors, FakeResponse()), retries=8)
    t.complete_text(prompt(), "A", {})
    assert max(sleeps) == 30.0
    assert sleeps[:6] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]


def test_authentication_failures_do_not_echo_the_provider_message() -> None:
    echoing = http_error(
        openai.AuthenticationError, 401, "Incorrect API key provided: sk-proj-abcd1234"
    )
    t, sleeps = transport(FakeResponses(echoing), secret="sk-proj-abcd1234")
    with pytest.raises(LLMCallError) as caught:
        t.complete_text(prompt(), "A", {})
    assert str(caught.value) == "authentication failed: check OPENAI_API_KEY"
    assert "sk-proj" not in str(caught.value)
    assert sleeps == []  # not retried


def test_other_client_errors_are_raised_without_retry_and_with_the_key_scrubbed() -> None:
    bad = http_error(openai.BadRequestError, 400, "Unsupported value for sk-secret-123: effort")
    t, sleeps = transport(FakeResponses(bad), secret="sk-secret-123")
    with pytest.raises(LLMCallError) as caught:
        t.complete_text(prompt(), "A", {})
    assert "HTTP 400" in str(caught.value)
    assert "<api-key>" in str(caught.value)
    assert "sk-secret-123" not in str(caught.value)
    assert sleeps == []
    assert scrub("nothing here", None) == "nothing here"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            FakeResponse(status="incomplete", reason="max_output_tokens"),
            "cut off .max_output_tokens.",
        ),
        (FakeResponse(status="failed"), "status was 'failed'"),
        (FakeResponse(text=""), "no text"),
        (FakeResponse(usage=None), "no usage record"),
    ],
)
def test_unusable_responses_are_raised(response: FakeResponse, message: str) -> None:
    t, _ = transport(FakeResponses(response))
    with pytest.raises(LLMOutputError, match=message):
        t.complete_text(prompt(), "A", {})


def test_a_response_without_a_status_is_accepted_if_it_has_text_and_usage() -> None:
    t, _ = transport(FakeResponses(FakeResponse(status=None)))
    assert t.complete_text(prompt(), "A", {}).text == '{"context": "x"}'


def test_incomplete_without_a_reason_still_fails() -> None:
    t, _ = transport(FakeResponses(FakeResponse(status="incomplete")))
    with pytest.raises(LLMOutputError, match=r"cut off \(None\)"):
        t.complete_text(prompt(), "A", {})


def test_create_transport_needs_the_key_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMCallError, match="OPENAI_API_KEY is not set"):
        create_transport("minimal", 5.0, 3)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    assert isinstance(create_transport("minimal", 5.0, 3), OpenAITransport)  # builds, no network
