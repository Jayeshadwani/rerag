"""Secret scrubbing: every pattern fires, look-alikes survive, nothing leaks into counts."""

import pytest

from make_fixtures import FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY, FAKE_AWS_SECRET_KEY
from noterecall.ingest import secrets
from noterecall.ingest.secrets import patterns_fingerprint, scrub

# Secret-shaped values are assembled from pieces so this file never contains a contiguous
# token that a secret scanner would flag. All are made up.
SAMPLES = {
    "anthropic_api_key": "sk-" + "ant-api03-" + "Z" * 24,
    "openai_api_key": "sk-" + "proj-" + "Q" * 30,
    "aws_access_key_id": "AKIA" + "ABCDEFGHIJKLMNOP",
    "github_token": "gh" + "p_" + "A1" * 18,
    "slack_token": "xox" + "b-" + "1234567890-" + "abcdefghij",
    "google_api_key": "AI" + "za" + "B" * 35,
    "private_key": "-----BEGIN RSA PRIVATE"
    + " KEY-----\nMIIBFAKE\n-----END RSA PRIVATE"
    + " KEY-----",
}


@pytest.mark.parametrize(("name", "value"), sorted(SAMPLES.items()))
def test_each_pattern_redacts_its_secret(name: str, value: str) -> None:
    result = scrub(f"before {value} after")
    assert result.text == f"before [REDACTED:{name}] after"
    assert result.counts == {name: 1}


def test_github_fine_grained_token_is_redacted() -> None:
    value = "github" + "_pat_" + "x" * 30
    assert scrub(f"t={value}").text == "t=[REDACTED:github_token]"


def test_aws_secret_key_keeps_its_context_and_quotes() -> None:
    value = "w" * 40
    result = scrub(f"aws_secret_access_key = '{value}'")
    assert result.text == "aws_secret_access_key = '[REDACTED:aws_secret_access_key]'"
    assert result.counts == {"aws_secret_access_key": 1}


def test_bearer_token_keeps_the_scheme() -> None:
    token = "abcdefghij" * 3
    assert scrub(f"Authorization: Bearer {token}").text == (
        "Authorization: Bearer [REDACTED:bearer_token]"
    )


def test_unterminated_private_key_redacts_to_end_of_text() -> None:
    begin = "-----BEGIN PRIVATE" + " KEY-----"
    result = scrub(f"keep this\n{begin}\nMIIfake\nmore body")
    assert result.text == "keep this\n[REDACTED:private_key]"


def test_terminated_private_key_only_redacts_the_block() -> None:
    value = SAMPLES["private_key"]
    assert scrub(f"a\n{value}\nb").text == "a\n[REDACTED:private_key]\nb"


def test_anthropic_key_is_not_double_counted_as_openai() -> None:
    assert scrub(SAMPLES["anthropic_api_key"]).counts == {"anthropic_api_key": 1}


def test_multiple_secrets_are_counted_per_type() -> None:
    aws, slack = SAMPLES["aws_access_key_id"], SAMPLES["slack_token"]
    text = f"{aws} and {aws} and {slack}"
    assert scrub(text).counts == {"aws_access_key_id": 2, "slack_token": 1}


@pytest.mark.parametrize(
    "text",
    [
        "plain prose with no secrets",
        "commit 3f786850e387550fdab836ed7e6dc881de23001b",  # 40 hex chars, no AWS context
        "from sklearn.model_selection import train_test_split",
        "task-ant-colony optimisation",  # contains 'ant-' but not an sk- key
        "pip install sk-learn",
        "Bearer short",
        "AKIA123",
        "x" * 40,
    ],
)
def test_look_alikes_are_left_alone(text: str) -> None:
    result = scrub(text)
    assert result.text == text
    assert result.counts == {}


def test_scrub_is_idempotent() -> None:
    once = scrub(f"k={SAMPLES['github_token']}")
    twice = scrub(once.text)
    assert twice.text == once.text
    assert twice.counts == {}


def test_fixture_secrets_are_all_scrubbed() -> None:
    text = (
        f"a {FAKE_ANTHROPIC_KEY}\nid {FAKE_AWS_ACCESS_KEY}\n"
        f"aws_secret_access_key = '{FAKE_AWS_SECRET_KEY}'"
    )
    result = scrub(text)
    for value in (FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY, FAKE_AWS_SECRET_KEY):
        assert value not in result.text
    assert result.counts == {
        "anthropic_api_key": 1,
        "aws_access_key_id": 1,
        "aws_secret_access_key": 1,
    }


def test_fingerprint_is_stable_and_changes_with_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    before = patterns_fingerprint()
    assert before == patterns_fingerprint()
    monkeypatch.setattr(secrets, "PATTERNS", secrets.PATTERNS[:-1])
    assert patterns_fingerprint() != before
