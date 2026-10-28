"""The `live` marker is skipped by default and runs only with NOTERECALL_LIVE=1."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
LIVE_TEST = """
import pytest

@pytest.mark.live
def test_needs_network():
    assert True

def test_offline():
    assert True
"""


def _run(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, live: str | None
) -> pytest.RunResult:
    if live is None:
        monkeypatch.delenv("NOTERECALL_LIVE", raising=False)
    else:
        monkeypatch.setenv("NOTERECALL_LIVE", live)
    pytester.makeini("[pytest]\nmarkers = live: live test\n")
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(LIVE_TEST)
    return pytester.runpytest_inprocess("-p", "no:cacheprovider")


def test_live_tests_skipped_by_default(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(pytester, monkeypatch, None).assert_outcomes(passed=1, skipped=1)


def test_live_tests_skipped_for_other_values(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(pytester, monkeypatch, "0").assert_outcomes(passed=1, skipped=1)


def test_live_tests_run_when_enabled(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(pytester, monkeypatch, "1").assert_outcomes(passed=2)
