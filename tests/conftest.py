"""Shared pytest configuration: `live` tests run only when NOTERECALL_LIVE=1.

Also the fixtures the command-line tests share: an offline tokenizer and a synthetic corpus config.
"""

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
import tiktoken
from hypothesis import settings

from make_fixtures import FIXTURE_DIR
from noterecall import cli
from noterecall.chunking.tokens import Tokenizer

RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "cleaning_rules.yaml"

# Deterministic property tests: the same examples every run (ADR-015 reproducibility).
settings.register_profile("noterecall", derandomize=True, max_examples=200, deadline=None)
settings.load_profile("noterecall")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `live` unless NOTERECALL_LIVE=1 (no network in the default run)."""
    if os.environ.get("NOTERECALL_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test: set NOTERECALL_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def tiny_tokenizer() -> Tokenizer:
    """Offline byte-level tokenizer: one token per UTF-8 byte (hand-computable counts)."""
    encoding = tiktoken.Encoding(
        "tiny-bytes",
        pat_str=r"[\s\S]",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )
    return Tokenizer(encoding)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch, tiny_tokenizer: Tokenizer) -> Iterator[None]:
    for name in ("NOTERECALL_LOGGING__LEVEL", "NOTERECALL_PATHS__RAW_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_tokenizer", lambda _name: tiny_tokenizer)
    yield
    structlog.reset_defaults()


@pytest.fixture
def config(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    shutil.copytree(FIXTURE_DIR, raw)
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[paths]
raw_dir = "{raw}"
db_path = "{tmp_path / "state" / "notes.sqlite"}"
results_dir = "{tmp_path / "results"}"
gold_path = "{tmp_path / "state" / "gold.json"}"

[logging]
level = "INFO"

[cleaning]
rules_path = "{RULES_PATH}"
max_output_lines = 20

[tokenizer]
encoding = "tiny-bytes"

[chunking]
candidate_caps = [64, 128]
""",
        encoding="utf-8",
    )
    return path
