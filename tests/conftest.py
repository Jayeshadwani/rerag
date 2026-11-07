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
from typer.testing import CliRunner

from make_fixtures import FIXTURE_DIR
from noterecall import cli
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import load_settings
from noterecall.index.embeddings import EmbeddingSpec, HashEmbedder
from noterecall.index.store import connect, notebook_cells
from noterecall.models import Cell
from noterecall.retrieval.rerank import OverlapReranker

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
abstention_path = "{tmp_path / "state" / "abstention.json"}"

[logging]
level = "INFO"

[cleaning]
rules_path = "{RULES_PATH}"
max_output_lines = 20

[tokenizer]
encoding = "tiny-bytes"

[chunking]
candidate_caps = [64, 128]

[llm]
provider = "openai"
model = "test-model"
reasoning_effort = "minimal"
timeout_s = 5.0
max_retries = 2
concurrency = 2
max_usd = 1.0
cache_path = "{tmp_path / "state" / "llm_cache.sqlite"}"

[llm.prices."test-model"]
input = 1.0
cached_input = 0.1
output = 4.0

[api]
artifacts_dir = "artifacts"
history_turns = 5
max_question_chars = 500
max_turn_chars = 3000
daily_limit = 15
rewrite_max_output_tokens = 200

[generation]
max_chunks = 2
max_output_tokens = 300
max_false_refusal_pct = 5

[retrieval]
candidate_depth = 5
fused_depth = 3
final_k = 2
budget_tokens = 100
sensitivity_budgets = [50, 200]
rrf_default = 60
rrf_challengers = [10, 30]
latency_budget_ms = 1000.0

[[embeddings.candidates]]
id = "small"
revision = "r1"
query_prefix = "q: "
passage_prefix = ""

[[embeddings.candidates]]
id = "base"
revision = "r2"
query_prefix = "q: "
passage_prefix = "p: "

[rerank.model]
id = "ranker"
revision = "r3"
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def notebooks(config: Path) -> list[tuple[str, list[Cell]]]:
    """The fixture corpus after `noterecall sync`, as stored cells."""
    assert CliRunner().invoke(cli.app, ["sync", "--config", str(config)]).exit_code == 0
    structlog.reset_defaults()  # the CLI pointed logging at a stream that is now closed
    conn = connect(load_settings(config).paths.db_path)
    try:
        return list(notebook_cells(conn))
    finally:
        conn.close()


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the real model loaders; returns the ids of the embedding models that were loaded."""
    loaded: list[str] = []

    def embedder(spec: EmbeddingSpec) -> HashEmbedder:
        loaded.append(spec.id)
        return HashEmbedder()

    monkeypatch.setattr(cli, "load_embedder", embedder)
    monkeypatch.setattr(cli, "load_cross_encoder", lambda _spec: OverlapReranker())
    return loaded


PRIVATE_GOLD_FILES = [
    Path(__file__).resolve().parents[1] / "data" / "rerag-data" / "gold" / name
    for name in ("gold.json", "abstention.json")
]


def _private_state() -> list[tuple[bool, int | None]]:
    return [(p.exists(), p.stat().st_mtime_ns if p.exists() else None) for p in PRIVATE_GOLD_FILES]


@pytest.fixture(scope="session", autouse=True)
def _private_gold_files_are_never_touched() -> Iterator[None]:
    """The owner's private gold files must not be created or changed by any test."""
    before = _private_state()
    yield
    assert _private_state() == before, "a test wrote to a real private gold file"
