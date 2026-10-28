"""Shared pytest configuration: `live` tests run only when NOTERECALL_LIVE=1."""

import os

import pytest
import tiktoken

from noterecall.chunking.tokens import Tokenizer


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
