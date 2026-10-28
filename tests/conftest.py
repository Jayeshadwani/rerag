"""Shared pytest configuration: `live` tests run only when NOTERECALL_LIVE=1."""

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `live` unless NOTERECALL_LIVE=1 (no network in the default run)."""
    if os.environ.get("NOTERECALL_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test: set NOTERECALL_LIVE=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
