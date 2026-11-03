"""Response cache and spend ledger: key = sha256(model, prompt version, input) (PLAN section 1).

Every successful API answer is stored once, keyed by a hash of the model, the reasoning effort,
the output cap, the prompt's name and version and its full text. A second request with the same
key is answered from the file and costs nothing; editing a prompt file changes its version and so
every key. The file lives under `data/` (private, never committed). Each row keeps the provider's
usage numbers, so the same table is the spend ledger: the cost report and the money cap are
computed from it.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import NamedTuple

from noterecall.llm.client import Prompt, Usage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    prompt_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    text TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    reasoning_tokens INTEGER NOT NULL,
    latency_s REAL NOT NULL
);
"""


class Stored(NamedTuple):
    """One cached answer and the usage of the call that produced it."""

    text: str
    usage: Usage
    latency_s: float


class SpendRow(NamedTuple):
    """Usage aggregated over the stored answers of one (prompt, model)."""

    prompt_name: str
    model: str
    calls: int
    usage: Usage


def cache_key(prompt: Prompt, effort: str) -> str:
    """sha256 over everything that can change the answer."""
    payload = [
        prompt.name,
        prompt.version,
        prompt.model,
        effort,
        prompt.max_output_tokens,
        prompt.instructions,
        prompt.shared,
        prompt.variable,
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class ResponseCache:
    """SQLite-backed cache. Safe to share between threads."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def get(self, key: str) -> Stored | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT text, input_tokens, cached_tokens, cache_write_tokens, output_tokens,"
                " reasoning_tokens, latency_s FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return Stored(row[0], Usage(**dict(zip(_USAGE_FIELDS, row[1:6], strict=True))), row[6])

    def put(self, key: str, prompt: Prompt, stored: Stored) -> None:
        """Store an answer; an existing key is left alone (the first answer wins)."""
        u = stored.usage
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    prompt.name,
                    prompt.version,
                    prompt.model,
                    stored.text,
                    u.input_tokens,
                    u.cached_tokens,
                    u.cache_write_tokens,
                    u.output_tokens,
                    u.reasoning_tokens,
                    stored.latency_s,
                ),
            )

    def spend(self) -> list[SpendRow]:
        """Usage per (prompt, model) over everything stored so far."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT prompt_name, model, count(*), sum(input_tokens), sum(cached_tokens),"
                " sum(cache_write_tokens), sum(output_tokens), sum(reasoning_tokens)"
                " FROM responses GROUP BY prompt_name, model ORDER BY prompt_name, model"
            ).fetchall()
        return [
            SpendRow(r[0], r[1], r[2], Usage(**dict(zip(_USAGE_FIELDS, r[3:8], strict=True))))
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_USAGE_FIELDS = (
    "input_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)
