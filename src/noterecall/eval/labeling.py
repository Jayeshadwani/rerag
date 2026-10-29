"""Search and session logic behind `noterecall label`. Implements ADR-009 (human-written labels).

The tool helps the owner find and mark relevant cells; it never decides relevance. Search is
deliberately simple and independent of the retrievers under test (word-token TF-IDF, a substring
grep, and browsing a notebook in order), because labels pooled from the systems being evaluated
would be biased toward them. All state for one query lives in `QuerySession`; its `handle`
method takes one line of input and returns the text to show, so the whole dialogue is testable
without a terminal.
"""

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import NamedTuple

from noterecall.chunking.structural import build_sections
from noterecall.eval.gold import CellKey
from noterecall.models import Cell

_WORD = re.compile(r"\w+")
PAGE = 8
SNIPPET_CHARS = 400


class Entry(NamedTuple):
    """One distinct cell (by notebook and raw hash) the owner can label."""

    notebook_path: str
    cell_index: int
    cell_type: str
    raw_hash: str
    heading_path: tuple[str, ...]
    source: str
    same_notebook_copies: int
    other_notebooks: int

    @property
    def key(self) -> CellKey:
        return (self.notebook_path, self.raw_hash)


def words(text: str) -> list[str]:
    """Lowercase word tokens."""
    return _WORD.findall(text.lower())


def build_entries(notebooks: Sequence[tuple[str, list[Cell]]]) -> list[Entry]:
    """Distinct non-blank cells in notebook and cell order, with their heading paths."""
    holders: dict[str, set[str]] = {}
    for path, cells in notebooks:
        for cell in cells:
            holders.setdefault(cell.raw_hash, set()).add(path)
    entries: list[Entry] = []
    for path, cells in notebooks:
        paths = {c.index: s.path for s in build_sections(cells) for c in s.cells}
        counts = Counter(c.raw_hash for c in cells)
        seen: set[str] = set()
        for cell in cells:
            if not cell.source.strip() or cell.raw_hash in seen:
                continue
            seen.add(cell.raw_hash)
            entries.append(
                Entry(
                    path,
                    cell.index,
                    cell.cell_type.value,
                    cell.raw_hash,
                    paths[cell.index],
                    cell.source,
                    counts[cell.raw_hash],
                    len(holders[cell.raw_hash]) - 1,
                )
            )
    return entries


def search(entries: Sequence[Entry], query: str, limit: int = PAGE) -> list[Entry]:
    """TF-IDF ranking of cells for the query's words; cells with no matching word are omitted."""
    terms = set(words(query))
    docs = [Counter(words(e.source)) for e in entries]
    scored: list[tuple[float, int]] = []
    for term in terms:
        df = sum(1 for d in docs if term in d)
        if df == 0:
            continue
        idf = math.log(1 + len(entries) / df)
        scored.extend((idf * (1 + math.log(d[term])), i) for i, d in enumerate(docs) if term in d)
    totals: dict[int, float] = {}
    for score, i in scored:
        totals[i] = totals.get(i, 0.0) + score
    ranked = sorted(
        totals, key=lambda i: (-totals[i], entries[i].notebook_path, entries[i].cell_index)
    )
    return [entries[i] for i in ranked[:limit]]


def grep(entries: Sequence[Entry], needle: str, limit: int = PAGE) -> list[Entry]:
    """Cells whose source contains the text, case-insensitively, in corpus order."""
    lowered = needle.lower()
    return [e for e in entries if lowered in e.source.lower()][:limit]


def render_entry(entry: Entry, number: int, marked: bool) -> str:
    """Show one cell: number, mark, location, heading path and a trimmed source."""
    heading = " > ".join(entry.heading_path) or "(no heading)"
    body = entry.source.strip()
    if len(body) > SNIPPET_CHARS:
        body = body[:SNIPPET_CHARS] + " ..."
    notes = []
    if entry.same_notebook_copies > 1:
        notes.append(f"{entry.same_notebook_copies} identical cells here")
    if entry.other_notebooks:
        notes.append(f"also in {entry.other_notebooks} other notebook(s)")
    note = f" [{'; '.join(notes)}]" if notes else ""
    flag = "[x]" if marked else "[ ]"
    head = (
        f"{flag} {number}. {entry.notebook_path} cell {entry.cell_index} ({entry.cell_type}){note}"
    )
    return f"{head}\n    {heading}\n{_indent(body)}"


def _indent(text: str) -> str:
    return "\n".join("    | " + line for line in text.splitlines())


class QuerySession:
    """The labelling dialogue for one query. `handle(line)` returns the text to show."""

    HELP = (
        "words: search | /grep TEXT | /nb [N] list or browse notebooks | /more next page\n"
        "/mark N.. | /markall N.. (identical cells in every notebook) | /unmark N.. | /marked\n"
        "/done save this query | /abort discard it | /help"
    )

    def __init__(
        self,
        entries: Sequence[Entry],
        query_text: str,
        unanswerable: bool,
        initial: Sequence[CellKey] = (),
    ) -> None:
        self.entries = list(entries)
        self.query_text = query_text
        self.unanswerable = unanswerable
        self.marked: set[CellKey] = set(initial)
        self.status: str = "open"
        self._listing: list[Entry] = []
        self._offset = 0
        self._page: list[Entry] = []
        self._confirm_empty = False
        self._notebooks = sorted({e.notebook_path for e in self.entries})

    def handle(self, line: str) -> str:
        """Run one input line."""
        line = line.strip()
        if not line:
            return self.HELP
        command, _, rest = line.partition(" ") if line.startswith("/") else ("", "", line)
        if command != "/done":
            self._confirm_empty = False
        actions: dict[str, Callable[[], str]] = {
            "/grep": lambda: self._show(grep(self.entries, rest, limit=10**9)),
            "/nb": lambda: self._notebook(rest),
            "/more": self._more,
            "/mark": lambda: self._mark(rest, all_copies=False),
            "/markall": lambda: self._mark(rest, all_copies=True),
            "/unmark": lambda: self._unmark(rest),
            "/marked": self._marked,
            "/done": self._done,
            "/abort": self._abort,
            "/help": lambda: self.HELP,
        }
        if command == "":
            return self._show(search(self.entries, rest, limit=10**9))
        if command not in actions:
            return f"unknown command {command}; /help lists the commands"
        return actions[command]()

    def _show(self, results: list[Entry]) -> str:
        self._listing, self._offset = results, 0
        return self._page_text() if results else "no cells match"

    def _page_text(self) -> str:
        self._page = self._listing[self._offset : self._offset + PAGE]
        lines = [
            render_entry(e, n, e.key in self.marked) for n, e in enumerate(self._page, start=1)
        ]
        left = len(self._listing) - self._offset - len(self._page)
        if left > 0:
            lines.append(f"... {left} more (/more)")
        return "\n".join(lines)

    def _more(self) -> str:
        if self._offset + PAGE >= len(self._listing):
            return "nothing more to show"
        self._offset += PAGE
        return self._page_text()

    def _notebook(self, rest: str) -> str:
        if not rest.strip():
            return "\n".join(f"{n}. {p}" for n, p in enumerate(self._notebooks, start=1))
        number = _number(rest, len(self._notebooks))
        if number is None:
            return "give a notebook number from /nb"
        path = self._notebooks[number - 1]
        return self._show([e for e in self.entries if e.notebook_path == path])

    def _pick(self, rest: str) -> list[Entry] | str:
        picked: list[Entry] = []
        for token in rest.split():
            number = _number(token, len(self._page))
            if number is None:
                return f"'{token}' is not a number on the current page"
            picked.append(self._page[number - 1])
        return picked or "give one or more numbers from the current page"

    def _mark(self, rest: str, all_copies: bool) -> str:
        picked = self._pick(rest)
        if isinstance(picked, str):
            return picked
        for entry in picked:
            copies = [e for e in self.entries if e.raw_hash == entry.raw_hash]
            self.marked.update(e.key for e in (copies if all_copies else [entry]))
        return f"{len(self.marked)} cell(s) marked"

    def _unmark(self, rest: str) -> str:
        picked = self._pick(rest)
        if isinstance(picked, str):
            return picked
        self.marked.difference_update(e.key for e in picked)
        return f"{len(self.marked)} cell(s) marked"

    def _marked(self) -> str:
        by_key = {e.key: e for e in self.entries}
        lines = [
            render_entry(by_key[k], n, True)
            if k in by_key
            else f"[x] {n}. {k[0]} (no stored cell {k[1][:8]})"
            for n, k in enumerate(sorted(self.marked), start=1)
        ]
        return "\n".join(lines) or "nothing marked"

    def _done(self) -> str:
        if self.unanswerable and self.marked:
            return "an unanswerable query must have no labels; /unmark them or /abort"
        if self.unanswerable and not self._confirm_empty:
            self._confirm_empty = True
            top = search(self.entries, self.query_text, limit=5)
            shown = "\n".join(render_entry(e, n, False) for n, e in enumerate(top, start=1))
            return f"{shown}\nclosest cells above. /done again to confirm that none is relevant"
        if not self.unanswerable and not self.marked:
            return "mark at least one relevant cell, or /abort"
        self.status = "done"
        return "saved"

    def _abort(self) -> str:
        self.status = "aborted"
        return "discarded"


def _number(token: str, upper: int) -> int | None:
    token = token.strip()
    if token.isdigit() and 1 <= int(token) <= upper:
        return int(token)
    return None
