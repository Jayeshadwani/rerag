"""Gold set schema, loader, validator and frozen dev/test split. Implements ADR-009.

The gold set is written by the owner and lives in the private data repo (`paths.gold_path`);
this module never writes labels on its own. Relevance is labelled at the cell level as
`(notebook_path, raw_hash)` so labels stay valid across chunking ablations. The split is
assigned once by `freeze_gold`: within each query type, queries are ranked by a hash of
`seed:id` and the first `round_half_up(0.3 * n)` become the test split. That makes the split
deterministic and independent of file order. After freezing, adding queries or editing a split
is a validation error; fixing labels is still allowed.
"""

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SPLIT_SEED = 20261029
MIN_QUERIES = 60
MIN_UNANSWERABLE_FRACTION = 0.10

CellKey = tuple[str, str]


class GoldError(Exception):
    """Raised when the gold file cannot be read, written or frozen."""


class QueryType(StrEnum):
    """The five strata of ADR-009."""

    CONCEPT = "concept"
    CODE_IDENTIFIER = "code_identifier"
    HOW_DID_I = "how_did_i"
    CROSS_NOTEBOOK = "cross_notebook"
    UNANSWERABLE = "unanswerable"


class Split(StrEnum):
    """Dev is for tuning; test is run once per milestone."""

    DEV = "dev"
    TEST = "test"


class CellRef(BaseModel):
    """A relevant cell: the notebook it is in and its pre-cleaning content hash."""

    model_config = ConfigDict(frozen=True)

    notebook_path: str = Field(min_length=1)
    raw_hash: str = Field(min_length=1)

    @property
    def key(self) -> CellKey:
        return (self.notebook_path, self.raw_hash)


class GoldQuery(BaseModel):
    """One owner-written query with its relevant cells and, once frozen, its split."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    type: QueryType
    relevant: tuple[CellRef, ...] = ()
    split: Split | None = None


class GoldSet(BaseModel):
    """The whole file. `split_hash` is set by freezing and covers (id, type, split)."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    split_hash: str | None = None
    queries: tuple[GoldQuery, ...] = ()

    @property
    def frozen(self) -> bool:
        return self.split_hash is not None


class Issue(BaseModel):
    """One validation finding. Messages carry ids and counts, never query text."""

    model_config = ConfigDict(frozen=True)

    level: Literal["error", "warning"]
    code: str
    query_id: str | None
    message: str


def relevant_keys(query: GoldQuery) -> frozenset[CellKey]:
    """The query's relevant cells as (notebook_path, raw_hash) pairs."""
    return frozenset(ref.key for ref in query.relevant)


def load_gold(path: Path) -> GoldSet:
    """Read and validate the gold file's structure."""
    try:
        return GoldSet.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise GoldError(f"cannot read the gold set at {path}: {exc}") from exc


def save_gold(gold: GoldSet, path: Path) -> None:
    """Write stably formatted JSON atomically (temp file in the same directory, then rename)."""
    text = json.dumps(gold.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        Path(temp).replace(path)
    except OSError as exc:
        raise GoldError(f"cannot write the gold set at {path}: {exc}") from exc


def held_out_count(n: int) -> int:
    """Test-split size for a stratum of n queries: 0.3 * n rounded half up, in integers."""
    return (3 * n + 5) // 10


def _rank(seed: int, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{query_id}".encode()).hexdigest()


def assign_splits(queries: tuple[GoldQuery, ...], seed: int = SPLIT_SEED) -> dict[str, Split]:
    """Stratified 70/30 split by query type; deterministic and independent of query order."""
    assignment: dict[str, Split] = {}
    for qtype in QueryType:
        ids = sorted((q.id for q in queries if q.type is qtype), key=lambda i: (_rank(seed, i), i))
        cut = held_out_count(len(ids))
        assignment.update({i: Split.TEST if n < cut else Split.DEV for n, i in enumerate(ids)})
    return assignment


def compute_split_hash(queries: tuple[GoldQuery, ...]) -> str:
    """Hash of the sorted (id, type, split) triples: detects edited splits and added queries."""
    lines = sorted(f"{q.id}\0{q.type.value}\0{q.split.value if q.split else ''}" for q in queries)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def freeze_gold(gold: GoldSet, seed: int = SPLIT_SEED) -> GoldSet:
    """Assign the split once. Refuses a gold set that is already frozen."""
    if gold.frozen:
        raise GoldError("the gold set is already frozen; the split is fixed once")
    assignment = assign_splits(gold.queries, seed)
    queries = tuple(q.model_copy(update={"split": assignment[q.id]}) for q in gold.queries)
    return GoldSet(schema_version=1, split_hash=compute_split_hash(queries), queries=queries)


def _issue(level: Literal["error", "warning"], code: str, qid: str | None, msg: str) -> Issue:
    return Issue(level=level, code=code, query_id=qid, message=msg)


def _query_issues(query: GoldQuery) -> list[Issue]:
    issues: list[Issue] = []
    keys = [ref.key for ref in query.relevant]
    if len(keys) != len(set(keys)):
        issues.append(
            _issue("error", "duplicate_label", query.id, "the same cell is labelled twice")
        )
    answerable = query.type is not QueryType.UNANSWERABLE
    if answerable and not keys:
        issues.append(_issue("error", "no_labels", query.id, "an answerable query has no labels"))
    if not answerable and keys:
        issues.append(
            _issue("error", "labelled_unanswerable", query.id, "unanswerable but labelled")
        )
    if query.type is QueryType.CROSS_NOTEBOOK and len({path for path, _ in keys}) < 2:
        issues.append(
            _issue("warning", "single_notebook", query.id, "labels span fewer than 2 notebooks")
        )
    return issues


def _orphan_issues(query: GoldQuery, notebooks: Mapping[str, frozenset[str]]) -> list[Issue]:
    issues: list[Issue] = []
    elsewhere = {h for hs in notebooks.values() for h in hs}
    for ref in query.relevant:
        if ref.raw_hash in notebooks.get(ref.notebook_path, frozenset()):
            continue
        if ref.raw_hash in elsewhere:
            msg = (
                f"label {ref.raw_hash[:8]} is in no cell of {ref.notebook_path} (exists elsewhere)"
            )
            issues.append(_issue("warning", "moved", query.id, msg))
        else:
            msg = f"label {ref.raw_hash[:8]} in {ref.notebook_path} matches no stored cell"
            issues.append(_issue("error", "orphan", query.id, msg))
    return issues


def _split_issues(gold: GoldSet) -> list[Issue]:
    splits = [q.split for q in gold.queries]
    if not gold.frozen:
        if any(s is not None for s in splits):
            return [
                _issue(
                    "error",
                    "split_without_freeze",
                    None,
                    "splits are set but the set is not frozen",
                )
            ]
        return []
    missing = [q.id for q in gold.queries if q.split is None]
    if missing:
        return [
            _issue("error", "added_after_freeze", i, "query has no split: added after freeze")
            for i in missing
        ]
    if gold.split_hash != compute_split_hash(gold.queries):
        return [
            _issue("error", "split_tampered", None, "ids, types or splits changed since freeze")
        ]
    return []


def _completeness_issues(gold: GoldSet) -> list[Issue]:
    counts = Counter(q.type for q in gold.queries)
    total = len(gold.queries)
    issues: list[Issue] = []
    if total < MIN_QUERIES:
        issues.append(
            _issue(
                "error", "too_few_queries", None, f"{total} queries; need at least {MIN_QUERIES}"
            )
        )
    if counts[QueryType.UNANSWERABLE] < MIN_UNANSWERABLE_FRACTION * total or total == 0:
        n = counts[QueryType.UNANSWERABLE]
        issues.append(
            _issue(
                "error",
                "too_few_unanswerable",
                None,
                f"{n} of {total} unanswerable; need at least 10%",
            )
        )
    issues.extend(
        _issue("error", "missing_type", None, f"no queries of type {t.value}")
        for t in QueryType
        if counts[t] == 0
    )
    return issues


def validate_gold(
    gold: GoldSet, notebooks: Mapping[str, frozenset[str]], require_complete: bool = False
) -> list[Issue]:
    """All findings. `notebooks` maps each stored notebook path to its cells' raw hashes.

    `require_complete` adds the ADR-009 size and stratification requirements (needed to freeze).
    """
    issues: list[Issue] = []
    duplicate_ids = [i for i, n in Counter(q.id for q in gold.queries).items() if n > 1]
    issues.extend(
        _issue("error", "duplicate_id", i, "query id appears more than once") for i in duplicate_ids
    )
    for query in gold.queries:
        issues.extend(_query_issues(query))
        issues.extend(_orphan_issues(query, notebooks))
    issues.extend(_split_issues(gold))
    if require_complete:
        issues.extend(_completeness_issues(gold))
    return issues


def next_query_id(gold: GoldSet) -> str:
    """The next free id of the form q001, q002, ..."""
    taken = {q.id for q in gold.queries}
    n = len(taken) + 1
    while f"q{n:03d}" in taken:
        n += 1
    return f"q{n:03d}"


class GoldSummary(BaseModel):
    """Aggregate counts only: no query text, no paths."""

    model_config = ConfigDict(frozen=True)

    queries: int
    frozen: bool
    by_type: dict[str, int]
    by_split: dict[str, int]
    by_type_and_split: dict[str, dict[str, int]]
    labels_per_answerable_query: tuple[int, float, int] | None


def _median(sorted_values: list[int]) -> float:
    if not sorted_values:
        return 0.0
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[middle])
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2


def summarize(gold: GoldSet) -> GoldSummary:
    """Counts by type and split, and (min, median, max) labels per answerable query."""
    by_type = Counter(q.type.value for q in gold.queries)
    by_split = Counter(q.split.value if q.split else "unassigned" for q in gold.queries)
    cross: dict[str, dict[str, int]] = {}
    for q in gold.queries:
        row = cross.setdefault(q.type.value, {})
        key = q.split.value if q.split else "unassigned"
        row[key] = row.get(key, 0) + 1
    sizes = sorted(len(q.relevant) for q in gold.queries if q.type is not QueryType.UNANSWERABLE)
    median = _median(sizes)
    return GoldSummary(
        queries=len(gold.queries),
        frozen=gold.frozen,
        by_type=dict(by_type),
        by_split=dict(by_split),
        by_type_and_split=cross,
        labels_per_answerable_query=(sizes[0], median, sizes[-1]) if sizes else None,
    )
