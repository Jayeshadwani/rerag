"""Gold schema, atomic save/load, validator, frozen split and summary (synthetic data only)."""

import json
from pathlib import Path

import pytest

from noterecall.eval.gold import (
    CellRef,
    GoldError,
    GoldQuery,
    GoldSet,
    Issue,
    QueryType,
    Split,
    assign_splits,
    compute_split_hash,
    freeze_gold,
    load_gold,
    next_query_id,
    relevant_keys,
    save_gold,
    summarize,
    validate_gold,
)
from noterecall.eval.gold import (
    held_out_count as split_size,
)

NOTEBOOKS = {
    "a.ipynb": frozenset({"h1", "h2"}),
    "b.ipynb": frozenset({"h2", "h3"}),
}


def query(
    qid: str, qtype: QueryType, *refs: tuple[str, str], split: Split | None = None
) -> GoldQuery:
    relevant = tuple(CellRef(notebook_path=p, raw_hash=h) for p, h in refs)
    return GoldQuery(id=qid, text=f"text {qid}", type=qtype, relevant=relevant, split=split)


def codes(issues: list[Issue]) -> list[tuple[str, str, str | None]]:
    return sorted((i.level, i.code, i.query_id) for i in issues)


def test_cell_ref_and_relevant_keys() -> None:
    q = query("q1", QueryType.CONCEPT, ("a.ipynb", "h1"), ("b.ipynb", "h2"))
    assert relevant_keys(q) == {("a.ipynb", "h1"), ("b.ipynb", "h2")}


def test_save_is_stable_atomic_and_round_trips(tmp_path: Path) -> None:
    gold = GoldSet(queries=(query("q1", QueryType.CONCEPT, ("a.ipynb", "h1")),))
    path = tmp_path / "deep" / "gold.json"
    save_gold(gold, path)
    first = path.read_bytes()
    assert load_gold(path) == gold
    save_gold(gold, path)
    assert path.read_bytes() == first  # same bytes every time
    assert sorted(p.name for p in path.parent.iterdir()) == ["gold.json"]  # no temp files left
    data = json.loads(first)
    assert list(data) == sorted(data)  # keys are sorted for readable diffs


def test_save_keeps_non_ascii_text(tmp_path: Path) -> None:
    gold = GoldSet(queries=(GoldQuery(id="q1", text="é ₹ 日本", type=QueryType.CONCEPT),))
    save_gold(gold, tmp_path / "g.json")
    assert "é ₹ 日本" in (tmp_path / "g.json").read_text(encoding="utf-8")
    assert load_gold(tmp_path / "g.json") == gold


def test_load_errors_are_typed(tmp_path: Path) -> None:
    with pytest.raises(GoldError, match="cannot read the gold set"):
        load_gold(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1, "queries": [{"id": "", "text": "t", "type": "concept"}]}')
    with pytest.raises(GoldError, match="cannot read the gold set"):
        load_gold(bad)
    bad.write_text('{"schema_version": 2, "queries": []}')
    with pytest.raises(GoldError, match="cannot read the gold set"):
        load_gold(bad)


def test_save_to_an_unwritable_place_fails_loudly(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(GoldError, match="cannot write the gold set"):
        save_gold(GoldSet(), blocker / "g.json")


def test_next_query_id_skips_taken_ids() -> None:
    assert next_query_id(GoldSet()) == "q001"
    gold = GoldSet(
        queries=(
            query("q001", QueryType.CONCEPT, ("a.ipynb", "h1")),
            GoldQuery(id="q003", text="t", type=QueryType.UNANSWERABLE),
        )
    )
    assert next_query_id(gold) == "q004"  # 2 queries -> try q003 (taken) -> q004
    assert next_query_id(GoldSet(queries=(gold.queries[1],))) == "q002"


def test_validator_accepts_a_consistent_set() -> None:
    gold = GoldSet(
        queries=(
            query("q1", QueryType.CONCEPT, ("a.ipynb", "h1")),
            query("q2", QueryType.CROSS_NOTEBOOK, ("a.ipynb", "h1"), ("b.ipynb", "h3")),
            query("q3", QueryType.UNANSWERABLE),
        )
    )
    assert validate_gold(gold, NOTEBOOKS) == []


def test_validator_finds_each_kind_of_problem() -> None:
    gold = GoldSet(
        queries=(
            query("dup", QueryType.CONCEPT, ("a.ipynb", "h1")),
            query("dup", QueryType.CONCEPT, ("a.ipynb", "h1")),
            query("twice", QueryType.CONCEPT, ("a.ipynb", "h1"), ("a.ipynb", "h1")),
            query("none", QueryType.HOW_DID_I),
            query("unans", QueryType.UNANSWERABLE, ("a.ipynb", "h1")),
            query("gone", QueryType.CODE_IDENTIFIER, ("a.ipynb", "zzz")),
            query("nobook", QueryType.CODE_IDENTIFIER, ("c.ipynb", "h9")),
            query("deleted", QueryType.CONCEPT, ("c.ipynb", "h1")),
            query("moved", QueryType.CONCEPT, ("a.ipynb", "h3")),
            query("single", QueryType.CROSS_NOTEBOOK, ("a.ipynb", "h1")),
        )
    )
    assert codes(validate_gold(gold, NOTEBOOKS)) == sorted(
        [
            ("error", "duplicate_id", "dup"),
            ("error", "duplicate_label", "twice"),
            ("error", "no_labels", "none"),
            ("error", "labelled_unanswerable", "unans"),
            ("error", "orphan", "gone"),
            ("error", "orphan", "nobook"),  # neither the notebook nor the hash exists anywhere
            ("warning", "moved", "deleted"),  # c.ipynb is gone, but h1 exists in a.ipynb
            ("warning", "moved", "moved"),  # h3 exists only in b.ipynb
            ("warning", "single_notebook", "single"),
        ]
    )


def test_orphan_message_names_the_notebook_and_a_hash_prefix_but_not_the_query_text() -> None:
    gold = GoldSet(queries=(query("q1", QueryType.CONCEPT, ("a.ipynb", "abcdef0123456789")),))
    (issue,) = validate_gold(gold, NOTEBOOKS)
    assert "a.ipynb" in issue.message
    assert "abcdef01" in issue.message
    assert "text q1" not in issue.message


def _full_set(unanswerable: int = 6, total: int = 60) -> GoldSet:
    """A complete synthetic set: types in rotation, the last `unanswerable` unanswerable."""
    answerable = [
        QueryType.CONCEPT,
        QueryType.CODE_IDENTIFIER,
        QueryType.HOW_DID_I,
        QueryType.CROSS_NOTEBOOK,
    ]
    queries = []
    for n in range(total - unanswerable):
        qtype = answerable[n % 4]
        refs = (
            [("a.ipynb", "h1"), ("b.ipynb", "h3")]
            if qtype is QueryType.CROSS_NOTEBOOK
            else [("a.ipynb", "h1")]
        )
        queries.append(query(f"q{n:03d}", qtype, *refs))
    queries += [query(f"u{n:03d}", QueryType.UNANSWERABLE) for n in range(unanswerable)]
    return GoldSet(queries=tuple(queries))


def test_completeness_requirements() -> None:
    assert validate_gold(_full_set(), NOTEBOOKS, require_complete=True) == []
    short = codes(
        validate_gold(_full_set(total=59, unanswerable=6), NOTEBOOKS, require_complete=True)
    )
    assert short == [("error", "too_few_queries", None)]
    few_unanswerable = codes(
        validate_gold(_full_set(unanswerable=5), NOTEBOOKS, require_complete=True)
    )
    assert few_unanswerable == [("error", "too_few_unanswerable", None)]  # 5 of 60 < 10%
    only_concepts = GoldSet(
        queries=tuple(query(f"q{i}", QueryType.CONCEPT, ("a.ipynb", "h1")) for i in range(60))
    )
    got = codes(validate_gold(only_concepts, NOTEBOOKS, require_complete=True))
    assert ("error", "missing_type", None) in got
    assert sum(1 for c in got if c[1] == "missing_type") == 4
    empty = codes(validate_gold(GoldSet(), NOTEBOOKS, require_complete=True))
    assert ("error", "too_few_unanswerable", None) in empty
    # without require_complete, size is not checked
    assert validate_gold(_full_set(total=59), NOTEBOOKS) == []


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (8, 2), (10, 3), (12, 4), (15, 5), (20, 6)],
)
def test_held_out_count_is_30_percent_rounded_half_up(n: int, expected: int) -> None:
    # 0.3 * n: 0.3, 0.6, 0.9, 1.2, 1.5 -> 2, 2.4, 3.0, 3.6, 4.5 -> 5, 6.0
    assert split_size(n) == expected


def test_assign_splits_is_stratified_and_deterministic() -> None:
    gold = (
        _full_set()
    )  # 14 each of 4 answerable types + 6... (54 answerable: 14,14,13,13) + 6 unanswerable
    first = assign_splits(gold.queries)
    by_type: dict[QueryType, list[Split]] = {}
    for q in gold.queries:
        by_type.setdefault(q.type, []).append(first[q.id])
    for qtype, splits in by_type.items():
        assert splits.count(Split.TEST) == split_size(len(splits)), qtype
    assert assign_splits(tuple(reversed(gold.queries))) == first  # file order does not matter
    assert assign_splits(gold.queries, seed=1) != first  # the seed matters


def test_a_stratum_is_unaffected_by_other_strata() -> None:
    gold = _full_set()
    before = assign_splits(gold.queries)
    extra = (*gold.queries, query("zzz", QueryType.CONCEPT, ("a.ipynb", "h1")))
    after = assign_splits(extra)
    other_types = [q.id for q in gold.queries if q.type is not QueryType.CONCEPT]
    assert all(before[i] == after[i] for i in other_types)


def test_freeze_assigns_once_and_the_hash_detects_changes() -> None:
    frozen = freeze_gold(_full_set())
    assert frozen.frozen
    assert all(q.split is not None for q in frozen.queries)
    assert frozen.split_hash == compute_split_hash(frozen.queries)
    assert validate_gold(frozen, NOTEBOOKS) == []
    with pytest.raises(GoldError, match="already frozen"):
        freeze_gold(frozen)
    summary = summarize(frozen)
    # per type (14, 14, 13, 13, 6 queries): test sizes 4, 4, 4, 4, 2 = 18; dev 42
    assert summary.by_split == {"dev": 42, "test": 18}


def test_tampering_after_freeze_is_an_error() -> None:
    frozen = freeze_gold(_full_set())
    moved = tuple(
        q.model_copy(update={"split": Split.DEV if q.split is Split.TEST else Split.TEST})
        if n == 0
        else q
        for n, q in enumerate(frozen.queries)
    )
    tampered = frozen.model_copy(update={"queries": moved})
    assert ("error", "split_tampered", None) in codes(validate_gold(tampered, NOTEBOOKS))
    added = frozen.model_copy(
        update={"queries": (*frozen.queries, query("new", QueryType.CONCEPT, ("a.ipynb", "h1")))}
    )
    assert ("error", "added_after_freeze", "new") in codes(validate_gold(added, NOTEBOOKS))
    retyped = frozen.model_copy(
        update={
            "queries": (
                frozen.queries[0].model_copy(update={"type": QueryType.HOW_DID_I}),
                *frozen.queries[1:],
            )
        }
    )
    assert ("error", "split_tampered", None) in codes(validate_gold(retyped, NOTEBOOKS))
    removed = frozen.model_copy(update={"queries": frozen.queries[1:]})
    assert ("error", "split_tampered", None) in codes(validate_gold(removed, NOTEBOOKS))
    assert ("error", "split_without_freeze", None) in codes(
        validate_gold(
            GoldSet(queries=(query("q", QueryType.CONCEPT, ("a.ipynb", "h1"), split=Split.DEV),)),
            NOTEBOOKS,
        )
    )


def test_fixing_labels_after_freeze_is_allowed() -> None:
    frozen = freeze_gold(_full_set())
    first = frozen.queries[0]
    relabelled = first.model_copy(
        update={"relevant": (CellRef(notebook_path="a.ipynb", raw_hash="h2"),)}
    )
    fixed = frozen.model_copy(update={"queries": (relabelled, *frozen.queries[1:])})
    assert validate_gold(fixed, NOTEBOOKS) == []


def test_summary_counts_only() -> None:
    gold = GoldSet(
        queries=(
            query("a", QueryType.CONCEPT, ("a.ipynb", "h1")),
            query("b", QueryType.CONCEPT, ("a.ipynb", "h1"), ("b.ipynb", "h3"), ("a.ipynb", "h2")),
            query("c", QueryType.CONCEPT, ("a.ipynb", "h1"), ("b.ipynb", "h3")),
            query("d", QueryType.UNANSWERABLE),
        )
    )
    s = summarize(gold)
    assert s.queries == 4
    assert not s.frozen
    assert s.by_type == {"concept": 3, "unanswerable": 1}
    assert s.by_split == {"unassigned": 4}
    assert s.by_type_and_split == {"concept": {"unassigned": 3}, "unanswerable": {"unassigned": 1}}
    assert s.labels_per_answerable_query == (1, 2.0, 3)  # sizes 1, 3, 2 -> sorted 1, 2, 3
    assert "text a" not in s.model_dump_json()
    assert summarize(GoldSet()).labels_per_answerable_query is None
    even = GoldSet(
        queries=(
            query("a", QueryType.CONCEPT, ("a.ipynb", "h1")),
            query("b", QueryType.CONCEPT, ("a.ipynb", "h1"), ("a.ipynb", "h2")),
        )
    )
    assert summarize(even).labels_per_answerable_query == (1, 1.5, 2)
