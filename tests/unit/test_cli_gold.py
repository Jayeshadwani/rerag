"""`label` and `gold` commands end to end on the synthetic fixtures, in temporary paths."""

import json
from pathlib import Path

from typer.testing import CliRunner

from noterecall import cli
from noterecall.config import load_settings
from noterecall.eval.gold import (
    CellRef,
    GoldQuery,
    GoldSet,
    QueryType,
    freeze_gold,
    load_gold,
    save_gold,
)
from noterecall.index.store import connect, notebook_cells

runner = CliRunner()
SECRET_TEXT = "PRIVATE-QUERY-TEXT"


def _synced(config: Path) -> None:
    assert runner.invoke(cli.app, ["sync", "--config", str(config)]).exit_code == 0


def _gold_path(config: Path) -> Path:
    return load_settings(config).paths.gold_path


def _hash_of(config: Path, notebook: str, index: int) -> str:
    conn = connect(load_settings(config).paths.db_path)
    try:
        for path, cells in notebook_cells(conn):
            if path == notebook:
                return next(c.raw_hash for c in cells if c.index == index)
    finally:
        conn.close()
    raise AssertionError(notebook)


def _label(config: Path, *lines: str, extra: list[str] | None = None):  # type: ignore[no-untyped-def]  # noqa: ANN202
    args = ["label", "--config", str(config), *(extra or [])]
    return runner.invoke(cli.app, args, input="\n".join(lines) + "\n")


def test_label_saves_a_confirmed_query_with_the_owners_marks(config: Path) -> None:
    _synced(config)
    result = _label(config, SECRET_TEXT, "1", "groupby", "/mark 1", "/done", "n")
    assert result.exit_code == 0, result.stdout
    assert "saved q001" in result.stdout
    (query,) = load_gold(_gold_path(config)).queries
    assert (query.id, query.text, query.type) == ("q001", SECRET_TEXT, QueryType.CONCEPT)
    assert len(query.relevant) == 1
    ref = query.relevant[0]
    # "groupby" ranks cell 3 (code) first and cell 5 second; the owner marked number 1
    assert ref.notebook_path == "01_no_headings.ipynb"
    assert ref.raw_hash == _hash_of(config, "01_no_headings.ipynb", 3)
    assert SECRET_TEXT not in result.stderr


def test_label_writes_nothing_before_done_and_abort_discards(config: Path) -> None:
    _synced(config)
    result = _label(config, "some text", "1", "groupby", "/mark 1", "/abort", "n")
    assert result.exit_code == 0
    assert not _gold_path(config).exists()


def test_quitting_mid_query_saves_nothing(config: Path) -> None:
    _synced(config)
    result = _label(config, "some text", "1", "groupby", "/mark 1")  # input ends: aborted
    assert result.exit_code != 0
    assert not _gold_path(config).exists()


def test_label_appends_further_queries_and_never_touches_earlier_ones(config: Path) -> None:
    _synced(config)
    assert _label(config, "first", "1", "groupby", "/mark 1", "/done", "n").exit_code == 0
    before = _gold_path(config).read_text(encoding="utf-8")
    result = _label(config, "second", "5", "/grep zzzzzz", "/done", "/done", "n")
    assert result.exit_code == 0, result.stdout
    gold = load_gold(_gold_path(config))
    assert [q.id for q in gold.queries] == ["q001", "q002"]
    assert gold.queries[1].type is QueryType.UNANSWERABLE
    assert gold.queries[1].relevant == ()
    assert (
        json.loads(before)["queries"][0]
        == json.loads(_gold_path(config).read_text("utf-8"))["queries"][0]
    )


def test_label_asks_for_a_valid_type_and_can_loop_for_more(config: Path) -> None:
    _synced(config)
    result = _label(
        config, "t1", "9", "x", "2", "train_test_split", "/mark 1", "/done", "y",
        "t2", "1", "mean", "/mark 1", "/done", "n",
    )  # fmt: skip
    assert result.exit_code == 0, result.stdout
    assert "enter a number from 1 to 5" in result.stdout
    assert [q.type for q in load_gold(_gold_path(config)).queries] == [
        QueryType.CODE_IDENTIFIER,
        QueryType.CONCEPT,
    ]


def test_label_edit_replaces_labels_but_keeps_text_type_and_split(config: Path) -> None:
    _synced(config)
    nb = "02_deep_headings.ipynb"
    gold = GoldSet(
        queries=(
            GoldQuery(
                id="q001", text="keep me", type=QueryType.CONCEPT,
                relevant=(CellRef(notebook_path="gone.ipynb", raw_hash="dead"),),
            ),
        )
    )  # fmt: skip
    save_gold(gold, _gold_path(config))
    result = _label(
        config, "/marked", "train_test_split", "/mark 1", "/done", extra=["--edit", "q001"]
    )
    assert "gone.ipynb (no stored cell dead)" in result.stdout
    (query,) = load_gold(_gold_path(config)).queries
    assert (query.text, query.type) == ("keep me", QueryType.CONCEPT)
    assert {r.notebook_path for r in query.relevant} == {"gone.ipynb", nb}


def test_label_edit_of_an_unknown_id_fails(config: Path) -> None:
    _synced(config)
    save_gold(GoldSet(), _gold_path(config))
    result = _label(config, extra=["--edit", "q9"])
    assert result.exit_code == 1
    assert "no query with id q9" in result.stderr


def test_label_edit_abort_leaves_the_file_unchanged(config: Path) -> None:
    _synced(config)
    gold = GoldSet(queries=(GoldQuery(id="q001", text="t", type=QueryType.UNANSWERABLE),))
    save_gold(gold, _gold_path(config))
    before = _gold_path(config).read_bytes()
    assert _label(config, "/abort", extra=["--edit", "q001"]).exit_code == 0
    assert _gold_path(config).read_bytes() == before


def test_label_refuses_to_add_to_a_frozen_set_but_allows_edit(config: Path) -> None:
    _synced(config)
    h = _hash_of(config, "01_no_headings.ipynb", 3)
    ref = CellRef(notebook_path="01_no_headings.ipynb", raw_hash=h)
    queries = tuple(
        GoldQuery(
            id=f"q{n:03d}", text=f"t{n}", type=t,
            relevant=() if t is QueryType.UNANSWERABLE else (ref,),
        )
        for n, t in enumerate(list(QueryType) * 2)
    )  # fmt: skip
    save_gold(freeze_gold(GoldSet(queries=queries)), _gold_path(config))
    refused = _label(config)
    assert refused.exit_code == 1
    assert "frozen" in refused.stderr
    edited = _label(config, "/marked", "/abort", extra=["--edit", "q000"])
    assert edited.exit_code == 0


def test_label_and_validate_refuse_an_empty_database(config: Path) -> None:
    label = _label(config, "t", "1", "x", "/abort", "n")
    assert label.exit_code == 1
    assert "the database has no notebooks" in label.stderr
    save_gold(GoldSet(), _gold_path(config))
    validate = runner.invoke(cli.app, ["gold", "validate", "--config", str(config)])
    assert validate.exit_code == 1
    assert "the database has no notebooks" in validate.stderr


def _good_set(config: Path, n_extra: int = 0) -> GoldSet:
    h = _hash_of(config, "01_no_headings.ipynb", 3)
    ref = CellRef(notebook_path="01_no_headings.ipynb", raw_hash=h)
    ref2 = CellRef(
        notebook_path="02_deep_headings.ipynb",
        raw_hash=_hash_of(config, "02_deep_headings.ipynb", 4),
    )
    types = [
        QueryType.CONCEPT,
        QueryType.CODE_IDENTIFIER,
        QueryType.HOW_DID_I,
        QueryType.CROSS_NOTEBOOK,
    ]
    queries = [
        GoldQuery(
            id=f"q{n:03d}",
            text=f"t{n}",
            type=types[n % 4],
            relevant=(ref, ref2) if n % 4 == 3 else (ref,),
        )
        for n in range(54 + n_extra)
    ]
    queries += [GoldQuery(id=f"u{n}", text=f"u{n}", type=QueryType.UNANSWERABLE) for n in range(6)]
    return GoldSet(queries=tuple(queries))


def test_validate_reports_ok_and_prints_no_query_text(config: Path) -> None:
    _synced(config)
    save_gold(_good_set(config), _gold_path(config))
    result = runner.invoke(cli.app, ["gold", "validate", "--config", str(config), "--ready"])
    assert result.exit_code == 0, result.stdout
    assert "0 error(s), 0 warning(s)" in result.stdout
    assert "t1" not in result.stdout.replace("0 error(s)", "")


def test_validate_exits_1_on_orphans_and_names_them_without_query_text(config: Path) -> None:
    _synced(config)
    gold = GoldSet(
        queries=(
            GoldQuery(
                id="q001", text=SECRET_TEXT, type=QueryType.CONCEPT,
                relevant=(CellRef(notebook_path="01_no_headings.ipynb", raw_hash="f" * 64),),
            ),
        )
    )  # fmt: skip
    save_gold(gold, _gold_path(config))
    result = runner.invoke(cli.app, ["gold", "validate", "--config", str(config)])
    assert result.exit_code == 1
    assert "error   [orphan] q001:" in result.stdout
    assert "1 error(s), 0 warning(s)" in result.stdout
    assert SECRET_TEXT not in result.stdout + result.stderr


def test_validate_warnings_do_not_fail_and_ready_enforces_size(config: Path) -> None:
    _synced(config)
    h = _hash_of(config, "01_no_headings.ipynb", 3)
    ref = CellRef(notebook_path="01_no_headings.ipynb", raw_hash=h)
    gold = GoldSet(
        queries=(
            GoldQuery(id="q1", text="t", type=QueryType.CROSS_NOTEBOOK, relevant=(ref,)),
        )
    )  # fmt: skip
    save_gold(gold, _gold_path(config))
    plain = runner.invoke(cli.app, ["gold", "validate", "--config", str(config)])
    assert plain.exit_code == 0
    assert "warning [single_notebook] q1:" in plain.stdout
    ready = runner.invoke(cli.app, ["gold", "validate", "--config", str(config), "--ready"])
    assert ready.exit_code == 1
    assert "[too_few_queries]" in ready.stdout


def test_validate_with_a_missing_gold_file_fails_clearly(config: Path) -> None:
    _synced(config)
    result = runner.invoke(cli.app, ["gold", "validate", "--config", str(config)])
    assert result.exit_code == 1
    assert "cannot read the gold set" in result.stderr


def test_freeze_assigns_the_split_once(config: Path) -> None:
    _synced(config)
    save_gold(_good_set(config), _gold_path(config))
    result = runner.invoke(cli.app, ["gold", "freeze", "--config", str(config)])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip() == "frozen: 42 dev, 18 test"
    assert load_gold(_gold_path(config)).frozen
    again = runner.invoke(cli.app, ["gold", "freeze", "--config", str(config)])
    assert again.exit_code == 1
    assert "already frozen" in again.stderr


def test_freeze_refuses_an_incomplete_set_and_changes_nothing(config: Path) -> None:
    _synced(config)
    save_gold(GoldSet(queries=_good_set(config).queries[:10]), _gold_path(config))
    before = _gold_path(config).read_bytes()
    result = runner.invoke(cli.app, ["gold", "freeze", "--config", str(config)])
    assert result.exit_code == 1
    assert "[too_few_queries]" in result.stdout
    assert "not ready to freeze" in result.stderr
    assert _gold_path(config).read_bytes() == before


def test_stats_prints_counts_only(config: Path) -> None:
    _synced(config)
    save_gold(_good_set(config), _gold_path(config))
    result = runner.invoke(cli.app, ["gold", "stats", "--config", str(config)])
    assert result.exit_code == 0
    assert "queries: 60 (not frozen)" in result.stdout
    assert "concept: 14 unassigned 14" in result.stdout
    assert "unanswerable: 6 unassigned 6" in result.stdout
    assert "labels per answerable query (min/median/max): 1 / 1.0 / 2" in result.stdout
    assert "t1" not in result.stdout.replace("14", "")


def test_gold_path_override_is_used(config: Path, tmp_path: Path) -> None:
    _synced(config)
    other = tmp_path / "other" / "g.json"
    save_gold(GoldSet(queries=(GoldQuery(id="q1", text="t", type=QueryType.UNANSWERABLE),)), other)
    result = runner.invoke(
        cli.app, ["gold", "stats", "--config", str(config), "--gold-path", str(other)]
    )
    assert result.exit_code == 0
    assert "queries: 1 (not frozen)" in result.stdout
    assert not _gold_path(config).exists()


def test_gold_stats_and_freeze_report_a_missing_file(config: Path) -> None:
    _synced(config)
    for command in ("stats", "freeze"):
        result = runner.invoke(cli.app, ["gold", command, "--config", str(config)])
        assert result.exit_code == 1, command
        assert "cannot read the gold set" in result.stderr


def test_commands_report_a_missing_config_file(tmp_path: Path) -> None:
    missing = str(tmp_path / "nope.toml")
    for args in (["label"], ["gold", "validate"], ["gold", "freeze"], ["gold", "stats"]):
        result = runner.invoke(cli.app, [*args, "--config", missing])
        assert result.exit_code == 1, args
        assert "error: config file not found" in result.stderr
