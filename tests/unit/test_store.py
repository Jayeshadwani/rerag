"""SQLite store: schema, migrations, replace/delete semantics, duplicate marking."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from noterecall.index.store import (
    SCHEMA_VERSION,
    StoreError,
    cell_type_counts,
    connect,
    delete_notebooks,
    duplicate_counts,
    notebook_summaries,
    pipeline_hashes,
    replace_notebook,
    stored_notebooks,
)
from noterecall.models import Cell, CellType, Notebook


def _cell(index: int, clean_hash: str, cell_type: CellType = CellType.CODE) -> Cell:
    return Cell(
        index=index,
        cell_type=cell_type,
        raw_hash=f"raw-{clean_hash}",
        clean_hash=clean_hash,
        source=f"source {index}",
        output_text="",
    )


def _notebook(
    path: str, cells: list[Cell], file_hash: str = "f1", pipeline: str = "p1"
) -> Notebook:
    return Notebook(
        path=path,
        file_hash=file_hash,
        pipeline_hash=pipeline,
        raw_tokens=10,
        clean_tokens=7,
        counts={"cells_raw": len(cells)},
        cells=tuple(cells),
    )


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(tmp_path / "db" / "notes.sqlite")
    yield connection
    connection.close()


def test_connect_creates_schema_and_version(conn: sqlite3.Connection) -> None:
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {"notebooks", "cells", "duplicate_cells"} <= names
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_reopening_an_up_to_date_database_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "notes.sqlite"
    connect(path).close()
    second = connect(path)
    assert second.total_changes == 0
    second.close()


def test_database_from_a_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "newer.sqlite"
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(StoreError, match="newer than supported"):
        connect(path)


def test_foreign_database_is_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "foreign.sqlite"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE other (x)")
    raw.commit()
    raw.close()
    with pytest.raises(StoreError, match="no schema version"):
        connect(path)
    check = sqlite3.connect(path)
    assert {r[0] for r in check.execute("SELECT name FROM sqlite_master")} == {"other"}


def test_replace_inserts_and_returns_hashes(conn: sqlite3.Connection) -> None:
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "h0"), _cell(1, "h1")], "fa", "pa"))
    stored = stored_notebooks(conn)
    assert stored["a.ipynb"].file_hash == "fa"
    assert stored["a.ipynb"].pipeline_hash == "pa"
    assert conn.execute("SELECT count(*) FROM cells").fetchone()[0] == 2


def test_replace_removes_the_old_cells(conn: sqlite3.Connection) -> None:
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "h0"), _cell(1, "h1"), _cell(2, "h2")]))
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "new")], file_hash="f2"))
    rows = conn.execute("SELECT idx, clean_hash FROM cells").fetchall()
    assert rows == [(0, "new")]
    assert stored_notebooks(conn)["a.ipynb"].file_hash == "f2"


def test_delete_cascades_to_cells(conn: sqlite3.Connection) -> None:
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "h0")]))
    replace_notebook(conn, _notebook("b.ipynb", [_cell(0, "h1")]))
    delete_notebooks(conn, ["a.ipynb"])
    assert list(stored_notebooks(conn)) == ["b.ipynb"]
    assert conn.execute("SELECT notebook_path FROM cells").fetchall() == [("b.ipynb",)]


def test_invalid_cell_type_is_rejected_by_the_schema(conn: sqlite3.Connection) -> None:
    replace_notebook(conn, _notebook("a.ipynb", []))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO cells VALUES ('a.ipynb', 0, 'raw', 'r', 'c', 's', '')")


def test_summaries_carry_counts_but_no_path(conn: sqlite3.Connection) -> None:
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "h0"), _cell(1, "h1")]))
    (summary,) = notebook_summaries(conn)
    assert (summary.cells, summary.raw_tokens, summary.clean_tokens) == (2, 10, 7)
    assert summary.counts == {"cells_raw": 2}
    assert "a.ipynb" not in summary.model_dump_json()


def test_cell_type_and_pipeline_hash_queries(conn: sqlite3.Connection) -> None:
    cells = [_cell(0, "h0", CellType.MARKDOWN), _cell(1, "h1"), _cell(2, "h2")]
    replace_notebook(conn, _notebook("a.ipynb", cells, pipeline="p1"))
    replace_notebook(conn, _notebook("b.ipynb", [], pipeline="p2"))
    assert cell_type_counts(conn) == {"markdown": 1, "code": 2}
    assert pipeline_hashes(conn) == {"p1", "p2"}


def _dup_rows(conn: sqlite3.Connection) -> list[tuple[str, int, str, int]]:
    query = (
        "SELECT notebook_path, idx, canonical_notebook, canonical_idx"
        " FROM duplicate_cells ORDER BY notebook_path, idx"
    )
    return conn.execute(query).fetchall()


def test_duplicates_are_marked_within_and_across_notebooks(conn: sqlite3.Connection) -> None:
    # H1 appears at a:0 (canonical, first by path then index), a:1 (same notebook), b:0 (other).
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "H1"), _cell(1, "H1"), _cell(2, "U")]))
    replace_notebook(conn, _notebook("b.ipynb", [_cell(0, "H1"), _cell(1, "V")]))
    assert _dup_rows(conn) == [("a.ipynb", 1, "a.ipynb", 0), ("b.ipynb", 0, "a.ipynb", 0)]
    counts = duplicate_counts(conn)
    assert (counts.within_notebook, counts.across_notebooks) == (1, 1)


def test_canonical_copy_does_not_depend_on_insertion_order(conn: sqlite3.Connection) -> None:
    replace_notebook(conn, _notebook("b.ipynb", [_cell(0, "H1")]))
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "H1")]))
    assert _dup_rows(conn) == [("b.ipynb", 0, "a.ipynb", 0)]


def test_cell_type_is_part_of_duplicate_identity(conn: sqlite3.Connection) -> None:
    # Same clean_hash value but different types never occurs from the cleaner (the hash includes
    # the type), so duplicates are decided by clean_hash alone; distinct hashes are never marked.
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "H1"), _cell(1, "H2")]))
    assert _dup_rows(conn) == []
    counts = duplicate_counts(conn)
    assert (counts.within_notebook, counts.across_notebooks) == (0, 0)


def test_three_copies_give_two_extra_copies(conn: sqlite3.Connection) -> None:
    replace_notebook(conn, _notebook("a.ipynb", [_cell(0, "H"), _cell(1, "H"), _cell(2, "H")]))
    counts = duplicate_counts(conn)
    assert (counts.within_notebook, counts.across_notebooks) == (2, 0)
