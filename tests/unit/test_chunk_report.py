"""Chunk statistics: size summaries and a hand-computed two-cell notebook."""

import sqlite3
from pathlib import Path

import pytest

from noterecall.chunking.report import ChunkStats, compute_chunk_stats, output_verdicts, sizes
from noterecall.chunking.tokens import Tokenizer
from noterecall.config import Settings, load_settings
from noterecall.index.store import connect, notebook_cells, replace_notebook
from noterecall.ingest.stats import GitState, Provenance, StatsError
from noterecall.models import Cell, CellType, Notebook, OutputMode

PROVENANCE = Provenance(
    pipeline_hash="p",
    rules_sha256="r",
    tokenizer_name="tiny-bytes",
    git=GitState(sha="a" * 40, dirty=False),
)


def test_sizes_use_the_nearest_rank_percentile() -> None:
    ten = sizes([10, 1, 9, 2, 8, 3, 7, 4, 6, 5])
    # sorted 1..10: median (5+6)/2 = 5.5; p90 rank = ceil(0.9 * 10) = 9 -> the 9th value
    assert (ten.min, ten.median, ten.p90, ten.max) == (1, 5.5, 9, 10)
    three = sizes([3, 1, 2])
    assert (three.min, three.median, three.p90, three.max) == (1, 2.0, 3, 3)  # rank ceil(2.7) = 3
    one = sizes([5])
    assert (one.min, one.median, one.p90, one.max) == (5, 5.0, 5, 5)


def _cell(index: int, cell_type: CellType, source: str, output: str = "") -> Cell:
    return Cell(
        index=index,
        cell_type=cell_type,
        raw_hash=f"raw{index}",
        clean_hash=f"clean{index}",
        source=source,
        output_text=output,
    )


def _settings(tmp_path: Path, caps: str) -> Settings:
    base = Path(__file__).resolve().parents[2] / "config" / "default.toml"
    text = base.read_text(encoding="utf-8").replace(
        "candidate_caps = [256, 512, 1024]", f"candidate_caps = {caps}"
    )
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_settings(path)


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = connect(tmp_path / "db.sqlite")
    cells = (
        _cell(0, CellType.MARKDOWN, "# T\nintro"),  # 9 bytes
        _cell(1, CellType.CODE, "x = 1", output="42"),  # 5 bytes, output 2 bytes
    )
    notebook = Notebook(
        path="nb.ipynb",
        file_hash="f",
        pipeline_hash="p",
        raw_tokens=0,
        clean_tokens=0,
        counts={},
        cells=cells,
    )
    replace_notebook(connection, notebook)
    return connection


def test_two_cell_notebook_at_cap_12(
    conn: sqlite3.Connection, tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    stats = compute_chunk_stats(conn, _settings(tmp_path, "[12]"), PROVENANCE, tiny_tokenizer)
    rows = {(r.chunker, r.outputs): r for r in stats.rows}
    assert len(stats.rows) == 4  # structural x 3 output modes + fixed

    # outputs excluded: md (9) and code (5) together would be 9 + 2 + 5 = 16 > 12 -> two chunks
    none = rows[("structural", OutputMode.NONE)]
    assert none.chunks == 2
    assert (none.tokens.min, none.tokens.median, none.tokens.p90, none.tokens.max) == (5, 7.0, 9, 9)
    assert none.under_quarter_cap == 0  # cap / 4 = 3
    assert none.cells_split == 0
    assert all(v == 0 for v in none.flagged_by_reason.values())

    # outputs included: md alone (9); code + output = "x = 1\n\n42" = 9 -> still two chunks of 9
    for mode in (OutputMode.ALL, OutputMode.EVIDENCE):  # "42" is evidence
        row = rows[("structural", mode)]
        assert row.chunks == 2
        assert (row.tokens.min, row.tokens.max) == (9, 9)

    # fixed baseline: stream "# T\nintro\n\nx = 1" is 16 bytes -> windows of 12 and 4
    fixed = rows[("fixed", OutputMode.NONE)]
    assert fixed.chunks == 2
    assert (fixed.tokens.min, fixed.tokens.median, fixed.tokens.p90, fixed.tokens.max) == (
        4,
        8.0,
        12,
        12,
    )
    assert (
        fixed.cells_split == 1
    )  # cell 1... the second window shares "intro\n\nx" boundary? see below


def test_report_metadata_and_verdict_counts(
    conn: sqlite3.Connection, tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    stats = compute_chunk_stats(conn, _settings(tmp_path, "[12]"), PROVENANCE, tiny_tokenizer)
    assert (stats.notebooks, stats.cells) == (1, 2)
    assert stats.outputs_by_verdict["evidence"] == 1
    assert sum(stats.outputs_by_verdict.values()) == 1
    assert stats.tokenizer == "tiny-bytes"
    assert len(stats.evidence_rule_fingerprint) == 64
    assert ChunkStats.model_validate_json(stats.model_dump_json()) == stats


def test_verdicts_cover_every_kind(tiny_tokenizer: Tokenizer) -> None:
    cells = [
        _cell(0, CellType.CODE, "a", output="42"),
        _cell(1, CellType.CODE, "b", output="UserWarning: x"),
        _cell(2, CellType.CODE, "c", output=""),
        _cell(3, CellType.MARKDOWN, "d"),
    ]
    counts = output_verdicts([("nb", cells)], tiny_tokenizer)
    assert counts["evidence"] == 1
    assert counts["warning"] == 1
    assert sum(counts.values()) == 2  # empty outputs and markdown cells are not counted


def test_refuses_an_empty_or_stale_database(
    conn: sqlite3.Connection, tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    settings = _settings(tmp_path, "[12]")
    stale = PROVENANCE.model_copy(update={"pipeline_hash": "other"})
    with pytest.raises(StatsError, match="different pipeline"):
        compute_chunk_stats(conn, settings, stale, tiny_tokenizer)
    empty = connect(tmp_path / "empty.sqlite")
    with pytest.raises(StatsError, match="no notebooks"):
        compute_chunk_stats(empty, settings, PROVENANCE, tiny_tokenizer)


def test_notebook_cells_come_back_in_order(conn: sqlite3.Connection) -> None:
    ((path, cells),) = list(notebook_cells(conn))
    assert path == "nb.ipynb"
    assert [c.index for c in cells] == [0, 1]
    assert [c.cell_type for c in cells] == [CellType.MARKDOWN, CellType.CODE]


def test_unparsed_code_cells_are_counted_and_flagged(
    tmp_path: Path, tiny_tokenizer: Tokenizer
) -> None:
    connection = connect(tmp_path / "unparsed.sqlite")
    broken = _cell(0, CellType.CODE, "def f(:\n  aaaaaaaaaaaa")  # does not parse; 22 bytes
    notebook = Notebook(
        path="nb.ipynb",
        file_hash="f",
        pipeline_hash="p",
        raw_tokens=0,
        clean_tokens=0,
        counts={},
        cells=(broken,),
    )
    replace_notebook(connection, notebook)
    stats = compute_chunk_stats(connection, _settings(tmp_path, "[12]"), PROVENANCE, tiny_tokenizer)
    structural = next(
        r for r in stats.rows if r.chunker == "structural" and r.outputs is OutputMode.NONE
    )
    assert structural.unparsed_code_cells == 1
    assert structural.flagged_by_reason["unparsed_code"] == 1
    assert structural.chunks == 1  # never cut
