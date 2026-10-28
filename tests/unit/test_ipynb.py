"""Notebook parsing: cell types, output classification, newline handling, loud failures."""

import json
from pathlib import Path
from typing import Any

import pytest

from make_fixtures import FIXTURE_DIR
from noterecall.ingest.ipynb import IpynbLoader, NotebookParseError
from noterecall.ingest.loader import SourceLoader
from noterecall.models import CellType, OutputKind, RawNotebook

LOADER: SourceLoader = IpynbLoader()  # type-checks that IpynbLoader satisfies the Protocol


def _write_notebook(
    tmp_path: Path, cells: list[dict[str, Any]], minor: int = 0, name: str = "n.ipynb"
) -> Path:
    path = tmp_path / name
    document = {"nbformat": 4, "nbformat_minor": minor, "metadata": {}, "cells": cells}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _code(outputs: list[dict[str, Any]], source: str = "x") -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": outputs,
        "source": source,
    }


def _load(path: Path, root: Path) -> RawNotebook:
    return LOADER.load(path, root, "hash123")


def test_fixture_cells_and_types() -> None:
    nb = LOADER.load(FIXTURE_DIR / "01_no_headings.ipynb", FIXTURE_DIR, "h")
    kinds = [c.cell_type for c in nb.cells]
    md, code = CellType.MARKDOWN, CellType.CODE
    assert kinds == [md, code, md, code, code, md]
    assert [c.index for c in nb.cells] == [0, 1, 2, 3, 4, 5]
    assert (nb.path, nb.file_hash) == ("01_no_headings.ipynb", "h")


def test_stream_output_is_text() -> None:
    nb = LOADER.load(FIXTURE_DIR / "01_no_headings.ipynb", FIXTURE_DIR, "h")
    output = nb.cells[3].outputs[0]
    assert output.kind is OutputKind.TEXT
    assert output.text == "k\n1    3.5\n2    5.0\n"


def test_png_display_data_is_an_image_output() -> None:
    nb = LOADER.load(FIXTURE_DIR / "04_image_output.ipynb", FIXTURE_DIR, "h")
    assert [o.kind for o in nb.cells[1].outputs] == [OutputKind.IMAGE]
    assert nb.cells[1].outputs[0].text == ""  # no text carried for dropped outputs


def test_html_with_plain_text_keeps_text_and_flags_html() -> None:
    nb = LOADER.load(FIXTURE_DIR / "04_image_output.ipynb", FIXTURE_DIR, "h")
    output = nb.cells[2].outputs[0]
    assert (output.kind, output.text, output.has_html) == (OutputKind.TEXT, "   0\n0  1", True)


@pytest.mark.parametrize(
    ("output", "kind"),
    [
        (
            {"output_type": "display_data", "metadata": {}, "data": {"text/html": "<b>x</b>"}},
            OutputKind.HTML,
        ),
        (
            {"output_type": "display_data", "metadata": {}, "data": {"application/json": {"a": 1}}},
            OutputKind.OTHER,
        ),
        (
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/svg+xml": "<svg/>", "text/plain": "p"},
            },
            OutputKind.IMAGE,
        ),
        (
            {"output_type": "error", "ename": "E", "evalue": "v", "traceback": ["tb"]},
            OutputKind.ERROR,
        ),
    ],
)
def test_output_classification(tmp_path: Path, output: dict[str, Any], kind: OutputKind) -> None:
    path = _write_notebook(tmp_path, [_code([output])])
    assert [o.kind for o in _load(path, tmp_path).cells[0].outputs] == [kind]


def test_error_output_does_not_carry_the_traceback(tmp_path: Path) -> None:
    error = {"output_type": "error", "ename": "E", "evalue": "secret-looking", "traceback": ["tb"]}
    path = _write_notebook(tmp_path, [_code([error])])
    assert _load(path, tmp_path).cells[0].outputs[0].text == ""


def test_nbformat_4_0_notebook_without_cell_ids_parses(tmp_path: Path) -> None:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["line one\n", "line two"]},
        _code([], "x = 1"),
    ]
    nb = _load(_write_notebook(tmp_path, cells, minor=0), tmp_path)
    assert nb.cells[0].source == "line one\nline two"
    assert nb.cells[1].source == "x = 1"


def test_windows_newlines_are_normalised(tmp_path: Path) -> None:
    cells = [{"cell_type": "markdown", "metadata": {}, "source": "a\r\nb\rc"}]
    assert _load(_write_notebook(tmp_path, cells), tmp_path).cells[0].source == "a\nb\nc"


def test_path_is_relative_with_posix_separators(tmp_path: Path) -> None:
    path = _write_notebook(tmp_path, [], name="sub/dir/x.ipynb")
    assert _load(path, tmp_path).path == "sub/dir/x.ipynb"


def test_raw_cells_are_rejected_not_skipped(tmp_path: Path) -> None:
    cells = [{"cell_type": "raw", "metadata": {}, "source": "raw text"}]
    with pytest.raises(NotebookParseError, match="unsupported type 'raw'"):
        _load(_write_notebook(tmp_path, cells), tmp_path)


def test_non_json_file_is_a_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.ipynb"
    path.write_text("this is not json", encoding="utf-8")
    with pytest.raises(NotebookParseError, match=r"bad\.ipynb"):
        _load(path, tmp_path)


def test_json_that_is_not_a_notebook_is_a_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.ipynb"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(NotebookParseError, match=r"empty\.ipynb"):
        _load(path, tmp_path)


def test_non_utf8_file_is_a_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "latin.ipynb"
    path.write_bytes(b'{"cells": "\xff\xfe"}')
    with pytest.raises(NotebookParseError, match=r"latin\.ipynb"):
        _load(path, tmp_path)


def test_unknown_output_type_is_rejected_not_skipped(tmp_path: Path) -> None:
    weird = {"output_type": "weird", "text": "hi"}
    with pytest.raises(NotebookParseError, match="unknown output type: 'weird'"):
        _load(_write_notebook(tmp_path, [_code([weird])]), tmp_path)
