"""Each synthetic fixture notebook has the trait it exists to exercise."""

import hashlib
import re
from pathlib import Path

import nbformat
import pytest
from nbformat import NotebookNode

from make_fixtures import (
    BUILDERS,
    FAKE_ANTHROPIC_KEY,
    FAKE_AWS_ACCESS_KEY,
    FAKE_AWS_SECRET_KEY,
    FIXTURE_DIR,
    SHARED_SPLIT_CELL,
    build_fixtures,
    main,
)


def _load(name: str) -> NotebookNode:
    nb: NotebookNode = nbformat.read(FIXTURE_DIR / name, as_version=4)
    nbformat.validate(nb)
    return nb


def _sources(nb: NotebookNode, cell_type: str | None = None) -> list[str]:
    return [c.source for c in nb.cells if cell_type in (None, c.cell_type)]


def _hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def test_there_are_exactly_the_six_fixtures() -> None:
    on_disk = sorted(p.name for p in FIXTURE_DIR.glob("*.ipynb"))
    assert on_disk == sorted(BUILDERS)
    assert len(on_disk) == 6


def test_committed_files_match_generator(tmp_path: Path) -> None:
    assert main(["--out", str(tmp_path)]) == 0
    for name in BUILDERS:
        assert (tmp_path / name).read_bytes() == (FIXTURE_DIR / name).read_bytes(), name


def test_build_fixtures_returns_valid_notebooks() -> None:
    assert sorted(build_fixtures()) == sorted(BUILDERS)


def test_no_headings_has_no_heading_lines() -> None:
    nb = _load("01_no_headings.ipynb")
    markdown = _sources(nb, "markdown")
    assert len(markdown) >= 2
    assert not any(line.lstrip().startswith("#") for src in markdown for line in src.splitlines())


def test_deep_headings_reaches_h4_and_skips_a_level() -> None:
    nb = _load("02_deep_headings.ipynb")
    levels = {
        len(m.group(1))
        for src in _sources(nb, "markdown")
        for m in re.finditer(r"^(#{1,6}) ", src, flags=re.MULTILINE)
    }
    assert {1, 2, 3, 4} <= levels
    assert 5 in levels
    assert 6 not in levels


def test_huge_output_has_thousands_of_lines() -> None:
    nb = _load("03_huge_output.ipynb")
    lines = [
        line
        for cell in nb.cells
        for output in cell.get("outputs", [])
        for line in output.get("text", "").splitlines()
    ]
    assert len(lines) == 5000


def test_image_output_has_png_and_html_outputs() -> None:
    nb = _load("04_image_output.ipynb")
    mime_types = {
        mime for cell in nb.cells for out in cell.get("outputs", []) for mime in out.get("data", {})
    }
    assert {"image/png", "text/html"} <= mime_types


def test_duplicate_cells_exist_across_and_within_notebooks() -> None:
    deep = {_hash(s) for s in _sources(_load("02_deep_headings.ipynb"), "code")}
    images = [_hash(s) for s in _sources(_load("04_image_output.ipynb"), "code")]
    shared = _hash(SHARED_SPLIT_CELL)
    assert shared in deep  # same cell in two notebooks
    assert images.count(shared) == 2  # and repeated inside one notebook


def test_fake_secret_notebook_contains_every_planted_value() -> None:
    nb = _load("05_fake_secret.ipynb")
    text = "\n".join(
        _sources(nb) + [o.get("text", "") for c in nb.cells for o in c.get("outputs", [])]
    )
    for secret in (FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY, FAKE_AWS_SECRET_KEY):
        assert secret in text


def test_fake_secrets_are_the_documented_synthetic_values() -> None:
    assert FAKE_AWS_ACCESS_KEY.endswith("EXAMPLE")
    assert FAKE_AWS_SECRET_KEY.endswith("EXAMPLEKEY")
    assert FAKE_ANTHROPIC_KEY.startswith("sk-ant-api03-FAKEFAKE")


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_cell_ids_are_unique_within_each_notebook(name: str) -> None:
    ids = [c.id for c in _load(name).cells]
    assert len(ids) == len(set(ids))
