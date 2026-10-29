"""config/cleaning_rules.yaml: the instructor-note patterns match every style and no look-alike."""

import re
from pathlib import Path
from typing import Any

import nbformat
import pytest
import yaml

from make_fixtures import FIXTURE_DIR

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "cleaning_rules.yaml"


def _rule() -> dict[str, Any]:
    document = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert [r["id"] for r in document["rules"]] == ["instructor_note", "inline_data_uri"]
    rule: dict[str, Any] = document["rules"][0]
    return rule


def _marker() -> re.Pattern[str]:
    return re.compile(_rule()["marker"])


def _optional() -> re.Pattern[str]:
    condition = _rule()["extent"]["extended"]["when_any"][1]
    return re.compile(condition["rest_of_line_matches"])


def test_rule_has_the_approved_shape() -> None:
    rule = _rule()
    assert rule["cell_type"] == "markdown"
    assert rule["extent"]["default"] == "paragraph"
    assert rule["extent"]["extended"]["when_any"][0] == "bare_label"
    assert rule["list_item_scope"] is True
    assert rule["drop_cell_if_empty"] is True
    assert len(rule["extent"]["extended"]["stop_before"]) == 2


@pytest.mark.parametrize(
    "line",
    [
        '<span style="background-color: red;"> **[Note to Instructor]:** </span> Do x.',
        "<span style=background-color:red>**[Note to Instructor]:**</span> Do x.",
        "**Instructor Note:** Do x.",
        "**Instructor Note : Do x.**",
        "**`Instructor Note`**:",
        "> **INSTRUCTOR NOTE:**",
        ">**Instructor Note:**",
        "> instructor note:",
        "> **Instructor's Notes**",
        "> **Instructor\u2019s Notes**",
        "> <font color='red'>INSTRUCTOR NOTES</font> : **optional**",
        "> **<font color='red'>INSTRUCTOR NOTES</font> optional**",
        "<FONT color=red>Instructor Note:</FONT>",
        "<font color='red'> Instructor Note </font>",
        '    - <span style="background-color: red;"> **[Note to Instructor]:**</span> Do x.',
        "- **Instructor Notes:** Do x.",
        "[Note to the Instructor]: Do x.",
    ],
)
def test_marker_matches_every_known_style(line: str) -> None:
    assert _marker().search(line)


@pytest.mark.parametrize(
    "line",
    [
        "- Checking instructor ratings helps choose a course.",
        "Fine-tune the model with the Transformers Trainer class.",
        "The instructor notes are posted on the course page.",
        "Ask the instructor for notes.",
        'Reminder: <span style="color: red;"> **Note:** </span> lists are mutable.',
        '<span style="background-color: red;"> **[Ask Learners]:** </span> Which one?',
        "Instructor-led training is common.",
        "# Instructors and notes",
        "",
    ],
)
def test_marker_ignores_look_alikes(line: str) -> None:
    assert _marker().search(line) is None


@pytest.mark.parametrize(
    "rest",
    [
        " - Optional</font>",
        "</font> optional**",
        " : **optional, just for refrence**",
        " refrence only**",
        " reference only",
        " just for reference",
    ],
)
def test_optional_condition_matches(rest: str) -> None:
    assert _optional().search(rest)


@pytest.mark.parametrize("rest", ["", ":** Show the demo.", " Explain the options."])
def test_optional_condition_ignores_ordinary_notes(rest: str) -> None:
    assert _optional().search(rest) is None


def test_stop_patterns_match_headings_and_rules_only() -> None:
    stops = [re.compile(p) for p in _rule()["extent"]["extended"]["stop_before"]]

    def stops_at(line: str) -> bool:
        return any(p.match(line) for p in stops)

    for line in ["# Title", "### Sub heading", "   ## indented up to 3", "---", " ----- "]:
        assert stops_at(line), line
    for line in ["plain text", "#hashtag", "    # four-space code", "- list item", "--", ""]:
        assert not stops_at(line), line


def _marker_cells(name: str) -> list[str]:
    nb = nbformat.read(FIXTURE_DIR / name, as_version=4)
    pattern = _marker()
    return [
        cell.id for cell in nb.cells if cell.cell_type == "markdown" and pattern.search(cell.source)
    ]


def test_marker_finds_exactly_the_note_cells_in_the_fixture() -> None:
    found = _marker_cells("06_instructor_notes.ipynb")
    assert found == ["n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]


@pytest.mark.parametrize(
    "name",
    [
        "01_no_headings.ipynb",
        "02_deep_headings.ipynb",
        "03_huge_output.ipynb",
        "04_image_output.ipynb",
        "05_fake_secret.ipynb",
        "07_inline_images.ipynb",
    ],
)
def test_marker_finds_nothing_in_the_other_fixtures(name: str) -> None:
    assert _marker_cells(name) == []


def _uri_pattern() -> re.Pattern[str]:
    document = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rule = next(r for r in document["rules"] if r["id"] == "inline_data_uri")
    return re.compile(rule["pattern"])


@pytest.mark.parametrize(
    "text",
    [
        "![a](data:image/png;base64,QUJD)",
        '<img src="data:image/jpeg;base64,/9j/4AAQ==">',
        "data:image/svg+xml;base64,PHN2Zz4=",
        "data:application/pdf;base64,JVBERi0=",
        "x data:image/x-icon;base64,AAAB y",
    ],
)
def test_uri_pattern_matches_data_uris(text: str) -> None:
    assert _uri_pattern().search(text)


@pytest.mark.parametrize(
    "text",
    [
        "data:image/png;base64, nothing follows the comma",
        "data:image/png,rawtext",
        "the data:image/png;base64 syntax",
        "metadata:image/png;base64,QUJD",
        "https://example.com/a.png",
        "database64,QUJD",
    ],
)
def test_uri_pattern_ignores_look_alikes(text: str) -> None:
    assert _uri_pattern().search(text) is None
