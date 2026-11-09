"""The README is generated from results/ and must match the checked-in file."""

import re

import render_readme


def test_readme_is_up_to_date() -> None:
    assert render_readme.README.read_text(encoding="utf-8") == render_readme.render()


def test_every_placeholder_resolves() -> None:
    template = render_readme.TEMPLATE.read_text(encoding="utf-8")
    assert render_readme.PLACEHOLDER.search(template)
    assert "{{" not in render_readme.render()


def test_values_come_from_the_results_files() -> None:
    assert render_readme.lookup("onnx_parity_dev.decision") == "pass"
    assert render_readme.fmt(0.123456) == "0.1235"
    assert render_readme.fmt(None) == "n/a"
    assert render_readme.fmt(True) == "yes"


def test_numbers_in_the_new_sections_are_placeholders() -> None:
    template = render_readme.TEMPLATE.read_text(encoding="utf-8")
    body = template.split("## Notes from earlier milestones")[0]
    tables = [
        ln
        for ln in body.splitlines()
        if ln.startswith("|") and re.search(r"\d+\.\d+|\d+/\d+|\b\d{2,}\b", ln)
    ]
    assert all("{{" in ln for ln in tables if not ln.startswith("|---"))
