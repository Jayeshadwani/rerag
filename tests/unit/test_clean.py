"""Cleaning rules: rules-file loading, instructor notes, outputs, cells, whole notebooks.

Expected values for instructor-note cells were derived by hand from the rule in
config/cleaning_rules.yaml (each case has the arithmetic in a comment).
"""

import hashlib
from collections import Counter
from pathlib import Path

import nbformat
import pytest

from make_fixtures import (
    FAKE_ANTHROPIC_KEY,
    FAKE_AWS_ACCESS_KEY,
    FAKE_AWS_SECRET_KEY,
    FIXTURE_DIR,
)
from noterecall.chunking.tokens import Tokenizer
from noterecall.ingest.clean import (
    CleaningRules,
    InstructorNoteRule,
    RulesError,
    clean_notebook,
    clean_outputs,
    load_rules,
    remove_inline_data_uris,
    remove_instructor_notes,
)
from noterecall.ingest.ipynb import IpynbLoader
from noterecall.models import (
    CellType,
    Notebook,
    OutputKind,
    RawCell,
    RawNotebook,
    RawOutput,
)

RULES_PATH = Path(__file__).resolve().parents[2] / "config" / "cleaning_rules.yaml"


@pytest.fixture(scope="module")
def rules() -> CleaningRules:
    return load_rules(RULES_PATH)


@pytest.fixture(scope="module")
def rule(rules: CleaningRules) -> InstructorNoteRule:
    return rules.instructor_note


def _fixture_cells(name: str) -> dict[str, str]:
    nb = nbformat.read(FIXTURE_DIR / name, as_version=4)
    return {cell.id: cell.source for cell in nb.cells}


def _clean_fixture(
    name: str, rules: CleaningRules, tokenizer: Tokenizer, lines: int = 20
) -> Notebook:
    raw = IpynbLoader().load(FIXTURE_DIR / name, FIXTURE_DIR, "filehash")
    return clean_notebook(raw, rules, lines, tokenizer, "pipelinehash")


# ---------------------------------------------------------------- rules file


def test_real_rules_file_loads(rule: InstructorNoteRule) -> None:
    assert rule.extend_bare_label is True
    assert rule.optional_rest is not None
    assert len(rule.stop_before) == 2
    assert rule.list_item_scope is True
    assert rule.drop_cell_if_empty is True


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _variant(tmp_path: Path, old: str, new: str) -> Path:
    return _write(tmp_path, RULES_PATH.read_text(encoding="utf-8").replace(old, new))


def test_missing_rules_file_is_a_rules_error(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="not found"):
        load_rules(tmp_path / "nope.yaml")


def test_malformed_yaml_is_a_rules_error(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="not valid YAML"):
        load_rules(_write(tmp_path, "rules: [unclosed"))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("version: 1", "version: 2", "invalid"),
        ("list_item_scope: true", "list_item_scope: true\n    surprise: 1", "invalid"),
        ("- id: instructor_note", "- id: other_rule", "invalid"),
        ("          - bare_label", "          - mystery_condition", "unknown extended condition"),
    ],
)
def test_invalid_rules_are_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    with pytest.raises(RulesError, match=message):
        load_rules(_variant(tmp_path, old, new))


def test_invalid_regex_is_a_rules_error(tmp_path: Path) -> None:
    text = RULES_PATH.read_text(encoding="utf-8").replace(
        "instructors?|instructor", "instructors?(|instructor"
    )
    with pytest.raises(RulesError, match="invalid regex for marker"):
        load_rules(_write(tmp_path, text))


# ------------------------------------------------------ instructor-note engine

NOTE_CASES = [
    # id, expected text, notes, lines removed
    # n2: one-line note, whole cell -> "" (1 line)
    ("n2", "", 1, 1),
    # n3: marker is its own 1-line paragraph; blank lines around it merge -> 1 line removed
    (
        "n3",
        "Sorting arranges items in order.\n\nStable means equal items keep their original order.",
        1,
        1,
    ),
    # n4: label + 2 bullets are one paragraph (lines 0-2) -> 3 lines removed, tail kept
    ("n4", "Now compare the two approaches.", 1, 3),
    # n5: bare label at line 2 -> extended to end of 7-line cell: lines 2..6 = 5 removed
    ("n5", "Next, complexity.", 1, 5),
    # n6: bare label at line 2, stop at '---' on line 6 -> lines 2..5 = 4 removed
    ("n6", "### Space usage\n\n---\n\nIn-place sorts reuse the input list.", 1, 4),
    # n7: 'Optional' note at line 0, stop at heading on line 6 -> lines 0..5 = 6 removed
    ("n7", "## Quick recap\n\nSorting is everywhere.", 1, 6),
    # n8: 'refrence only' (typo) note at line 0, no stop -> whole 3-line cell
    ("n8", "", 1, 3),
    # n9: bullet note at line 2 (indent 4); sibling '- Partition' (indent 0) ends it -> 1 line
    ("n9", "Steps:\n- Pick a pivot\n- Partition the rest\n- Recurse on both halves", 1, 1),
]


@pytest.mark.parametrize(("cell_id", "expected", "notes", "removed"), NOTE_CASES)
def test_instructor_note_fixture_cells(
    rule: InstructorNoteRule, cell_id: str, expected: str, notes: int, removed: int
) -> None:
    result = remove_instructor_notes(_fixture_cells("06_instructor_notes.ipynb")[cell_id], rule)
    assert result.text == expected
    assert (result.notes, result.lines_removed) == (notes, removed)


@pytest.mark.parametrize("cell_id", ["k1", "k2", "k3", "k4", "k5", "k6"])
def test_look_alike_cells_are_untouched(rule: InstructorNoteRule, cell_id: str) -> None:
    text = _fixture_cells("06_instructor_notes.ipynb")[cell_id]
    result = remove_instructor_notes(text, rule)
    assert (result.text, result.notes, result.lines_removed) == (text, 0, 0)


def test_two_notes_in_separate_paragraphs_are_both_removed(rule: InstructorNoteRule) -> None:
    text = "a\n\n**Instructor Note:** x\n\nb\n\n**Instructor Note:** y\n\nc"
    result = remove_instructor_notes(text, rule)
    assert (result.text, result.notes, result.lines_removed) == ("a\n\nb\n\nc", 2, 2)


def test_second_marker_inside_a_removed_paragraph_is_not_double_counted(
    rule: InstructorNoteRule,
) -> None:
    result = remove_instructor_notes("a\n**Instructor Note:** x\n**Instructor Note:** y\n\nb", rule)
    # paragraph is lines 0-2; the note starts at line 1 and takes lines 1-2
    assert (result.text, result.notes, result.lines_removed) == ("a\n\nb", 1, 2)


def test_text_above_the_marker_in_the_same_paragraph_is_kept(rule: InstructorNoteRule) -> None:
    result = remove_instructor_notes("keep\n**Instructor Note:** drop\nalso drop\n\nkeep2", rule)
    assert result.text == "keep\n\nkeep2"


def test_bare_label_as_the_only_line_removes_the_cell(rule: InstructorNoteRule) -> None:
    assert remove_instructor_notes("**Instructor Note:**", rule).text == ""


def test_indent_of_bullets_deeper_than_the_marker_stays_inside_the_note(
    rule: InstructorNoteRule,
) -> None:
    text = "- a\n- **Instructor Note:** x\n    - detail of the note\n- b"
    # marker indent 0; the deeper '- detail' is part of the note; '- b' (indent 0) ends it
    result = remove_instructor_notes(text, rule)
    assert (result.text, result.lines_removed) == ("- a\n- b", 2)


def test_bulleted_note_with_no_later_sibling_runs_to_its_paragraph_end(
    rule: InstructorNoteRule,
) -> None:
    result = remove_instructor_notes("- a\n- **Instructor Note:** drop\n    - its detail", rule)
    assert (result.text, result.lines_removed) == ("- a", 2)


def test_switching_off_list_scope_takes_the_siblings_too(rule: InstructorNoteRule) -> None:
    text = _fixture_cells("06_instructor_notes.ipynb")["n9"]
    result = remove_instructor_notes(text, rule.model_copy(update={"list_item_scope": False}))
    assert result.text == "Steps:\n- Pick a pivot"  # paragraph rule: lines 2-4 removed
    assert result.lines_removed == 3


def test_switching_off_bare_label_extension_uses_the_paragraph(rule: InstructorNoteRule) -> None:
    text = _fixture_cells("06_instructor_notes.ipynb")["n5"]
    result = remove_instructor_notes(text, rule.model_copy(update={"extend_bare_label": False}))
    assert result.text == (
        "Next, complexity.\n\nDo not derive the bound in class.\n\n"
        "Point to the reading list instead."
    )
    assert result.lines_removed == 1


def test_stop_line_directly_after_the_label_ends_the_note(rule: InstructorNoteRule) -> None:
    # '<font>Instructor Note</font>' is alone on its line only if a blank follows; here a heading
    # is the very next line, so the paragraph rule applies (3 contiguous lines).
    result = remove_instructor_notes("<font color='red'>Instructor Note</font>\n# H\ntext", rule)
    assert result.text == ""


# ------------------------------------------------------------------- outputs


def _text(text: str, has_html: bool = False) -> RawOutput:
    return RawOutput(kind=OutputKind.TEXT, text=text, has_html=has_html)


def test_text_output_is_kept_and_trailing_newlines_trimmed() -> None:
    counts: Counter[str] = Counter()
    assert clean_outputs((_text("a\nb\n"),), 5, counts) == "a\nb"
    assert counts == {}


def test_output_at_the_limit_is_not_truncated() -> None:
    counts: Counter[str] = Counter()
    assert clean_outputs((_text("1\n2\n3"),), 3, counts) == "1\n2\n3"
    assert counts == {}


def test_long_output_is_truncated_with_a_marker() -> None:
    counts: Counter[str] = Counter()
    result = clean_outputs((_text("l1\nl2\nl3\nl4\nl5\n"),), 3, counts)
    # 5 lines, limit 3 -> 2 hidden
    assert result == "l1\nl2\nl3\n[... 2 more lines truncated]"
    assert counts == {"outputs_truncated": 1, "output_lines_truncated": 2}


@pytest.mark.parametrize(
    ("kind", "key"),
    [
        (OutputKind.IMAGE, "output_image_dropped"),
        (OutputKind.HTML, "output_html_dropped"),
        (OutputKind.OTHER, "output_other_dropped"),
        (OutputKind.ERROR, "output_error_dropped"),
    ],
)
def test_non_text_outputs_are_dropped_and_counted(kind: OutputKind, key: str) -> None:
    counts: Counter[str] = Counter()
    assert clean_outputs((RawOutput(kind=kind),), 5, counts) == ""
    assert counts == {key: 1}


def test_html_representation_is_dropped_but_plain_text_kept() -> None:
    counts: Counter[str] = Counter()
    assert clean_outputs((_text("   0\n0  1", has_html=True),), 5, counts) == "   0\n0  1"
    assert counts == {"output_html_representation_dropped": 1}


def test_outputs_are_joined_and_blank_ones_skipped() -> None:
    counts: Counter[str] = Counter()
    assert clean_outputs((_text("one\n"), _text("  \n"), _text("two")), 5, counts) == "one\ntwo"


def test_secret_in_output_is_scrubbed_before_truncation() -> None:
    key = "AKIA" + "ABCDEFGHIJKLMNOP"
    counts: Counter[str] = Counter()
    result = clean_outputs((_text("1\n2\n3\n" + key),), 3, counts)
    assert key not in result
    assert counts["secrets_aws_access_key_id"] == 1  # counted even though line 4 is truncated


# ------------------------------------------------------------- whole notebooks


def test_notebook_without_headings_drops_the_empty_cell(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    nb = _clean_fixture("01_no_headings.ipynb", rules, tiny_tokenizer)
    assert nb.counts == {"cells_raw": 6, "cells_dropped_empty": 1}
    assert [c.index for c in nb.cells] == [0, 1, 2, 3, 5]  # original positions; a5 (index 4) gone
    assert nb.cells[3].output_text == "k\n1    3.5\n2    5.0"


def test_huge_output_is_truncated_to_the_limit(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    nb = _clean_fixture("03_huge_output.ipynb", rules, tiny_tokenizer, lines=20)
    assert nb.counts["outputs_truncated"] == 1
    assert nb.counts["output_lines_truncated"] == 4980  # 5000 - 20
    output = nb.cells[1].output_text.split("\n")
    assert len(output) == 21
    assert output[-1] == "[... 4980 more lines truncated]"
    assert nb.clean_tokens < nb.raw_tokens


def test_image_and_html_outputs_are_removed(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    nb = _clean_fixture("04_image_output.ipynb", rules, tiny_tokenizer)
    assert nb.counts["output_image_dropped"] == 1
    assert nb.counts["output_html_representation_dropped"] == 1
    assert [c.output_text for c in nb.cells] == ["", "", "   0\n0  1", "", "", ""]
    assert len(nb.cells) == 6


def test_duplicate_cells_share_a_clean_hash(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    nb = _clean_fixture("04_image_output.ipynb", rules, tiny_tokenizer)
    by_index = {c.index: c for c in nb.cells}
    assert by_index[3].clean_hash == by_index[5].clean_hash  # same cell twice in one notebook
    assert by_index[3].clean_hash != by_index[2].clean_hash
    deep = _clean_fixture("02_deep_headings.ipynb", rules, tiny_tokenizer)
    assert by_index[3].clean_hash in {c.clean_hash for c in deep.cells}  # and across notebooks


def test_secrets_never_reach_the_cleaned_notebook(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    nb = _clean_fixture("05_fake_secret.ipynb", rules, tiny_tokenizer)
    stored = nb.model_dump_json()
    for secret in (FAKE_ANTHROPIC_KEY, FAKE_AWS_ACCESS_KEY, FAKE_AWS_SECRET_KEY):
        assert secret not in stored
    assert nb.counts["secrets_anthropic_api_key"] == 2  # code cell + markdown cell
    assert nb.counts["secrets_aws_access_key_id"] == 2  # code cell + its output
    assert nb.counts["secrets_aws_secret_access_key"] == 1


def test_instructor_notes_fixture_counts(rules: CleaningRules, tiny_tokenizer: Tokenizer) -> None:
    nb = _clean_fixture("06_instructor_notes.ipynb", rules, tiny_tokenizer)
    assert nb.counts["cells_raw"] == 15
    assert nb.counts["instructor_notes"] == 8
    assert nb.counts["instructor_note_cells_dropped"] == 2  # n2 and n8 become empty
    assert nb.counts["instructor_note_lines_removed"] == 24  # 1+1+3+5+4+6+3+1
    assert len(nb.cells) == 13
    code_cell = nb.cells[-1]
    assert code_cell.cell_type is CellType.CODE
    assert code_cell.source == "# Instructor note: this comment is inside a code cell"


def test_kept_empty_note_cell_when_rule_says_not_to_drop(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    keep = rules.model_copy(
        update={
            "instructor_note": rules.instructor_note.model_copy(
                update={"drop_cell_if_empty": False}
            )
        }
    )
    nb = _clean_fixture("06_instructor_notes.ipynb", keep, tiny_tokenizer)
    assert len(nb.cells) == 15
    assert nb.counts.get("instructor_note_cells_dropped", 0) == 0


def test_raw_hash_is_independent_of_cleaning(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    raw_secret_source = _fixture_cells("05_fake_secret.ipynb")["e2"]
    expected = hashlib.sha256(raw_secret_source.encode()).hexdigest()
    first = _clean_fixture("05_fake_secret.ipynb", rules, tiny_tokenizer, lines=5)
    second = _clean_fixture("05_fake_secret.ipynb", rules, tiny_tokenizer, lines=50)
    assert first.cells[1].raw_hash == second.cells[1].raw_hash == expected
    assert first.cells[1].clean_hash != expected  # the cleaned text differs from the raw text


def test_token_counts_before_and_after(rules: CleaningRules, tiny_tokenizer: Tokenizer) -> None:
    raw = RawNotebook(
        path="x.ipynb",
        file_hash="f",
        cells=(
            RawCell(index=0, cell_type=CellType.MARKDOWN, source="keep me"),  # 7 bytes
            RawCell(
                index=1, cell_type=CellType.MARKDOWN, source="**Instructor Note:** drop this"
            ),  # 30
            RawCell(
                index=2, cell_type=CellType.CODE, source="x=1", outputs=(_text("1\n"),)
            ),  # 3 + 2
        ),
    )
    nb = clean_notebook(raw, rules, 20, tiny_tokenizer, "p")
    # raw: 7 + 30 + 3 + 2 = 42. clean: 7 + (note cell dropped) + 3 + 1 ("1" after trimming) = 11.
    assert (nb.raw_tokens, nb.clean_tokens) == (42, 11)
    assert nb.counts == {
        "cells_raw": 3,
        "instructor_note_cells_dropped": 1,
        "instructor_note_lines_removed": 1,
        "instructor_notes": 1,
    }


def test_marker_shaped_lines_in_code_cells_are_left_alone(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    # A multi-line string in a code cell can contain a line shaped exactly like a note marker.
    source = 'text = """\n> **Instructor Note:** this is data, not a note\n"""'
    raw = RawNotebook(
        path="x.ipynb",
        file_hash="f",
        cells=(RawCell(index=0, cell_type=CellType.CODE, source=source),),
    )
    nb = clean_notebook(raw, rules, 20, tiny_tokenizer, "p")
    assert nb.cells[0].source == source
    assert "instructor_notes" not in nb.counts


# ------------------------------------------------- inline data URIs (proposal 001)

URI_CASES = [
    # id, expected text, URIs removed, payload characters removed
    ("i2", "![flow](data:image/png;base64,[removed])", 1, 400),
    ("i3", 'Before <img src="data:image/jpeg;base64,[removed]" width="40"> after', 1, 200),
    (
        "i4",
        "![a](data:image/gif;base64,[removed])\n\n![b](data:image/png;base64,[removed])",
        2,
        16,
    ),
    ("i8", "![x](data:image/png;base64,[removed])", 1, 12),
]


@pytest.mark.parametrize(("cell_id", "expected", "removed", "chars"), URI_CASES)
def test_data_uri_fixture_cells(
    rules: CleaningRules, cell_id: str, expected: str, removed: int, chars: int
) -> None:
    text = _fixture_cells("07_inline_images.ipynb")[cell_id]
    result = remove_inline_data_uris(text, rules.inline_data_uri)
    assert (result.text, result.removed, result.chars) == (expected, removed, chars)


@pytest.mark.parametrize("cell_id", ["i1", "i5", "i6"])
def test_cells_without_a_payload_are_untouched(rules: CleaningRules, cell_id: str) -> None:
    text = _fixture_cells("07_inline_images.ipynb")[cell_id]
    result = remove_inline_data_uris(text, rules.inline_data_uri)
    assert (result.text, result.removed, result.chars) == (text, 0, 0)


@pytest.mark.parametrize("mime", ["image/svg+xml", "application/pdf", "image/x-icon", "text/plain"])
def test_any_mime_type_is_handled(rules: CleaningRules, mime: str) -> None:
    result = remove_inline_data_uris(f"[f](data:{mime};base64,QUJDRA==)", rules.inline_data_uri)
    assert result.text == f"[f](data:{mime};base64,[removed])"
    assert (result.removed, result.chars) == (1, 8)  # "QUJDRA==" is 8 characters


def test_a_word_ending_in_data_is_not_a_data_uri(rules: CleaningRules) -> None:
    text = "metadata:image/png;base64,QUJDRA== is not a URI"
    assert remove_inline_data_uris(text, rules.inline_data_uri).removed == 0


def test_payload_stops_at_the_first_character_outside_the_base64_alphabet(
    rules: CleaningRules,
) -> None:
    result = remove_inline_data_uris("(data:image/png;base64,QUJD) and more", rules.inline_data_uri)
    assert result.text == "(data:image/png;base64,[removed]) and more"
    assert result.chars == 4


def test_removal_is_idempotent(rules: CleaningRules) -> None:
    once = remove_inline_data_uris("![x](data:image/png;base64,QUJDRA==)", rules.inline_data_uri)
    twice = remove_inline_data_uris(once.text, rules.inline_data_uri)
    assert (twice.text, twice.removed, twice.chars) == (once.text, 0, 0)


def test_very_large_payload_is_handled_in_linear_time(rules: CleaningRules) -> None:
    payload = "QUJD" * 750_000  # 3,000,000 characters
    result = remove_inline_data_uris(
        f"![x](data:image/png;base64,{payload})", rules.inline_data_uri
    )
    assert result.text == "![x](data:image/png;base64,[removed])"
    assert result.chars == 3_000_000


def test_inline_images_fixture_notebook(rules: CleaningRules, tiny_tokenizer: Tokenizer) -> None:
    nb = _clean_fixture("07_inline_images.ipynb", rules, tiny_tokenizer)
    assert nb.counts == {
        "cells_raw": 8,
        "inline_data_uri_removed": 5,
        "inline_data_uri_chars_removed": 628,
    }
    assert len(nb.cells) == 8  # nothing is dropped: every image cell keeps its markdown
    by_id = dict(zip(_fixture_cells("07_inline_images.ipynb"), nb.cells, strict=True))
    assert by_id["i2"].source == "![flow](data:image/png;base64,[removed])"
    assert by_id["i7"].cell_type is CellType.CODE
    assert by_id["i7"].source == _fixture_cells("07_inline_images.ipynb")["i7"]  # code untouched
    # 5 payloads (628 bytes) became "[removed]" (9 bytes each): 628 - 5 * 9 = 583 fewer tokens
    assert nb.raw_tokens - nb.clean_tokens == 583


def test_raw_hash_still_covers_the_payload(rules: CleaningRules, tiny_tokenizer: Tokenizer) -> None:
    nb = _clean_fixture("07_inline_images.ipynb", rules, tiny_tokenizer)
    raw_source = _fixture_cells("07_inline_images.ipynb")["i2"]
    cell = nb.cells[1]
    assert cell.raw_hash == hashlib.sha256(raw_source.encode()).hexdigest()
    assert cell.clean_hash != cell.raw_hash


def _markdown_notebook(source: str) -> RawNotebook:
    cell = RawCell(index=0, cell_type=CellType.MARKDOWN, source=source)
    return RawNotebook(path="x.ipynb", file_hash="f", cells=(cell,))


def test_a_secret_shaped_string_inside_a_payload_is_not_reported(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    lookalike = "AKIA" + "ABCDEFGHIJKLMNOP"  # random base64 can look like a key id
    nb = clean_notebook(
        _markdown_notebook(f"![x](data:image/png;base64,{lookalike})"),
        rules,
        20,
        tiny_tokenizer,
        "p",
    )
    assert nb.cells[0].source == "![x](data:image/png;base64,[removed])"
    assert "secrets_aws_access_key_id" not in nb.counts


def test_a_real_secret_next_to_an_image_is_still_scrubbed(
    rules: CleaningRules, tiny_tokenizer: Tokenizer
) -> None:
    key = "AKIA" + "ABCDEFGHIJKLMNOP"
    source = f"key {key}\n\n![x](data:image/png;base64,QUJDRA==)"
    nb = clean_notebook(_markdown_notebook(source), rules, 20, tiny_tokenizer, "p")
    assert key not in nb.cells[0].source
    assert nb.counts["secrets_aws_access_key_id"] == 1
    assert nb.counts["inline_data_uri_removed"] == 1


# ------------------------------------------------------- rules file, both rules


def _without_uri_rule() -> str:
    text = RULES_PATH.read_text(encoding="utf-8")
    return text[: text.index("\n  # Added by proposal 001")] + "\n"


def test_both_rules_load(rules: CleaningRules) -> None:
    assert rules.inline_data_uri.replacement == "[removed]"
    assert {"head", "payload"} <= set(rules.inline_data_uri.pattern.groupindex)


def test_missing_uri_rule_is_a_rules_error(tmp_path: Path) -> None:
    with pytest.raises(RulesError, match="invalid"):
        load_rules(_write(tmp_path, _without_uri_rule()))


def test_duplicate_rule_id_is_a_rules_error(tmp_path: Path) -> None:
    # Two complete instructor_note blocks and no inline_data_uri block: two rules, one missing.
    text = RULES_PATH.read_text(encoding="utf-8")
    note_block = text[
        text.index("  - id: instructor_note") : text.index("\n  # Added by proposal 001")
    ]
    doubled = text[: text.index("  - id: instructor_note")] + note_block + "\n" + note_block + "\n"
    with pytest.raises(RulesError, match="exactly once"):
        load_rules(_write(tmp_path, doubled))


def test_uri_pattern_without_the_named_groups_is_rejected(tmp_path: Path) -> None:
    text = RULES_PATH.read_text(encoding="utf-8").replace("(?P<payload>", "(")
    with pytest.raises(RulesError, match="lacks groups"):
        load_rules(_write(tmp_path, text))


def test_invalid_uri_regex_is_a_rules_error(tmp_path: Path) -> None:
    text = RULES_PATH.read_text(encoding="utf-8").replace("(?P<head>", "(?P<head")
    with pytest.raises(RulesError, match="invalid regex for inline_data_uri pattern"):
        load_rules(_write(tmp_path, text))


def test_non_string_replacement_is_rejected(tmp_path: Path) -> None:
    text = RULES_PATH.read_text(encoding="utf-8").replace(
        'replacement: "[removed]"', "replacement: [a, b]"
    )
    with pytest.raises(RulesError, match="invalid"):
        load_rules(_write(tmp_path, text))
