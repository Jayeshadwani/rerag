"""Splitters: spans tile the text exactly, and cut at the units named in proposal 002."""

import itertools
import warnings

import pytest

from noterecall.chunking.splitters import (
    Heading,
    Span,
    code_statement_spans,
    markdown_blocks,
    markdown_headings,
    sentence_spans,
)


def _slices(text: str, spans: list[Span]) -> list[str]:
    assert spans[0].start == 0
    assert spans[-1].end == len(text)
    assert all(a.end == b.start for a, b in itertools.pairwise(spans))
    return [text[s.start : s.end] for s in spans]


def test_code_is_cut_at_top_level_statements() -> None:
    source = (
        "import os\n# about x\nx = 1\n\n@dec\ndef f():\n    return 2\n\n\ny = 3; z = 4\n# tail\n"
    )
    spans = code_statement_spans(source)
    assert spans is not None
    assert _slices(source, spans) == [
        "import os\n",  # statement 1
        "# about x\nx = 1\n",  # a leading comment travels with its statement
        "\n@dec\ndef f():\n    return 2\n",  # blank line before; decorator and body in one piece
        "\n\ny = 3; z = 4\n# tail\n",  # two statements on one line stay together; tail comment kept
    ]


def test_compound_statements_and_multiline_strings_are_single_units() -> None:
    source = (
        'for i in range(3):\n    print(i)\n\nx = """a\n\nb"""\nif x:\n    pass\nelse:\n    pass\n'
    )
    spans = code_statement_spans(source)
    assert spans is not None
    assert _slices(source, spans) == [
        "for i in range(3):\n    print(i)\n",
        '\nx = """a\n\nb"""\n',  # the blank line inside the string is not a boundary
        "if x:\n    pass\nelse:\n    pass\n",
    ]


def test_ipython_magics_are_one_line_statements() -> None:
    source = "%matplotlib inline\nimport os\n!pip install x\n"
    spans = code_statement_spans(source)
    assert spans is not None
    assert _slices(source, spans) == ["%matplotlib inline\n", "import os\n", "!pip install x\n"]


@pytest.mark.parametrize("source", ["def f(:\n", "x = (1,\n", "    unexpected_indent = 1\n"])
def test_unparseable_code_returns_none(source: str) -> None:
    assert code_statement_spans(source) is None


def test_comment_only_and_empty_cells_are_one_span() -> None:
    assert code_statement_spans("# just a comment\n") == [Span(0, 17)]  # 16 chars + newline
    assert code_statement_spans("") == [Span(0, 0)]


def test_old_style_escapes_do_not_emit_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert code_statement_spans('import re\nre.compile("\\d+")\nx = "\\.\\s"\n') is not None


def test_markdown_blocks_split_paragraphs_and_fences() -> None:
    text = "Intro line\nsecond line\n\n\nPara two\n```py\n# not a heading\nx=1\n```\nafter\n"
    blocks = markdown_blocks(text)
    assert _slices(text, [b.span for b in blocks]) == [
        "Intro line\nsecond line\n\n\n",  # trailing blank lines stay with the paragraph
        "Para two\n",  # a fence interrupts a paragraph even without a blank line
        "```py\n# not a heading\nx=1\n```\n",
        "after\n",
    ]
    assert [b.fenced for b in blocks] == [False, False, True, False]


def test_unclosed_fence_runs_to_the_end() -> None:
    text = "before\n\n```\ncode\nmore code"
    blocks = markdown_blocks(text)
    assert _slices(text, [b.span for b in blocks]) == ["before\n\n", "```\ncode\nmore code"]
    assert blocks[-1].fenced is True


def test_longer_fence_is_not_closed_by_a_shorter_one() -> None:
    text = "````\n```\ninner\n```\n````\nafter\n"
    blocks = markdown_blocks(text)
    assert _slices(text, [b.span for b in blocks]) == ["````\n```\ninner\n```\n````\n", "after\n"]


def test_tilde_fence_and_info_string_with_backtick_rule() -> None:
    text = "~~~\nx\n~~~\n``` not `a` fence start\n"
    blocks = markdown_blocks(text)
    assert [b.fenced for b in blocks] == [True, False]  # a backtick fence's info has no backtick


def test_leading_blank_lines_join_the_first_block() -> None:
    text = "\n\nfirst\n\nsecond"
    assert _slices(text, [b.span for b in markdown_blocks(text)]) == ["\n\nfirst\n\n", "second"]


def test_empty_markdown_is_one_empty_block() -> None:
    assert [b.span for b in markdown_blocks("")] == [Span(0, 0)]


def test_headings_outside_fences_only() -> None:
    text = "# A\ntext\n```\n# no\n```\n## B ##\n####### too deep\n#nospace\n#\n"
    assert markdown_headings(text) == [Heading(1, "A"), Heading(2, "B")]


def test_headings_may_be_indented_up_to_three_spaces() -> None:
    assert markdown_headings("   ### Three\n    #### four spaces is code\n") == [
        Heading(3, "Three")
    ]


def test_sentences_split_after_terminal_punctuation_and_whitespace() -> None:
    text = "First sentence. Second one! Third `code. not` split? Yes."
    spans = sentence_spans(text, Span(0, len(text)))
    assert _slices(text, spans) == [
        "First sentence. ",
        "Second one! ",
        "Third `code. not` split? ",  # no break inside an inline code span
        "Yes.",
    ]


def test_sentence_split_respects_the_given_span() -> None:
    text = "ignored. Alpha one. Beta two. ignored"
    spans = sentence_spans(text, Span(9, 29))
    assert [text[s.start : s.end] for s in spans] == ["Alpha one. ", "Beta two."]


def test_no_split_before_lowercase_or_inside_numbers() -> None:
    text = "It costs 3.5 dollars e.g. a lot. but see v2.0 now"
    assert len(sentence_spans(text, Span(0, len(text)))) == 1


def test_a_closing_fence_may_be_indented_at_most_three_spaces() -> None:
    inner_four = (
        "```\nx\n    ```\ny\n```\n"  # four spaces: not a closing fence, the block continues
    )
    assert [b.span for b in markdown_blocks(inner_four)] == [Span(0, len(inner_four))]
    closed_three = "```\nx\n   ```\ny\n"  # three spaces: closes
    assert _slices(closed_three, [b.span for b in markdown_blocks(closed_three)]) == [
        "```\nx\n   ```\n",
        "y\n",
    ]


def test_text_after_a_closing_fence_marker_means_it_does_not_close() -> None:
    text = "```\nx\n``` still inside\n```\nafter\n"
    assert _slices(text, [b.span for b in markdown_blocks(text)]) == [
        "```\nx\n``` still inside\n```\n",
        "after\n",
    ]


def test_heading_text_is_never_empty() -> None:
    assert markdown_headings("# #\n## a ##\n") == [Heading(1, "#"), Heading(2, "a")]
