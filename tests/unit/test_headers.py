"""The deterministic context header (ADR-007)."""

from noterecall.enrich.headers import chunk_header, clean_heading, index_text, notebook_title
from noterecall.models import Chunk


def chunk(
    path: str = "ml/01_intro.ipynb", headings: tuple[str, ...] = (), text: str = "t"
) -> Chunk:
    return Chunk(
        id=f"{path}#{text}",
        notebook_path=path,
        heading_path=headings,
        pieces=(),
        text=text,
        tokens=1,
        cell_hashes=(),
    )


def test_clean_heading_strips_tags_marks_and_spaces_but_keeps_underscores() -> None:
    assert clean_heading("**<b>Hello</b>**   `train_test_split`\n  now") == (
        "Hello train_test_split now"
    )
    assert clean_heading("<br>") == ""


def test_notebook_title_is_the_file_stem() -> None:
    assert notebook_title("ml/week 1/01_intro.v2.ipynb") == "01_intro.v2"


def test_chunk_header_joins_title_and_every_heading_level_and_skips_empty_ones() -> None:
    c = chunk(headings=("Intro", "<span>Loss</span>", "", "`mse`"))
    assert chunk_header(c) == "01_intro > Intro > Loss > mse"
    assert chunk_header(chunk(headings=())) == "01_intro"


def test_index_text_puts_the_header_in_front_and_leaves_the_chunk_alone() -> None:
    c = chunk(headings=("A",), text="body")
    assert index_text(c, False) == "body"
    assert index_text(c, True) == "01_intro > A\n\nbody"
    assert c.text == "body"  # what is cited and shown is unchanged
