"""Search, rendering and the labelling dialogue (synthetic cells)."""

from noterecall.eval.labeling import (
    PAGE,
    Entry,
    QuerySession,
    build_entries,
    grep,
    render_entry,
    search,
    words,
)
from noterecall.models import Cell, CellType


def cell(index: int, cell_type: CellType, source: str, raw: str | None = None) -> Cell:
    h = raw if raw is not None else f"hash{index}"
    return Cell(index=index, cell_type=cell_type, raw_hash=h, clean_hash=h, source=source)


MD, CODE = CellType.MARKDOWN, CellType.CODE


def entries() -> list[Entry]:
    nb1 = [
        cell(0, MD, "# Intro\nalpha beta"),
        cell(1, CODE, "beta beta gamma"),
        cell(2, MD, "## Part two"),
        cell(3, CODE, "delta", raw="shared"),
        cell(4, CODE, "delta", raw="shared"),  # identical cell in the same notebook
        cell(5, CODE, "   "),  # blank: never offered
    ]
    nb2 = [cell(0, CODE, "delta", raw="shared"), cell(1, CODE, "epsilon")]
    return build_entries([("one.ipynb", nb1), ("two.ipynb", nb2)])


def test_words_lowercases_and_splits_on_non_word_characters() -> None:
    assert words("Train_Test-split(X, y)") == ["train_test", "split", "x", "y"]


def test_build_entries_dedupes_identical_cells_and_counts_copies() -> None:
    es = entries()
    assert [(e.notebook_path, e.cell_index) for e in es] == [
        ("one.ipynb", 0), ("one.ipynb", 1), ("one.ipynb", 2), ("one.ipynb", 3),
        ("two.ipynb", 0), ("two.ipynb", 1),
    ]  # fmt: skip
    shared = es[3]
    assert (shared.same_notebook_copies, shared.other_notebooks) == (2, 1)
    assert es[4].other_notebooks == 1
    assert es[0].other_notebooks == 0
    assert es[0].key == ("one.ipynb", "hash0")


def test_heading_paths_follow_the_chunker_sections() -> None:
    es = entries()
    assert es[0].heading_path == ("Intro",)
    assert es[1].heading_path == ("Intro",)
    assert es[2].heading_path == ("Intro", "Part two")
    assert es[3].heading_path == ("Intro", "Part two")
    assert es[4].heading_path == ()  # second notebook has no headings


def test_search_ranks_by_tf_idf_and_omits_non_matches() -> None:
    es = entries()  # 6 entries
    # "beta": df = 2 (cells 0 and 1). idf = ln(1 + 6/2) = ln 4. tf: cell0 1, cell1 2.
    # cell1 scores ln4 * (1 + ln 2) > cell0's ln4 * 1, so cell1 ranks first.
    assert [(e.notebook_path, e.cell_index) for e in search(es, "beta")] == [
        ("one.ipynb", 1),
        ("one.ipynb", 0),
    ]
    assert search(es, "nothing") == []
    assert search(es, "") == []
    # two words add their scores: "alpha beta" -> cell 0 has both (ln 7 + ln 4), cell 1 only beta
    top = search(es, "alpha beta")
    assert [e.cell_index for e in top][:2] == [0, 1]


def test_search_ties_break_by_corpus_order_and_limit_applies() -> None:
    es = entries()
    hits = search(es, "delta")  # shared cell in nb1 and the same hash in nb2: equal scores
    assert [(e.notebook_path, e.cell_index) for e in hits] == [("one.ipynb", 3), ("two.ipynb", 0)]
    assert len(search(es, "delta", limit=1)) == 1


def test_grep_is_a_case_insensitive_substring_in_corpus_order() -> None:
    es = entries()
    assert [e.cell_index for e in grep(es, "ETA")] == [0, 1]  # "beta" in cells 0 and 1
    assert grep(es, "zzz") == []
    assert len(grep(es, "e", limit=2)) == 2


def test_render_shows_location_marks_notes_and_trims_long_source() -> None:
    es = entries()
    text = render_entry(es[3], 2, marked=True)
    assert text.startswith("[x] 2. one.ipynb cell 3 (code)")
    assert "2 identical cells here" in text
    assert "also in 1 other notebook(s)" in text
    assert "Intro > Part two" in text
    assert "    | delta" in text
    long = es[0]._replace(source="x" * 1000)
    shown = render_entry(long, 1, marked=False)
    assert shown.startswith("[ ] 1.")
    assert shown.endswith("x ...")
    assert len(shown) < 700
    assert "(no heading)" in render_entry(es[0]._replace(heading_path=()), 1, False)


def run(session: QuerySession, *lines: str) -> list[str]:
    return [session.handle(line) for line in lines]


def test_mark_search_and_done() -> None:
    session = QuerySession(entries(), "q", unanswerable=False)
    out = run(session, "beta", "/mark 1", "/marked")
    assert "[ ] 1. one.ipynb cell 1" in out[0]
    assert out[1] == "1 cell(s) marked"
    assert out[2].startswith("[x] 1. one.ipynb cell 1")
    assert session.handle("/done") == "saved"
    assert session.status == "done"
    assert session.marked == {("one.ipynb", "hash1")}


def test_marked_cells_show_as_marked_in_later_listings() -> None:
    session = QuerySession(entries(), "q", False)
    run(session, "beta", "/mark 2")
    assert "[x] 1. one.ipynb cell 0" not in session.handle("beta")
    assert "[x] 2. one.ipynb cell 0" in session.handle("beta")


def test_mark_all_copies_marks_identical_cells_in_every_notebook() -> None:
    session = QuerySession(entries(), "q", False)
    run(session, "delta")
    assert session.handle("/mark 1") == "1 cell(s) marked"
    assert session.handle("/markall 1") == "2 cell(s) marked"
    assert session.marked == {("one.ipynb", "shared"), ("two.ipynb", "shared")}


def test_unmark_and_input_errors() -> None:
    session = QuerySession(entries(), "q", False)
    run(session, "beta", "/mark 1 2")
    assert session.handle("/unmark 1") == "1 cell(s) marked"
    assert "not a number on the current page" in session.handle("/mark 9")
    assert "not a number on the current page" in session.handle("/mark x")
    assert "give one or more numbers" in session.handle("/mark")
    assert "give one or more numbers" in session.handle("/unmark")
    assert session.handle("/unmark 7").startswith("'7' is not a number")
    assert "unknown command /nope" in session.handle("/nope")
    assert session.handle("") == QuerySession.HELP
    assert session.handle("/help") == QuerySession.HELP


def test_nothing_marked_and_no_results_messages() -> None:
    session = QuerySession(entries(), "q", False)
    assert session.handle("/marked") == "nothing marked"
    assert session.handle("qqqq") == "no cells match"
    assert session.handle("/grep zzzz") == "no cells match"


def test_paging_with_more() -> None:
    many = build_entries([("n.ipynb", [cell(i, CODE, f"word{i} common") for i in range(PAGE + 3)])])
    session = QuerySession(many, "q", False)
    first = session.handle("common")
    assert f"{PAGE}. n.ipynb" in first
    assert "... 3 more (/more)" in first
    second = session.handle("/more")
    assert "3. n.ipynb" in second
    assert "1. n.ipynb cell 8" in second  # numbering restarts on each page
    assert session.handle("/more") == "nothing more to show"
    assert session.handle("/mark 1") == "1 cell(s) marked"
    assert ("n.ipynb", "hash8") in session.marked


def test_notebook_listing_and_browsing() -> None:
    session = QuerySession(entries(), "q", False)
    assert session.handle("/nb") == "1. one.ipynb\n2. two.ipynb"
    page = session.handle("/nb 2")
    assert "two.ipynb cell 0" in page
    assert "two.ipynb cell 1" in page
    assert "one.ipynb" not in page
    assert session.handle("/nb 3") == "give a notebook number from /nb"
    assert session.handle("/nb x") == "give a notebook number from /nb"


def test_done_needs_a_label_for_answerable_queries() -> None:
    session = QuerySession(entries(), "q", False)
    assert session.handle("/done") == "mark at least one relevant cell, or /abort"
    assert session.status == "open"


def test_abort_discards() -> None:
    session = QuerySession(entries(), "q", False)
    run(session, "beta", "/mark 1")
    assert session.handle("/abort") == "discarded"
    assert session.status == "aborted"


def test_unanswerable_needs_confirmation_and_no_labels() -> None:
    session = QuerySession(entries(), "alpha", unanswerable=True)
    first = session.handle("/done")
    assert "one.ipynb cell 0" in first  # the closest cell is shown
    assert "/done again to confirm" in first
    assert session.status == "open"
    run(session, "alpha")  # any other command resets the confirmation
    assert "/done again to confirm" in session.handle("/done")
    assert session.handle("/done") == "saved"
    assert session.status == "done"
    assert session.marked == set()


def test_unanswerable_with_marks_is_refused() -> None:
    session = QuerySession(entries(), "alpha", True)
    run(session, "beta", "/mark 1")
    assert "must have no labels" in session.handle("/done")
    assert session.status == "open"


def test_initial_marks_support_editing_and_orphans_are_listed() -> None:
    session = QuerySession(
        entries(), "q", False, initial=[("one.ipynb", "hash1"), ("gone.ipynb", "deadbeef00")]
    )
    listing = session.handle("/marked")
    assert "one.ipynb cell 1" in listing
    assert "[x] 1. gone.ipynb (no stored cell deadbeef)" in listing
    assert session.handle("/done") == "saved"
    assert ("gone.ipynb", "deadbeef00") in session.marked
