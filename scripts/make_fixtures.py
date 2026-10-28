"""Generate the synthetic fixture notebooks under tests/fixtures/notebooks.

The notebooks are fully synthetic and deterministic (fixed cell ids, no timestamps),
so `tests/unit/test_fixtures.py` can regenerate them and compare with the committed files.
The "secrets" in 05 are made-up values (AWS's documented example credentials and an
obviously fake Anthropic-style key); they are not real credentials.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import nbformat
from nbformat import NotebookNode
from nbformat import v4 as nbf

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "notebooks"

# 1x1 transparent PNG, standard base64.
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEA"
    "wS2OUAAAAABJRU5ErkJggg=="
)

# Shared by 02 (cross-notebook duplicate with 04) and used twice inside 04.
SHARED_SPLIT_CELL = (
    "from sklearn.model_selection import train_test_split\n"
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)"
)

# Payloads for the inline-image fixture. They are synthetic base64-alphabet strings, not images.
URI_PAYLOADS = {
    "png_long": "QUJD" * 100,  # 400 characters
    "jpeg_mid": "AAAA" * 50,  # 200 characters
    "gif_short": "R0lGODlh",  # 8 characters
    "png_short": "iVBORw0K",  # 8 characters
    "png_last": "QUJDREVGR0g=",  # 12 characters
}

FAKE_AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
FAKE_AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # noqa: S105 (AWS docs example)
FAKE_ANTHROPIC_KEY = "sk-ant-api03-" + "FAKE" * 10 + "-NOTAREALKEY"


def _notebook(cells: list[NotebookNode]) -> NotebookNode:
    nb: NotebookNode = nbf.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    return nb


def _md(cell_id: str, source: str) -> NotebookNode:
    cell: NotebookNode = nbf.new_markdown_cell(source, id=cell_id)
    return cell


def _code(cell_id: str, source: str, outputs: list[NotebookNode] | None = None) -> NotebookNode:
    count = 1 if outputs else None
    cell: NotebookNode = nbf.new_code_cell(
        source, id=cell_id, outputs=outputs or [], execution_count=count
    )
    return cell


def _stdout(text: str) -> NotebookNode:
    output: NotebookNode = nbf.new_output("stream", name="stdout", text=text)
    return output


def no_headings() -> NotebookNode:
    """Markdown cells with prose only; no line starts with `#`."""
    return _notebook(
        [
            _md("a1", "Quick notes on pandas grouping. I keep forgetting the syntax."),
            _code("a2", "import pandas as pd\ndf = pd.DataFrame({'k': [1, 1, 2], 'v': [3, 4, 5]})"),
            _md("a3", "Group by a key, then aggregate with the mean of each group."),
            _code("a4", "df.groupby('k')['v'].mean()", [_stdout("k\n1    3.5\n2    5.0\n")]),
            _code("a5", ""),
            _md("a6", "Remember that groupby drops rows whose key is NaN by default."),
        ]
    )


def deep_headings() -> NotebookNode:
    """Headings from H1 to H4, including a heading with no body and a skipped level."""
    return _notebook(
        [
            _md("b1", "# Supervised Learning"),
            _md("b2", "## Model evaluation"),
            _md("b3", "### Splitting data"),
            _md("b4", "#### Hold-out split\nKeep a test set the model never sees."),
            _code("b5", SHARED_SPLIT_CELL),
            _md("b6", "#### Cross-validation\nUse `cross_val_score` for k-fold estimates."),
            _code("b7", "from sklearn.model_selection import cross_val_score"),
            _md("b8", "## Regularisation"),
            _md("b9", "##### Ridge versus lasso\nThis jumps from H2 to H5 on purpose."),
        ]
    )


def huge_output() -> NotebookNode:
    """A single cell printing 5000 lines."""
    lines = "".join(f"epoch {i:04d} loss={1 / (i + 1):.6f}\n" for i in range(5000))
    return _notebook(
        [
            _md("c1", "# Training log"),
            _code("c2", "for i in range(5000):\n    print(i)", [_stdout(lines)]),
            _md("c3", "The loss falls monotonically."),
        ]
    )


def image_output() -> NotebookNode:
    """Image, HTML and text outputs; a cell shared with 02 and one repeated inside this notebook."""
    plot = nbf.new_output(
        "display_data",
        data={"image/png": TINY_PNG_B64, "text/plain": "<Figure size 640x480>"},
    )
    table = nbf.new_output(
        "execute_result",
        data={"text/html": "<table><tr><td>1</td></tr></table>", "text/plain": "   0\n0  1"},
        execution_count=2,
        metadata={},
    )
    return _notebook(
        [
            _md("d1", "# Plots"),
            _code("d2", "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])", [plot]),
            _code("d3", "df.head()", [table]),
            _code("d4", SHARED_SPLIT_CELL),
            _md("d5", "Same split again, repeated in this notebook."),
            _code("d6", SHARED_SPLIT_CELL),
        ]
    )


def fake_secret() -> NotebookNode:
    """Synthetic credentials in a code cell, a markdown cell and an output."""
    return _notebook(
        [
            _md("e1", "# Calling the API"),
            _code(
                "e2",
                f"import os\nos.environ['ANTHROPIC_API_KEY'] = '{FAKE_ANTHROPIC_KEY}'",
            ),
            _code(
                "e3",
                f"aws_access_key_id = '{FAKE_AWS_ACCESS_KEY}'\n"
                f"aws_secret_access_key = '{FAKE_AWS_SECRET_KEY}'",
                [_stdout(f"loaded key {FAKE_AWS_ACCESS_KEY}\n")],
            ),
            _md("e4", f"Do not paste keys like {FAKE_ANTHROPIC_KEY} into notebooks."),
        ]
    )


def instructor_notes() -> NotebookNode:
    """One cell per instructor-note style, plus look-alikes that must NOT be treated as notes.

    All wording is invented; only the markup patterns mirror the styles found in real notebooks.
    Cell ids starting with `n` are notes, `k` are cells that must be left alone.
    """
    return _notebook(
        [
            _md("k1", "# Sorting basics"),
            _md(
                "n2",
                '<span style="background-color: red;"> **[Note to Instructor]:** </span> '
                "Pause here and let the group try it first.",
            ),
            _md(
                "n3",
                "Sorting arranges items in order.\n\n"
                "<span style=background-color:red>**Instructor Note:**</span> "
                "Mention stability.\n\n"
                "Stable means equal items keep their original order.",
            ),
            _md(
                "n4",
                "> **Instructor Note:**\n- Draw the two lists on the board.\n"
                "- Keep this under two minutes.\n\nNow compare the two approaches.",
            ),
            _md(
                "n5",
                "Next, complexity.\n\n<font color='red'> Instructor Note </font>\n\n"
                "Do not derive the bound in class.\n\nPoint to the reading list instead.",
            ),
            _md(
                "n6",
                "### Space usage\n\n<FONT color=red>Instructor Note:</FONT>\n\n"
                "Skip this slide if time is short.\n\n---\n\n"
                "In-place sorts reuse the input list.",
            ),
            _md(
                "n7",
                "> <font color='red'>INSTRUCTOR NOTES - Optional</font>\n\n"
                "**Why merge sort?**\n\nIt is a classic divide and conquer example.\n\n"
                "## Quick recap\n\nSorting is everywhere.",
            ),
            _md(
                "n8",
                "> **<font color='red'>INSTRUCTOR NOTES</font> refrence only**\n\n"
                "A longer derivation lives in the appendix.",
            ),
            _md(
                "n9",
                "Steps:\n- Pick a pivot\n"
                '    - <span style="background-color: red;"> **[Note to Instructor]:**</span> '
                "Show the animation here.\n"
                "- Partition the rest\n- Recurse on both halves",
            ),
            _md("k2", "- Checking instructor ratings helps choose a course."),
            _md("k3", "Fine-tune the model with the Transformers Trainer class."),
            _md("k4", "The instructor notes are posted on the course page."),
            _md(
                "k5",
                '<span style="background-color: red;"> **[Ask Learners]:** </span> '
                "Which sort would you pick?",
            ),
            _md("k6", '<span style="color: red;"> **Note:** </span> Python lists are mutable.'),
            _code("k7", "# Instructor note: this comment is inside a code cell"),
        ]
    )


def inline_images() -> NotebookNode:
    """Markdown cells with inline base64 data URIs in several shapes, plus look-alikes.

    Cells `i2`, `i3`, `i4`, `i8` hold 5 data URIs in total (payloads of 400, 200, 8, 8 and 12
    characters = 628). `i5`, `i6` (no payload) and the code cell `i7` must be left alone.
    """
    p = URI_PAYLOADS
    return _notebook(
        [
            _md("i1", "# Diagrams"),
            _md("i2", f"![flow](data:image/png;base64,{p['png_long']})"),
            _md(
                "i3", f'Before <img src="data:image/jpeg;base64,{p["jpeg_mid"]}" width="40"> after'
            ),
            _md(
                "i4",
                f"![a](data:image/gif;base64,{p['gif_short']})\n\n"
                f"![b](data:image/png;base64,{p['png_short']})",
            ),
            _md("i5", "![remote](https://example.com/a.png)"),
            _md(
                "i6",
                "The data:image/png;base64 syntax embeds files. "
                "A prefix like data:text/plain;base64, with no payload is left alone.",
            ),
            _code("i7", f'img = "data:image/png;base64,{p["png_last"]}"'),
            _md("i8", f"![x](data:image/png;base64,{p['png_last']})"),
        ]
    )


BUILDERS: dict[str, Callable[[], NotebookNode]] = {
    "01_no_headings.ipynb": no_headings,
    "02_deep_headings.ipynb": deep_headings,
    "03_huge_output.ipynb": huge_output,
    "04_image_output.ipynb": image_output,
    "05_fake_secret.ipynb": fake_secret,
    "06_instructor_notes.ipynb": instructor_notes,
    "07_inline_images.ipynb": inline_images,
}


def build_fixtures() -> dict[str, NotebookNode]:
    """Return every fixture notebook keyed by file name, validated against the nbformat schema."""
    notebooks = {name: build() for name, build in BUILDERS.items()}
    for nb in notebooks.values():
        nbformat.validate(nb)
    return notebooks


def main(argv: Sequence[str] | None = None) -> int:
    """Write the fixtures to `--out` (default: tests/fixtures/notebooks)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--out", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, nb in build_fixtures().items():
        nbformat.write(nb, args.out / name)
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
