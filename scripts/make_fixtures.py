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


BUILDERS: dict[str, Callable[[], NotebookNode]] = {
    "01_no_headings.ipynb": no_headings,
    "02_deep_headings.ipynb": deep_headings,
    "03_huge_output.ipynb": huge_output,
    "04_image_output.ipynb": image_output,
    "05_fake_secret.ipynb": fake_secret,
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
