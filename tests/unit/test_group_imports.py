"""The commands an image runs must not need packages from other dependency groups.

The image build runs `sync`, `quantize-onnx` and `build-index` with the build-image group
(no torch, no sentence-transformers); the API image has neither those nor typer. Here the evaluation
packages are made unimportable and the commands are loaded anyway. `scripts/check_groups.sh` (CI job
`groups`) does the same with real, separate installs.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BLOCK = (
    "import sys\n"
    "for name in ('sentence_transformers', 'torch', 'transformers'):\n"
    "    sys.modules[name] = None\n"
)


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter, code from this file
        [sys.executable, "-I", "-c", BLOCK + code],
        capture_output=True, text=True, check=False, env={"PYTHONPATH": str(ROOT / "src")},
    )  # fmt: skip


@pytest.mark.parametrize("args", [[], ["sync"], ["quantize-onnx"], ["build-index"]])
def test_build_commands_load_without_the_evaluation_packages(args: list[str]) -> None:
    code = (
        f"sys.argv = ['noterecall', *{args!r}, '--help']\nfrom noterecall.cli import app\napp()\n"
    )
    done = run(code)
    assert done.returncode == 0, done.stderr[-500:]
    assert "Usage" in done.stdout


def test_the_api_loads_without_the_evaluation_packages_or_typer() -> None:
    done = run("sys.modules['typer'] = None\nimport noterecall.api, noterecall.serve.runtime\n")
    assert done.returncode == 0, done.stderr[-500:]
