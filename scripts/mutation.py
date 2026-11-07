"""Mutation testing of the pure core (CLAUDE.md section 3; `make mutation`).

For each target module, every mutation site is changed one at a time (a comparison swapped, `+`
and `-` exchanged, `and` and `or` exchanged, an integer moved by one, a boolean flipped, a `not`
dropped), the module's tests run, and the mutant counts as killed if any test fails. The mutants
are written into a temporary copy of the tree, so the checkout is never touched. Mutants that
survive are listed with file, line and change: each is either a test to write or an equivalent
change to explain. Results go to `results/mutation.json` (file names, lines and counts only).
mutmut 3.8.1 is pinned but cannot run this src-layout project (it copies only the mutated files
into `mutants/`, where the package then does not import), so this runner does the same job with
the standard library.
"""

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
S = "src/noterecall/"
TARGETS: dict[str, list[str]] = {
    S + "retrieval/fusion.py": ["tests/unit/test_fusion.py"],
    S + "eval/metrics.py": ["tests/unit/test_metrics.py"],
    S + "generate/citations.py": ["tests/unit/test_citations.py"],
    S + "serve/ratelimit.py": ["tests/unit/test_ratelimit.py", "tests/unit/test_serve.py"],
}
SWAPS: dict[type[ast.AST], Callable[[], Any]] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Add: ast.Sub, ast.Sub: ast.Add,
    ast.Mult: ast.Div, ast.Div: ast.Mult, ast.And: ast.Or, ast.Or: ast.And,
}  # fmt: skip
TIMEOUT_S = 300


@dataclass(frozen=True)
class Site:
    index: int
    line: int
    change: str


class Mutator(ast.NodeTransformer):
    """Counts mutation sites; applies the one numbered `target` (or none when it is -1)."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.count = 0
        self.sites: list[Site] = []
        self._docstrings: set[int] = set()

    def _hit(self, node: ast.AST, change: str) -> bool:
        index = self.count
        self.count += 1
        self.sites.append(Site(index, getattr(node, "lineno", 0), change))
        return index == self.target

    def visit_Module(self, node: ast.Module) -> ast.AST:
        for scope in ast.walk(node):
            if isinstance(
                scope, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                first = scope.body[0] if scope.body else None
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    self._docstrings.add(id(first.value))
        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            swap = SWAPS.get(type(op))
            if swap and self._hit(node, f"{type(op).__name__} -> {swap.__name__}"):
                node.ops[i] = swap()
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        swap = SWAPS.get(type(node.op))
        if swap and self._hit(node, f"{type(node.op).__name__} -> {swap.__name__}"):
            node.op = swap()
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        swap = SWAPS[type(node.op)]
        if self._hit(node, f"{type(node.op).__name__} -> {swap.__name__}"):
            node.op = swap()
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self._hit(node, "not removed"):
            return node.operand
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if id(node) in self._docstrings:
            return node
        if isinstance(node.value, bool):
            if self._hit(node, f"{node.value} -> {not node.value}"):
                return ast.copy_location(ast.Constant(not node.value), node)
        elif isinstance(node.value, int) and self._hit(node, f"{node.value} -> {node.value + 1}"):
            return ast.copy_location(ast.Constant(node.value + 1), node)
        return node


def sites_of(tree: ast.Module) -> list[Site]:
    mutator = Mutator(-1)
    mutator.visit(copy.deepcopy(tree))
    return mutator.sites


def mutant_source(tree: ast.Module, index: int) -> str:
    mutator = Mutator(index)
    return ast.unparse(ast.fix_missing_locations(mutator.visit(copy.deepcopy(tree))))


def tests_fail(work: Path, tests: list[str]) -> bool:
    try:
        run = subprocess.run(  # noqa: S603 - fixed argument list
            [sys.executable, "-m", "pytest", *tests, "-x", "-q", "-p", "no:cacheprovider"],
            cwd=work, capture_output=True, timeout=TIMEOUT_S, check=False,
            env={**os.environ, "PYTHONPATH": str(work / "src")},
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return True  # a mutant that hangs is detected
    return run.returncode != 0


def run_file(work: Path, rel: str, tests: list[str]) -> dict[str, object]:
    path = work / rel
    original = path.read_text(encoding="utf-8")
    if tests_fail(work, tests):
        raise SystemExit(f"the tests of {rel} fail without any mutation; fix them first")
    tree = ast.parse(original)
    survivors: list[dict[str, object]] = []
    sites = sites_of(tree)
    try:
        for site in sites:
            path.write_text(mutant_source(tree, site.index), encoding="utf-8")
            killed = tests_fail(work, tests)
            print(
                f"{'killed' if killed else 'SURVIVED':9s} {rel}:{site.line} {site.change}",
                flush=True,
            )
            if not killed:
                survivors.append({"line": site.line, "change": site.change})
    finally:
        path.write_text(original, encoding="utf-8")
    return {"mutants": len(sites), "killed": len(sites) - len(survivors), "survivors": survivors}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "tree"  # mutants are written into a copy, never into the checkout
        shutil.copytree(
            ROOT,
            work,
            ignore=shutil.ignore_patterns(".git", ".venv", "data", "mutants", "__pycache__"),
        )
        results = {rel: run_file(work, rel, tests) for rel, tests in TARGETS.items()}
    total = sum(int(str(r["mutants"])) for r in results.values())
    killed = sum(int(str(r["killed"])) for r in results.values())
    out = {"tool": "scripts/mutation.py", "mutants": total, "killed": killed, "files": results}
    target = ROOT / "results" / "mutation.json"
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"mutation: {killed}/{total} killed; wrote {target}")


if __name__ == "__main__":
    main()
