"""Banned-pattern check for src/ (CLAUDE.md section 2). Exits 1 if any pattern is found.

Banned: the marker words TODO/FIXME/XXX (anywhere in the text, comments and docstrings
included), any mention of NotImplementedError, functions whose body is only `...` or
`pass` (a docstring does not count as a body), `except` clauses that catch
Exception/BaseException (bare, named, aliased, or inside a tuple), and functions whose
body is longer than 40 lines (the docstring is not counted). `...` is allowed only as the
body of a method inside a `Protocol` class or an `@overload` function.
"""

import argparse
import ast
import re
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

MARKER_RE = re.compile(r"\b(TODO|FIXME|XXX)\b")
NOT_IMPLEMENTED_RE = re.compile(r"\bNotImplementedError\b")
BROAD_EXCEPTIONS = {"Exception", "BaseException"}
SCANNED_SUFFIXES = {".py", ".md"}
MAX_BODY_LINES = 40

type FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class Violation(NamedTuple):
    """One banned pattern at a file location."""

    path: Path
    line: int
    message: str

    def render(self) -> str:
        """Format as `path:line: message`."""
        return f"{self.path}:{self.line}: {self.message}"


def _scan_text(path: Path, text: str) -> Iterator[Violation]:
    for number, line in enumerate(text.splitlines(), start=1):
        marker = MARKER_RE.search(line)
        if marker:
            yield Violation(path, number, f"banned marker {marker.group(1)}")
        if NOT_IMPLEMENTED_RE.search(line):
            yield Violation(path, number, "banned NotImplementedError")


def _dotted_name(node: ast.expr) -> str:
    """Last name component of `Name`, `Attribute` or `Subscript` (`typing.Protocol[T]`)."""
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _is_protocol(node: ast.ClassDef) -> bool:
    return any(_dotted_name(base) == "Protocol" for base in node.bases)


def _is_overload(node: FunctionNode) -> bool:
    return any(_dotted_name(dec) == "overload" for dec in node.decorator_list)


def _body_without_docstring(node: FunctionNode) -> list[ast.stmt]:
    body = list(node.body)
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        body = body[1:]
    return body


def _stub_kind(node: FunctionNode) -> str | None:
    """Return "pass" or "..." if the function body is only that statement, else None."""
    body = _body_without_docstring(node)
    if len(body) != 1:
        return None
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return "pass"
    if (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    ):
        return "..."
    return None


def _body_line_count(node: FunctionNode) -> int:
    """Lines from the first body statement (docstring excluded) to the end of the function."""
    body = _body_without_docstring(node)
    if not body or node.end_lineno is None:
        return 0
    return node.end_lineno - body[0].lineno + 1


def _protocol_methods(tree: ast.AST) -> set[int]:
    """ids of functions defined directly in a Protocol class body."""
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and _is_protocol(node):
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    allowed.add(id(child))
    return allowed


def _check_function(path: Path, node: FunctionNode, in_protocol: bool) -> Iterator[Violation]:
    length = _body_line_count(node)
    if length > MAX_BODY_LINES:
        message = f"function {node.name!r} body is {length} lines (max {MAX_BODY_LINES})"
        yield Violation(path, node.lineno, message)
    kind = _stub_kind(node)
    if kind == "pass":
        yield Violation(path, node.lineno, f"function {node.name!r} has `pass` as its body")
    elif kind == "..." and not (in_protocol or _is_overload(node)):
        yield Violation(path, node.lineno, f"function {node.name!r} has `...` as its body")


def _caught_names(handler_type: ast.expr) -> Iterator[str]:
    if isinstance(handler_type, ast.Tuple):
        for element in handler_type.elts:
            yield from _caught_names(element)
    else:
        yield _dotted_name(handler_type)


def _check_handler(path: Path, node: ast.ExceptHandler) -> Iterator[Violation]:
    if node.type is None:
        yield Violation(path, node.lineno, "bare `except:`")
        return
    broad = BROAD_EXCEPTIONS.intersection(_caught_names(node.type))
    for name in sorted(broad):
        yield Violation(path, node.lineno, f"`except` catches {name}")


def _scan_python(path: Path, text: str) -> Iterator[Violation]:
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        yield Violation(path, exc.lineno or 1, f"file does not parse: {exc.msg}")
        return
    protocol_methods = _protocol_methods(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield from _check_function(path, node, id(node) in protocol_methods)
        elif isinstance(node, ast.ExceptHandler):
            yield from _check_handler(path, node)


def find_violations(path: Path) -> list[Violation]:
    """All banned patterns in one file, ordered by line."""
    text = path.read_text(encoding="utf-8")
    found = list(_scan_text(path, text))
    if path.suffix == ".py":
        found.extend(_scan_python(path, text))
    return sorted(found, key=lambda v: (v.line, v.message))


def scan(root: Path) -> list[Violation]:
    """Scan every .py and .md file under `root`; fails if `root` is not a directory."""
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in SCANNED_SUFFIXES)
    return [violation for file in files for violation in find_violations(file)]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: 0 if clean, 1 if violations were found."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("root", nargs="?", default="src", type=Path)
    args = parser.parse_args(argv)
    violations = scan(args.root)
    for violation in violations:
        print(violation.render())
    if violations:
        print(f"banned-pattern check FAILED: {len(violations)} violation(s)")
        return 1
    print(f"banned-pattern check passed ({args.root})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
