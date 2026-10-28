"""Every banned pattern must be caught, and allowed look-alikes must not be."""

import textwrap
from pathlib import Path

import pytest

from check_banned import find_violations, main, scan

REPO_SRC = Path(__file__).resolve().parents[2] / "src"


def _violations(tmp_path: Path, source: str, name: str = "mod.py") -> list[tuple[int, str]]:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return [(v.line, v.message) for v in find_violations(path)]


@pytest.mark.parametrize("word", ["TODO", "FIXME", "XXX"])
def test_marker_words_are_caught_in_comments(tmp_path: Path, word: str) -> None:
    found = _violations(tmp_path, f"x = 1\n# {word}: later\n")
    assert found == [(2, f"banned marker {word}")]


def test_marker_word_is_caught_in_docstring_and_markdown(tmp_path: Path) -> None:
    assert _violations(tmp_path, '"""TODO"""\n') == [(1, "banned marker TODO")]
    assert _violations(tmp_path, "intro\nFIXME here\n", "prompt.md") == [(2, "banned marker FIXME")]


@pytest.mark.parametrize("text", ["TODOS = 1", "todo = 1", "XXXL = 2", "MYTODO = 3"])
def test_marker_look_alikes_are_not_caught(tmp_path: Path, text: str) -> None:
    assert _violations(tmp_path, text + "\n") == []


def test_not_implemented_error_is_caught(tmp_path: Path) -> None:
    source = """
    def f() -> None:
        raise NotImplementedError
    """
    assert _violations(tmp_path, source) == [(3, "banned NotImplementedError")]


def test_pass_body_is_caught(tmp_path: Path) -> None:
    source = """
    def f() -> None:
        pass
    """
    assert _violations(tmp_path, source) == [(2, "function 'f' has `pass` as its body")]


def test_pass_after_docstring_is_still_a_stub(tmp_path: Path) -> None:
    source = '''
    def f() -> None:
        """Doc."""
        pass
    '''
    assert _violations(tmp_path, source) == [(2, "function 'f' has `pass` as its body")]


def test_ellipsis_body_is_caught_in_plain_function_and_async(tmp_path: Path) -> None:
    source = """
    def f() -> None: ...

    async def g() -> None:
        ...
    """
    assert _violations(tmp_path, source) == [
        (2, "function 'f' has `...` as its body"),
        (4, "function 'g' has `...` as its body"),
    ]


def test_ellipsis_body_is_caught_in_ordinary_class_method(tmp_path: Path) -> None:
    source = '''
    class Thing:
        def method(self) -> None:
            """Doc."""
            ...
    '''
    assert _violations(tmp_path, source) == [(3, "function 'method' has `...` as its body")]


def test_ellipsis_allowed_in_protocol_methods(tmp_path: Path) -> None:
    source = '''
    import typing
    from typing import Protocol

    class A(Protocol):
        def run(self) -> None: ...

    class B[T](typing.Protocol[T]):
        def get(self) -> T:
            """Doc."""
            ...
    '''
    assert _violations(tmp_path, source) == []


def test_pass_not_allowed_even_in_protocol(tmp_path: Path) -> None:
    source = """
    from typing import Protocol

    class A(Protocol):
        def run(self) -> None:
            pass
    """
    assert _violations(tmp_path, source) == [(5, "function 'run' has `pass` as its body")]


def test_ellipsis_in_function_nested_in_protocol_method_is_caught(tmp_path: Path) -> None:
    source = """
    from typing import Protocol

    class A(Protocol):
        def run(self) -> None:
            def inner() -> None: ...
    """
    assert _violations(tmp_path, source) == [(6, "function 'inner' has `...` as its body")]


def test_ellipsis_allowed_for_overload(tmp_path: Path) -> None:
    source = """
    import typing
    from typing import overload

    @overload
    def f(x: int) -> int: ...
    @typing.overload
    def f(x: str) -> str: ...
    def f(x: int | str) -> int | str:
        return x
    """
    assert _violations(tmp_path, source) == []


def test_legitimate_uses_of_pass_and_ellipsis_are_not_flagged(tmp_path: Path) -> None:
    source = """
    class MyError(Exception):
        pass

    def f(flag: bool) -> tuple[int, ...]:
        if flag:
            pass
        return (1, 2)

    def g() -> object:
        return ...
    """
    assert _violations(tmp_path, source) == []


@pytest.mark.parametrize(
    ("clause", "message"),
    [
        ("except:", "bare `except:`"),
        ("except Exception:", "`except` catches Exception"),
        ("except Exception as exc:", "`except` catches Exception"),
        ("except BaseException:", "`except` catches BaseException"),
        ("except (ValueError, Exception):", "`except` catches Exception"),
        ("except builtins.Exception:", "`except` catches Exception"),
        ("except (KeyError, (OSError, BaseException)):", "`except` catches BaseException"),
    ],
)
def test_broad_except_clauses_are_caught(tmp_path: Path, clause: str, message: str) -> None:
    source = f"import builtins\ntry:\n    x = 1\n{clause}\n    raise\n"
    assert _violations(tmp_path, source) == [(4, message)]


def test_specific_except_clauses_are_allowed(tmp_path: Path) -> None:
    source = """
    try:
        x = int("1")
    except (ValueError, OSError) as exc:
        raise RuntimeError("bad") from exc
    """
    assert _violations(tmp_path, source) == []


def test_unparseable_file_is_reported(tmp_path: Path) -> None:
    found = _violations(tmp_path, "def broken(:\n")
    assert len(found) == 1
    assert found[0][0] == 1
    assert found[0][1].startswith("file does not parse")


def test_scan_walks_directories_and_ignores_other_suffixes(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("# TODO\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.md").write_text("FIXME\n", encoding="utf-8")
    (tmp_path / "pkg" / "c.txt").write_text("TODO\n", encoding="utf-8")
    found = scan(tmp_path)
    assert [(v.path.name, v.line) for v in found] == [("a.py", 1), ("b.md", 1)]


def test_scan_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        scan(tmp_path / "missing")


def test_main_exit_codes_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "ok.py").write_text('"""Fine."""\n', encoding="utf-8")
    assert main([str(tmp_path)]) == 0
    assert "passed" in capsys.readouterr().out

    (tmp_path / "bad.py").write_text("# TODO\n", encoding="utf-8")
    assert main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert f"{tmp_path / 'bad.py'}:1: banned marker TODO" in out
    assert "FAILED: 1 violation(s)" in out


def test_real_src_tree_is_clean() -> None:
    assert scan(REPO_SRC) == []
