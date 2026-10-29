"""Evidence-output rule: each verdict, its boundaries, and the order the checks run in.

The offline tokenizer counts one token per byte, so the 64-token limit is 64 bytes here."""

import pytest

from noterecall.chunking.evidence import (
    MAX_EVIDENCE_TOKENS,
    EvidenceVerdict,
    is_evidence_output,
    judge_output,
    rule_fingerprint,
)
from noterecall.chunking.tokens import Tokenizer

V = EvidenceVerdict


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("42", V.EVIDENCE),
        ("Accuracy: 0.93", V.EVIDENCE),
        ("The mean of the sample is 3.5", V.EVIDENCE),
        ("True", V.EVIDENCE),
        ("mean    3.5", V.EVIDENCE),  # one aligned row is not a table
        ("[1 2 3 4 5 6 7 8]", V.EVIDENCE),  # exactly 8 numbers is allowed
        ("", V.EMPTY),
        ("  \n\n", V.EMPTY),
        ("first\n[... 12 more lines truncated]", V.TRUNCATED),
        ("50%|████| 5/10 [00:01<00:00, 5it/s]", V.PROGRESS),  # under 64 bytes
        ("Epoch 3/10 [=====>....] ETA: 4s", V.PROGRESS),
        ("12.5 it/s", V.PROGRESS),
        ("UserWarning: the thing is odd", V.WARNING),
        ("FutureWarning: x", V.WARNING),
        ("this function is deprecated", V.WARNING),
        ("<matplotlib.axes._subplots.AxesSubplot at 0x7f3a2c1d>", V.OBJECT_REPR),
        ("<__main__.Foo object at 0x7f3a2c1d>", V.OBJECT_REPR),
        ("<Figure size 640x480 with 1 Axes>", V.OBJECT_REPR),
        ("array([1, 2, 3])", V.ARRAY),
        ("[1 2 3 4 5 6 7 8 9]", V.ARRAY),  # 9 numbers on one line
        ("   a  b\n0  1  2\n1  3  4", V.TABLE),
        ("x    y\nz    w", V.TABLE),
    ],
)
def test_verdicts(tiny_tokenizer: Tokenizer, text: str, verdict: EvidenceVerdict) -> None:
    assert judge_output(text, tiny_tokenizer) is verdict
    assert is_evidence_output(text, tiny_tokenizer) is (verdict is V.EVIDENCE)


def test_the_length_limit_is_inclusive(tiny_tokenizer: Tokenizer) -> None:
    assert MAX_EVIDENCE_TOKENS == 64
    assert judge_output("x" * 64, tiny_tokenizer) is V.EVIDENCE
    assert judge_output("x" * 65, tiny_tokenizer) is V.TOO_LONG


def test_checks_run_in_a_fixed_order(tiny_tokenizer: Tokenizer) -> None:
    long_warning = "UserWarning: " + "x" * 80
    assert judge_output(long_warning, tiny_tokenizer) is V.TOO_LONG  # length before content
    truncated_long = "x" * 80 + "\n[... 3 more lines truncated]"
    assert judge_output(truncated_long, tiny_tokenizer) is V.TRUNCATED  # truncation before length
    warning_in_progress = "100%|███| 1/1 [00:00<00:00] UserWarning"
    assert (
        judge_output(warning_in_progress, tiny_tokenizer) is V.PROGRESS
    )  # progress before warning


def test_numbers_on_different_lines_do_not_add_up(tiny_tokenizer: Tokenizer) -> None:
    text = "1 2 3 4 5\n6 7 8 9 10"
    assert judge_output(text, tiny_tokenizer) is V.EVIDENCE  # at most 8 per line


def test_fingerprint_is_a_stable_sha256() -> None:
    assert rule_fingerprint() == rule_fingerprint()
    assert len(rule_fingerprint()) == 64
