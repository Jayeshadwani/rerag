"""Tokenizer wrapper, tested with an offline byte-level encoding (no vocabulary download)."""

import pytest

from noterecall.chunking.tokens import Tokenizer, TokenizerError, load_tokenizer


def test_counts_utf8_bytes_with_the_tiny_encoding(tiny_tokenizer: Tokenizer) -> None:
    assert tiny_tokenizer.count("") == 0
    assert tiny_tokenizer.count("abc") == 3
    assert tiny_tokenizer.count("é") == 2  # two UTF-8 bytes
    assert tiny_tokenizer.count("a b\n") == 4


def test_special_token_text_counts_as_plain_text(tiny_tokenizer: Tokenizer) -> None:
    # "<|endoftext|>" is 13 characters; as ordinary text it is 13 tokens, and does not raise.
    assert tiny_tokenizer.count("<|endoftext|>") == 13


def test_name_is_the_encoding_name(tiny_tokenizer: Tokenizer) -> None:
    assert tiny_tokenizer.name == "tiny-bytes"


def test_unknown_encoding_name_raises() -> None:
    with pytest.raises(TokenizerError, match="no-such-encoding"):
        load_tokenizer("no-such-encoding")


@pytest.mark.live
def test_cl100k_base_loads_and_counts() -> None:
    tokenizer = load_tokenizer("cl100k_base")
    assert tokenizer.name == "cl100k_base"
    # BPE merges characters, so the token count is below the character count.
    assert 0 < tokenizer.count("hello world") < len("hello world")


def test_download_failure_becomes_a_tokenizer_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(name: str) -> object:
        raise ConnectionError("proxy said no")

    monkeypatch.setattr("noterecall.chunking.tokens.tiktoken.get_encoding", offline)
    with pytest.raises(TokenizerError, match="proxy said no"):
        load_tokenizer("cl100k_base")
