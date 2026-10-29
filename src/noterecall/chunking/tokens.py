"""The single tokenizer used everywhere (PLAN section 5). Supports ADR-004 (token caps).

`Tokenizer` wraps a `tiktoken.Encoding`. Special-token strings that appear in notebooks
(for example `<|endoftext|>`) are counted as ordinary text instead of raising.
"""

import tiktoken


class TokenizerError(Exception):
    """Raised when an encoding cannot be loaded (unknown name or failed vocabulary download)."""


class Tokenizer:
    """Counts tokens with one fixed encoding."""

    def __init__(self, encoding: tiktoken.Encoding) -> None:
        self._encoding = encoding

    @property
    def name(self) -> str:
        """Encoding name, recorded in results files."""
        return self._encoding.name

    def count(self, text: str) -> int:
        """Number of tokens in `text`; special-token lookalikes count as plain text."""
        return len(self._encoding.encode(text, disallowed_special=()))

    def token_starts(self, text: str) -> list[int]:
        """Character offset at which each token of `text` starts.

        A token that begins inside a multi-byte character reports that character's start, so
        slicing `text` at these offsets never cuts a character in two.
        """
        tokens = self._encoding.encode(text, disallowed_special=())
        decoded, offsets = self._encoding.decode_with_offsets(tokens)
        if decoded != text:
            raise TokenizerError("decoding the tokens did not reproduce the text")
        return offsets


def load_tokenizer(encoding_name: str) -> Tokenizer:
    """Load a named tiktoken encoding. Downloads its vocabulary once, then uses the cache."""
    try:
        return Tokenizer(tiktoken.get_encoding(encoding_name))
    except (ValueError, OSError) as exc:
        raise TokenizerError(
            f"cannot load tokenizer {encoding_name!r}: {exc}. "
            "A known encoding needs network access once to download its vocabulary."
        ) from exc
