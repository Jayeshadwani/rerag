"""Code-aware tokenizer for BM25. Implements ADR-005 (and the plain baseline it is compared with).

Notes mix prose and code, so identifiers must match however they are written. Every dotted
identifier is emitted whole and as its components (`df.groupby` gives `df.groupby`, `df`,
`groupby`), and every component that is made of several words (snake_case, camelCase,
PascalCase, acronyms) is emitted whole and as its parts (`train_test_split` gives
`train_test_split`, `train`, `test`, `split`; `trainTestSplit` gives `traintestsplit`, `train`,
`test`, `split`). Everything is lowercased. Numbers are kept as tokens. Digits do not split a
word (`layer1` stays whole). There are no stop words and no stemming: BM25's IDF handles
common words. The plain tokenizer is just lowercase word characters, the ablation baseline.
"""

import re

_IDENT = r"[^\W\d]\w*"
_ATOM = re.compile(rf"{_IDENT}(?:\.{_IDENT})*|\d+(?:\.\d+)?")
_WORD = re.compile(r"\w+")
# a word boundary inside an identifier: lower->Upper (trainTest), ACRONYM->Word (HTTPServer)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize_plain(text: str) -> list[str]:
    """Lowercased runs of word characters (the baseline in the ablation)."""
    return _WORD.findall(text.lower())


def split_words(component: str) -> list[str]:
    """Lowercased words of one identifier component: split on underscores and camel humps."""
    words: list[str] = []
    for chunk in component.split("_"):
        if chunk:
            words.extend(part.lower() for part in _CAMEL.split(chunk) if part)
    return words


def _component_tokens(component: str) -> list[str]:
    whole = component.lower()
    parts = split_words(component)
    if len(parts) > 1:
        return [whole, *parts]
    if parts and parts[0] != whole:  # e.g. `_private` or `__init__`: the stripped word too
        return [whole, parts[0]]
    return [whole]


def tokenize_code(text: str) -> list[str]:
    """Code-aware tokens of notebook text (see the module docstring)."""
    tokens: list[str] = []
    for atom in _ATOM.findall(text):
        if atom[0].isdigit():
            tokens.append(atom)
            continue
        components = atom.split(".")
        if len(components) > 1:
            tokens.append(atom.lower())
        for component in components:
            tokens.extend(_component_tokens(component))
    return tokens
