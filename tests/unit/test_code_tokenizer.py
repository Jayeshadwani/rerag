"""The code-aware tokenizer against hand-written expectations."""

import pytest

from noterecall.index.code_tokenizer import split_words, tokenize_code, tokenize_plain


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("df.groupby", ["df.groupby", "df", "groupby"]),
        ("train_test_split", ["train_test_split", "train", "test", "split"]),
        ("trainTestSplit", ["traintestsplit", "train", "test", "split"]),
        ("TrainTestSplit", ["traintestsplit", "train", "test", "split"]),
        ("HTTPServer", ["httpserver", "http", "server"]),
        ("np.linalg.norm", ["np.linalg.norm", "np", "linalg", "norm"]),
        ("sklearn.model_selection.train_test_split", [
            "sklearn.model_selection.train_test_split",
            "sklearn", "model_selection", "model", "selection",
            "train_test_split", "train", "test", "split",
        ]),
        ("__init__", ["__init__", "init"]),
        ("_private", ["_private", "private"]),
        ("groupby", ["groupby"]),
        ("layer1", ["layer1"]),  # digits do not split a word
        ("Mean", ["mean"]),
    ],
)  # fmt: skip
def test_identifiers(text: str, expected: list[str]) -> None:
    assert tokenize_code(text) == expected


def test_prose_and_code_mix_with_punctuation_dropped() -> None:
    text = "Use `cross_val_score` (k-fold) on X_train, then df.head()."
    assert tokenize_code(text) == [
        "use", "cross_val_score", "cross", "val", "score", "k", "fold", "on",
        "x_train", "x", "train", "then", "df.head", "df", "head",
    ]  # fmt: skip


def test_numbers_are_kept_whole() -> None:
    assert tokenize_code("lr=0.01, n=10") == ["lr", "0.01", "n", "10"]
    assert tokenize_code("3.14.15") == ["3.14", "15"]  # a number is digits with one fraction


def test_non_ascii_letters_are_identifier_characters() -> None:
    assert tokenize_code("café_ñandú") == ["café_ñandú", "café", "ñandú"]


def test_camel_and_snake_spellings_share_their_parts() -> None:
    assert set(tokenize_code("trainTestSplit")) & set(tokenize_code("train_test_split")) == {
        "train",
        "test",
        "split",
    }


def test_empty_and_symbol_only_text_give_no_tokens() -> None:
    assert tokenize_code("") == []
    assert tokenize_code("()[]{}+-*/ ,;:") == []


def test_plain_tokenizer_is_lowercase_word_runs() -> None:
    assert tokenize_plain("df.groupby('K')['v'].mean()") == ["df", "groupby", "k", "v", "mean"]
    assert tokenize_plain("train_test_split") == ["train_test_split"]
    assert tokenize_plain("") == []


def test_split_words() -> None:
    assert split_words("a_b__cD") == ["a", "b", "c", "d"]
    assert split_words("") == []
    assert split_words("___") == []
