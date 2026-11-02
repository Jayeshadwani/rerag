"""BM25 index: hand-computed scores, ranking, ties and edge cases."""

import math

import pytest

from noterecall.index.code_tokenizer import tokenize_code, tokenize_plain
from noterecall.index.lexical import LexicalError, LexicalIndex
from noterecall.models import Chunk


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid, notebook_path="n", heading_path=(), pieces=(), text=text, tokens=1, cell_hashes=()
    )


CHUNKS = [
    chunk("c0", "pandas groupby mean"),
    chunk("c1", "train test split split"),
    chunk("c2", "numpy array"),
    chunk("c3", "numpy array numpy"),
]


def test_scores_match_the_lucene_bm25_formula() -> None:
    index = LexicalIndex.build(CHUNKS, tokenize_plain)
    (hit,) = index.search("groupby", 5)
    # N = 4, df(groupby) = 1: idf = ln(1 + (4 - 1 + 0.5) / (1 + 0.5)) = ln(3.3333) = 1.20397
    # doc c0 has 3 tokens; lengths 3, 4, 2, 3 -> average 3; tf = 1, k1 = 1.5, b = 0.75:
    # tf / (tf + k1 * (1 - b + b * 3 / 3)) = 1 / (1 + 1.5) = 0.4; score = 1.20397 * 0.4 = 0.48159
    assert hit.chunk_id == "c0"
    assert hit.score == pytest.approx(math.log(1 + 3.5 / 1.5) * 0.4, abs=1e-5)


def test_ranking_prefers_higher_term_frequency_and_shorter_text() -> None:
    index = LexicalIndex.build(CHUNKS, tokenize_plain)
    hits = index.search("numpy", 5)
    assert [h.chunk_id for h in hits] == ["c3", "c2"]  # tf 2 beats tf 1 here
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].score > hits[1].score > 0


def test_only_matching_chunks_are_returned_and_k_limits_them() -> None:
    index = LexicalIndex.build(CHUNKS, tokenize_plain)
    assert index.search("zzz", 5) == []
    assert [h.chunk_id for h in index.search("numpy", 1)] == ["c3"]


def test_empty_query_tokens_return_nothing() -> None:
    index = LexicalIndex.build(CHUNKS, tokenize_plain)
    assert index.search("", 5) == []
    assert index.search("!!! ???", 5) == []


def test_equal_scores_keep_chunk_order() -> None:
    twins = [chunk("b", "alpha"), chunk("a", "alpha")]
    index = LexicalIndex.build(twins, tokenize_plain)
    assert [h.chunk_id for h in index.search("alpha", 5)] == ["b", "a"]  # corpus order, not id


def test_code_tokenizer_matches_across_identifier_styles() -> None:
    chunks = [chunk("snake", "use train_test_split here"), chunk("other", "nothing relevant")]
    plain = LexicalIndex.build(chunks, tokenize_plain)
    code = LexicalIndex.build(chunks, tokenize_code)
    assert plain.search("trainTestSplit", 5) == []  # plain: one unrelated word
    assert [h.chunk_id for h in code.search("trainTestSplit", 5)] == ["snake"]
    assert [h.chunk_id for h in code.search("df.train_test_split", 5)] == ["snake"]


def test_invalid_inputs_fail() -> None:
    with pytest.raises(LexicalError, match="zero chunks"):
        LexicalIndex.build([], tokenize_plain)
    with pytest.raises(LexicalError, match="k must be at least 1"):
        LexicalIndex.build(CHUNKS, tokenize_plain).search("x", 0)
