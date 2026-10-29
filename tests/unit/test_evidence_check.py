"""Evidence-rule validation: sampling, label checking, and the precision/recall arithmetic."""

import json
from pathlib import Path

import pytest

from noterecall.chunking.evidence import EvidenceVerdict
from noterecall.chunking.evidence_check import (
    EvidenceCheckError,
    OutputRef,
    SampleItem,
    SampleManifest,
    draw_sample,
    evaluate,
    read_labels,
    read_manifest,
    render_sample,
    score_items,
    wilson_interval,
    write_manifest,
)


def ref(i: int, keeps: bool) -> OutputRef:
    verdict = EvidenceVerdict.EVIDENCE if keeps else EvidenceVerdict.TOO_LONG
    return OutputRef(notebook_path=f"nb{i // 10}.ipynb", cell_index=i, verdict=verdict)


REFS = [ref(i, keeps=i < 6) for i in range(16)]  # 6 kept, 10 dropped


def test_wilson_matches_the_closed_form() -> None:
    # p = 0.8, n = 25, z = 1.96: centre = (0.8 + 1.96^2/50) / (1 + 1.96^2/25) = 0.7601,
    # half-width = 1.96 * sqrt(0.0064 + 1.96^2/2500) / 1.153664 = 0.1514
    interval = wilson_interval(20, 25)
    assert interval.low == pytest.approx(0.6087, abs=1e-3)
    assert interval.high == pytest.approx(0.9115, abs=1e-3)
    # all failures: high = z^2 / (n + z^2) = 3.8416 / 13.8416; all successes: low = n / (n + z^2)
    assert wilson_interval(0, 10).low == 0.0
    assert wilson_interval(0, 10).high == pytest.approx(3.8416 / 13.8416, abs=1e-6)
    assert wilson_interval(10, 10).low == pytest.approx(10 / 13.8416, abs=1e-6)
    assert wilson_interval(10, 10).high == 1.0
    with pytest.raises(ValueError, match="positive"):
        wilson_interval(0, 0)


def test_sample_has_the_requested_strata_and_ids() -> None:
    manifest = draw_sample(REFS, per_stratum=3, seed=1, fingerprint="fp")
    assert [i.sample_id for i in manifest.items] == [1, 2, 3, 4, 5, 6]
    assert sum(i.rule_keeps for i in manifest.items) == 3
    assert (manifest.population_kept, manifest.population_dropped) == (6, 10)
    keys = {(i.notebook_path, i.cell_index) for i in manifest.items}
    assert len(keys) == 6  # no output sampled twice
    assert manifest.rule_fingerprint == "fp"


def test_sampling_is_reproducible_and_seed_dependent() -> None:
    a = draw_sample(REFS, 3, seed=1, fingerprint="fp")
    assert a == draw_sample(
        list(reversed(REFS)), 3, seed=1, fingerprint="fp"
    )  # input order is irrelevant
    assert a != draw_sample(REFS, 3, seed=2, fingerprint="fp")


def test_small_strata_are_taken_whole() -> None:
    manifest = draw_sample(REFS, per_stratum=50, seed=1, fingerprint="fp")
    assert len(manifest.items) == 16


def test_the_sample_does_not_reveal_the_stratum_in_its_order() -> None:
    orders = {
        tuple(i.rule_keeps for i in draw_sample(REFS, 3, seed=s, fingerprint="fp").items)
        for s in range(20)
    }
    assert len(orders) > 1  # shuffled, so the labeller cannot infer the verdict from position


def test_manifest_round_trips_through_a_file(tmp_path: Path) -> None:
    manifest = draw_sample(REFS, 3, seed=1, fingerprint="fp")
    write_manifest(manifest, tmp_path / "deep" / "sample.json")
    assert read_manifest(tmp_path / "deep" / "sample.json") == manifest


def test_unreadable_manifest_is_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(EvidenceCheckError, match="cannot read the sample"):
        read_manifest(tmp_path / "missing.json")
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EvidenceCheckError, match="cannot read the sample"):
        read_manifest(tmp_path / "bad.json")


def _manifest(kept: int, dropped: int, pop_kept: int, pop_dropped: int) -> SampleManifest:
    items = [
        SampleItem(sample_id=i + 1, notebook_path="n", cell_index=i, rule_keeps=i < kept)
        for i in range(kept + dropped)
    ]
    return SampleManifest(
        seed=1,
        rule_fingerprint="fp",
        population_kept=pop_kept,
        population_dropped=pop_dropped,
        items=tuple(items),
    )


def test_precision_and_weighted_recall_hand_computed() -> None:
    # population: 100 kept, 300 dropped. sample: 25 of each.
    # labelled evidence: 20 of the 25 kept, 5 of the 25 dropped.
    manifest = _manifest(25, 25, 100, 300)
    labels = {i.sample_id: (i.sample_id <= 20) or (26 <= i.sample_id <= 30) for i in manifest.items}
    result = evaluate(manifest, labels)
    assert result.precision == pytest.approx(20 / 25)
    # estimated evidence kept = 20 * 100/25 = 80; missed = 5 * 300/25 = 60; recall = 80 / 140
    assert result.recall_weighted == pytest.approx(80 / 140)
    assert result.evidence_prevalence_weighted == pytest.approx(140 / 400)
    assert result.precision_wilson95 is not None
    assert result.precision_wilson95.low == pytest.approx(0.6087, abs=1e-3)
    assert (result.sampled_kept, result.sampled_dropped) == (25, 25)
    assert (result.labelled_evidence_in_kept, result.labelled_evidence_in_dropped) == (20, 5)


def test_a_perfect_rule() -> None:
    manifest = _manifest(2, 2, 2, 2)
    result = evaluate(manifest, {1: True, 2: True, 3: False, 4: False})
    assert (result.precision, result.recall_weighted) == (1.0, 1.0)


def test_degenerate_cases_do_not_divide_by_zero() -> None:
    no_kept = evaluate(_manifest(0, 2, 0, 5), {1: True, 2: False})
    assert no_kept.precision is None
    assert no_kept.precision_wilson95 is None
    assert no_kept.recall_weighted == 0.0  # evidence exists (1 of 2 dropped) but none is kept
    no_evidence = evaluate(_manifest(1, 1, 3, 3), {1: False, 2: False})
    assert no_evidence.precision == 0.0
    assert no_evidence.recall_weighted is None  # nothing to recall


def _labels_file(tmp_path: Path, labels: dict[str, str] | None, raw: str | None = None) -> Path:
    path = tmp_path / "labels.json"
    path.write_text(raw if raw is not None else json.dumps({"labels": labels}), encoding="utf-8")
    return path


def test_labels_must_cover_exactly_the_sample(tmp_path: Path) -> None:
    manifest = _manifest(1, 1, 1, 1)
    assert read_labels(_labels_file(tmp_path, {"1": "E", "2": "N"}), manifest) == {
        1: True,
        2: False,
    }
    with pytest.raises(EvidenceCheckError, match=r"missing \[2\]"):
        read_labels(_labels_file(tmp_path, {"1": "E"}), manifest)
    with pytest.raises(EvidenceCheckError, match=r"unexpected \[3\]"):
        read_labels(_labels_file(tmp_path, {"1": "E", "2": "N", "3": "N"}), manifest)


@pytest.mark.parametrize("raw", ['{"labels": {"1": "X", "2": "N"}}', "not json", '{"nope": 1}'])
def test_malformed_labels_are_rejected(tmp_path: Path, raw: str) -> None:
    with pytest.raises(EvidenceCheckError, match="cannot read the labels"):
        read_labels(_labels_file(tmp_path, None, raw), _manifest(1, 1, 1, 1))
    with pytest.raises(EvidenceCheckError, match="cannot read the labels"):
        read_labels(tmp_path / "missing.json", _manifest(1, 1, 1, 1))


def test_render_sample_shows_ids_and_sizes_but_not_paths_or_verdicts() -> None:
    manifest = SampleManifest(
        seed=1,
        rule_fingerprint="fp",
        population_kept=1,
        population_dropped=1,
        items=(
            SampleItem(
                sample_id=1, notebook_path="secret/path.ipynb", cell_index=4, rule_keeps=True
            ),
            SampleItem(
                sample_id=2, notebook_path="secret/path.ipynb", cell_index=9, rule_keeps=False
            ),
        ),
    )
    texts = {("secret/path.ipynb", 4): "line one\nline two\n", ("secret/path.ipynb", 9): "x" * 700}
    text = render_sample(manifest, texts)
    assert "#1  (2 lines, 17 characters)\nline one\nline two" in text
    assert "#2  (1 lines, 700 characters)\n" + "x" * 600 + "\n[... 100 more characters]" in text
    assert "secret" not in text
    assert "True" not in text
    assert "False" not in text


def test_score_items_keeps_sample_order_and_never_carries_text() -> None:
    manifest = SampleManifest(
        seed=1,
        rule_fingerprint="f" * 64,
        population_kept=2,
        population_dropped=2,
        items=(
            SampleItem(sample_id=1, notebook_path="a.ipynb", cell_index=3, rule_keeps=False),
            SampleItem(sample_id=2, notebook_path="b.ipynb", cell_index=0, rule_keeps=True),
        ),
    )
    verdicts = {1: EvidenceVerdict.TABLE, 2: EvidenceVerdict.EVIDENCE}
    scores = score_items(manifest, {1: True, 2: False}, verdicts)
    assert [(s.sample_id, s.stratum, s.rule_verdict, s.labelled_evidence) for s in scores] == [
        (1, "dropped", EvidenceVerdict.TABLE, True),
        (2, "kept", EvidenceVerdict.EVIDENCE, False),
    ]
    assert "ipynb" not in scores[0].model_dump_json()
    with pytest.raises(EvidenceCheckError, match="no rule verdict for sample ids \\[2\\]"):
        score_items(manifest, {1: True, 2: False}, {1: EvidenceVerdict.TABLE})
