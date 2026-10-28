"""docs/adr/ADR-NNN.md are verbatim copies of the locked ADRs in PLAN.md section 2."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _plan_adrs() -> dict[str, str]:
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    start = plan.index("## 2. Architecture Decision Records (LOCKED)")
    section = plan[start : plan.index("\n---\n", start)]
    parts = re.split(r"(?m)^(?=### ADR-)", section)[1:]
    return {part[4:11]: part.rstrip("\n") + "\n" for part in parts}


def test_plan_has_fifteen_adrs() -> None:
    assert sorted(_plan_adrs()) == [f"ADR-{n:03d}" for n in range(1, 16)]


def test_adr_files_match_plan_verbatim() -> None:
    for adr_id, text in _plan_adrs().items():
        copy = (ROOT / "docs" / "adr" / f"{adr_id}.md").read_text(encoding="utf-8")
        assert copy == text, adr_id


def test_no_extra_adr_files() -> None:
    on_disk = sorted(p.stem for p in (ROOT / "docs" / "adr").glob("*.md"))
    assert on_disk == sorted(_plan_adrs())
