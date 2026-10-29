"""Check the bootstrap on synthetic data with known effects (results/bootstrap_validation.json).

Settings: 60 paired deltas (the ADR-009 minimum number of queries), sd 0.1, true effects 0,
0.02 and 0.05, 1,000 synthetic experiments each, 1,000 resamples per bootstrap. Only synthetic
numbers are involved. Usage: `python scripts/validate_bootstrap.py [output.json]`.
"""

import sys
from pathlib import Path

from noterecall.eval.stats import SimulationSpec, validate_bootstrap
from noterecall.ingest.stats import git_state

SEED = 20261029
EFFECTS = (0.0, 0.02, 0.05)
TRIALS = 1000
RESAMPLES = 1000


def main(argv: list[str] | None = None) -> int:
    """Run the simulations and write the JSON file."""
    args = sys.argv[1:] if argv is None else argv
    target = Path(args[0]) if args else Path("results") / "bootstrap_validation.json"
    specs = [
        SimulationSpec(effect=effect, sd=0.1, n=60, trials=TRIALS, resamples=RESAMPLES)
        for effect in EFFECTS
    ]
    report = validate_bootstrap(specs, SEED, git_state())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
