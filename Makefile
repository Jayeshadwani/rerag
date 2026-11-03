# Make targets from CLAUDE.md section 8. Targets whose implementation arrives in a later
# milestone fail loudly (exit 1) instead of pretending to work.

UV := uv run --locked

.PHONY: setup check lint types test banned sync eval eval-header eval-test results mutation

setup:
	uv sync --locked

check: lint types test banned

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

types:
	$(UV) mypy

test:
	$(UV) pytest --cov --cov-report=term-missing

banned:
	$(UV) python scripts/check_banned.py src

sync:
	$(UV) noterecall sync

MILESTONE ?= M5

eval:
	$(UV) noterecall eval --split dev

# M5: the header on/off row on dev (the M4 decisions are the base).
eval-header:
	$(UV) noterecall eval --split dev --suite header

# Runs the test split ONCE per milestone (the ledger in results/test_ledger.json refuses a repeat).
eval-test:
	$(UV) noterecall eval --split test --milestone $(MILESTONE)

results:
	$(UV) python scripts/render_results.py results/corpus_stats.json

mutation:
	@echo "make mutation: not available until M9 (mutmut on fusion, metrics, planner, citations)" >&2; exit 1
