# Make targets from CLAUDE.md section 8. Targets whose implementation arrives in a later
# milestone fail loudly (exit 1) instead of pretending to work.

UV := uv run --locked

.PHONY: setup check lint types test banned sync eval eval-test results mutation

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
	@echo "make sync: not available until M1 (ingest + sync)" >&2; exit 1

eval:
	@echo "make eval: not available until M4 (retrieval ablation runner)" >&2; exit 1

eval-test:
	@echo "make eval-test: not available until M4 (retrieval ablation runner)" >&2; exit 1

results:
	@echo "make results: not available until M4 (scripts/render_results.py)" >&2; exit 1

mutation:
	@echo "make mutation: not available until M9 (mutmut on fusion, metrics, planner, citations)" >&2; exit 1
