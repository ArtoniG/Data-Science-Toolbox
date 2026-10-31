.PHONY: help install lint format type imports test coverage coverage-gate security check clean

help:
	@echo "install        install package with dev extras (editable)"
	@echo "lint           ruff check"
	@echo "format         ruff format (writes changes)"
	@echo "type           mypy src"
	@echo "imports        import-linter contract check"
	@echo "test           pytest"
	@echo "coverage       pytest with coverage report"
	@echo "coverage-gate  pytest with --cov-fail-under=80 (enable after foundation stabilizes)"
	@echo "security       bandit scan over src"
	@echo "check          lint + type + imports + test + security (what CI runs)"

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type:
	mypy src

imports:
	lint-imports

test:
	pytest

coverage:
	pytest --cov=ds_toolbox --cov-report=term-missing --cov-report=html

coverage-gate:
	pytest --cov=ds_toolbox --cov-fail-under=80

security:
	bandit -q -r src/

check: lint type imports test security

clean:
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage