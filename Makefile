.PHONY: check test test-all

check:
	uv run ruff check src tests tools
	uv run ruff format --check src tests tools
	uv run mypy src/unisupermimic
	uv run pyright src/unisupermimic

test:
	uv run pytest

test-all:
	uv run pytest -m ''
