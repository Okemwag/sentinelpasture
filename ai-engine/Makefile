.PHONY: help install install-dev test lint format type-check clean docs

help:
	@echo "AI Engine - Development Commands"
	@echo ""
	@echo "install          Install package and dependencies"
	@echo "install-dev      Install package with development dependencies"
	@echo "test             Run tests with coverage"
	@echo "test-fast        Run tests without coverage"
	@echo "lint             Run linting checks"
	@echo "format           Format code with black and isort"
	@echo "type-check       Run type checking with mypy"
	@echo "clean            Clean build artifacts and cache"
	@echo "docs             Build documentation"
	@echo "all              Run format, lint, type-check, and test"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"

test:
	pytest tests/ -v --cov=ai_engine --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v

lint:
	flake8 ai_engine tests --max-line-length=100 --ignore=E203,W503
	black --check ai_engine tests
	isort --check-only ai_engine tests

format:
	black ai_engine tests
	isort ai_engine tests

type-check:
	mypy ai_engine --ignore-missing-imports

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

all: format lint type-check test
