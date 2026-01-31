# Scholar Agent Backend - Makefile
# Convenient commands for development, testing, and deployment

.PHONY: help install dev-install lint format type-check test test-unit test-integration test-coverage test-performance clean

# Default target
help:
	@echo "Scholar Agent Backend - Available Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install          Install production dependencies"
	@echo "  make dev-install      Install development dependencies"
	@echo "  make dev              Start development server"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run Ruff linter"
	@echo "  make format           Format code with Black and Ruff"
	@echo "  make type-check       Run MyPy type checker"
	@echo "  make security         Run Bandit security scanner"
	@echo "  make check            Run all code quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-e2e         Run end-to-end tests"
	@echo "  make test-performance Run performance tests"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean            Clean build artifacts"
	@echo "  make clean-all        Clean everything including venv"

# ==========================================
# Development Setup
# ==========================================

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-timeout httpx
	pip install ruff black mypy bandit safety pip-audit

dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# ==========================================
# Code Quality
# ==========================================

lint:
	@echo "Running Ruff linter..."
	ruff check . --fix
	@echo "Linting complete!"

lint-check:
	@echo "Running Ruff linter (check only)..."
	ruff check .

format:
	@echo "Formatting with Black..."
	black .
	@echo "Sorting imports with Ruff..."
	ruff check . --select I --fix
	@echo "Formatting complete!"

format-check:
	@echo "Checking format with Black..."
	black --check .

type-check:
	@echo "Running MyPy type checker..."
	mypy . --ignore-missing-imports
	@echo "Type checking complete!"

security:
	@echo "Running Bandit security scanner..."
	bandit -r . -ll -ii -x tests
	@echo "Security scan complete!"

check: lint-check format-check type-check
	@echo "All checks passed!"

# ==========================================
# Testing
# ==========================================

test:
	@echo "Running all tests..."
	pytest tests/ -v --tb=short

test-unit:
	@echo "Running unit tests..."
	pytest tests/ -v -m "not integration and not slow and not e2e and not performance" --tb=short

test-integration:
	@echo "Running integration tests..."
	pytest tests/ -v -m "integration" --tb=short

test-e2e:
	@echo "Running end-to-end tests..."
	pytest tests/test_e2e.py tests/test_integration.py -v --tb=short

test-performance:
	@echo "Running performance tests..."
	pytest tests/test_performance.py -v -m "slow or performance" --tb=short

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v \
		--cov=. \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=70 \
		-m "not slow and not performance"
	@echo ""
	@echo "Coverage report generated in htmlcov/"

test-quick:
	@echo "Running quick smoke tests..."
	pytest tests/ -v -x --tb=line -q -m "not slow and not integration and not e2e and not performance"

test-parallel:
	@echo "Running tests in parallel..."
	pytest tests/ -v -n auto --tb=short -m "not slow"

# ==========================================
# Maintenance
# ==========================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "Clean complete!"

clean-all: clean
	rm -rf .venv venv
	@echo "Full clean complete!"

# ==========================================
# Docker
# ==========================================

docker-build:
	docker build -t scholar-agent-backend .

docker-run:
	docker run -p 8000:8000 --env-file .env scholar-agent-backend

# ==========================================
# Database
# ==========================================

db-migrate:
	@echo "Running database migrations..."
	alembic upgrade head

db-reset:
	@echo "Resetting database..."
	alembic downgrade base
	alembic upgrade head
