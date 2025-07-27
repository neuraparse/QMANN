# QMANN Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev test lint format clean build docker paper benchmark reproduce

# Default target
help:
	@echo "QMANN Development Commands"
	@echo "========================"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install package for production"
	@echo "  install-dev  Install package for development"
	@echo "  setup        Complete development setup"
	@echo ""
	@echo "Development:"
	@echo "  test         Run all tests"
	@echo "  test-fast    Run fast tests only"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run all linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo ""
	@echo "Build:"
	@echo "  build        Build Python package"
	@echo "  docker       Build Docker images"
	@echo "  paper        Build research paper"
	@echo ""
	@echo "Benchmarks:"
	@echo "  benchmark    Run performance benchmarks"
	@echo "  benchmark-quick  Run quick benchmarks"
	@echo "  plot-bench   Generate benchmark plots"
	@echo ""
	@echo "Reproducibility:"
	@echo "  reproduce    Run full reproduction pipeline"
	@echo "  reproduce-quick  Run quick reproduction"
	@echo "  validate     Validate results against published values"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean        Clean build artifacts"
	@echo "  clean-all    Clean all generated files"
	@echo "  quicktest    Quick development test"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

setup: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Testing targets
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-cov:
	pytest tests/ -v --cov=src/qmann --cov-report=html --cov-report=term-missing

test-integration:
	pytest tests/ -v -m "integration"

# Code quality targets
lint: format type-check
	flake8 src/ tests/
	@echo "All linting checks passed!"

format:
	black src/ tests/ scripts/ benchmarks/
	isort src/ tests/ scripts/ benchmarks/

type-check:
	mypy src/qmann/

# Build targets
build:
	python -m build

docker:
	docker build -f docker/Dockerfile -t qmann:latest .
	docker build -f docker/Dockerfile --target development -t qmann:dev .
	docker build -f docker/Dockerfile --target benchmark -t qmann:benchmark .

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Paper targets
paper:
	cd paper && make pdf

paper-check:
	cd paper && make check-pdfa

paper-clean:
	cd paper && make clean

# Benchmark targets
benchmark:
	python benchmarks/run.py --benchmarks all --output benchmarks/results.csv

benchmark-quick:
	python benchmarks/run.py --benchmarks comparison --quick --output benchmarks/quick_results.csv

plot-bench:
	python scripts/make_plot.py --input benchmarks/results.csv --output paper/figs/

# Reproducibility targets
reproduce:
	./scripts/reproduce.sh

reproduce-quick:
	./scripts/reproduce.sh --quick

validate:
	python scripts/validate_results.py --results-dir results/reproduction

# Data targets
download-data:
	python scripts/download_data.py

verify-data:
	python scripts/verify_data.py

# Maintenance targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean paper-clean
	rm -rf results/
	rm -rf models/
	rm -rf logs/
	rm -rf .mypy_cache/
	docker system prune -f

quicktest:
	@echo "Running quick test suite..."
	./scripts/quicktest.sh

# Development helpers
dev-server:
	docker-compose up qmann-dev

jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

tensorboard:
	tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# CI/CD helpers
ci-test: install-dev lint test

ci-build: build docker

ci-paper: paper paper-check

# Release helpers
tag-release:
	@read -p "Enter version (e.g., v1.0.0): " version; \
	git tag -a $$version -m "Release $$version"; \
	git push origin $$version

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/

docs-serve:
	cd docs/_build && python -m http.server 8080

# Security
security-check:
	bandit -r src/
	safety check

# Performance profiling
profile:
	python -m cProfile -o profile.stats scripts/profile_qmann.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
memory-profile:
	mprof run scripts/memory_profile_qmann.py
	mprof plot

# Environment info
env-info:
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Git version: $(shell git --version)"
	@echo "Docker version: $(shell docker --version 2>/dev/null || echo 'Docker not installed')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not installed')"

# Project statistics
stats:
	@echo "Project Statistics"
	@echo "=================="
	@echo "Lines of Python code:"
	@find src/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "Lines of test code:"
	@find tests/ -name "*.py" -exec wc -l {} + | tail -1
	@echo "Number of Python files:"
	@find src/ -name "*.py" | wc -l
	@echo "Number of test files:"
	@find tests/ -name "*.py" | wc -l
