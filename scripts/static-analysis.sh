#!/bin/bash
set -e

echo "Running mypy..."
mypy tpk tests benchmarks

echo "Running bandit..."
bandit -c pyproject.toml -r tpk

echo "Running semgrep..."
semgrep scan --config auto --error
