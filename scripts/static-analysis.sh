#!/bin/bash
set -e

echo "Running mypy..."
mypy temporal_data_kit tests benchmarks

echo "Running bandit..."
bandit -c pyproject.toml -r temporal_data_kit

echo "Running semgrep..."
semgrep scan --config auto --error
