#!/usr/bin/env bash

set -o errexit

# Change directory to the project root directory.
cd "$(dirname "$0")"/..

# Install the dependencies into the mypy env.
pip install --editable ".[dev]" \
 --retries 1 \
 --no-input \
 --quiet

./scripts/static-analysis.sh
