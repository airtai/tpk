#!/bin/bash

echo "Running pyup_dirs..."
pyup_dirs --py38-plus --recursive tpk examples tests benchmarks

echo "Running ruff..."
ruff tpk examples tests benchmarks --fix

echo "Running black..."
black tpk examples tests benchmarks
