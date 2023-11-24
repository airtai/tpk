#!/bin/bash

echo "Running pyup_dirs..."
pyup_dirs --py38-plus --recursive temporal_data_kit examples tests benchmarks

echo "Running ruff..."
ruff temporal_data_kit examples tests benchmarks --fix

echo "Running black..."
black temporal_data_kit examples tests benchmarks
