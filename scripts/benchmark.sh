#!/usr/bin/env bash

echo "Running benchmarks for TXMixer on M5..."
python -m benchmarks.run_tsmixer --model tsmixer --n_block 2 --hidden_size 64 --dropout 0 --data_dir data/m5/
