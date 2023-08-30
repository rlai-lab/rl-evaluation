#!/bin/bash
set -e
pyright --stats

export PYTHONPATH=RlEvaluation
python3 -m unittest discover -p "*test_*.py"
