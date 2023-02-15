#!/bin/bash
set -e
mypy -p RlEvaluation

export PYTHONPATH=RlEvaluation
python3 -m unittest discover -p "*test_*.py"
