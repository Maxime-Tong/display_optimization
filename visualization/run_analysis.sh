#!/usr/bin/env bash
# Execution wrapper for the LUT specialization visualization suite.
#
# Usage:
#   ./run_analysis.sh                       # default 3dgs conda env python
#   PYTHON=/path/to/python ./run_analysis.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
else
  PYTHON_BIN="D:/miniconda3/envs/3dgs/python.exe"
fi

"$PYTHON_BIN" visualize_lut.py --config config.yaml
