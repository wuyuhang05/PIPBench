#!/usr/bin/env bash
set -euo pipefail

python -m pipbench.data download \
  --repo-id "${HF_DATASET_REPO:-AirRain03/PIPBench}" \
  --local-dir "${PIPBENCH_DATA_DIR:-data/pipbench}" \
  "$@"
