#!/usr/bin/env bash
set -euo pipefail

python -m pipbench.enrich_prompts_api "$@"
