#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PE_DIR="$ROOT/third_party/qwen-vl-finetune"

export USE_TF="${USE_TF:-0}"

cd "$PE_DIR"
torchrun "$@"
