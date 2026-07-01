#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

rm -rf .smoke
mkdir -p .smoke/data/L2-benchmark/images/0 .smoke/preds .smoke/arena .smoke/results

python - <<'PY'
from pathlib import Path
from PIL import Image
import json

root = Path(".smoke/data")
for name, color in {
    "0.png": (255, 0, 0),
    "1.png": (0, 255, 0),
    "2.png": (0, 0, 255),
}.items():
    Image.new("RGB", (32, 32), color).save(root / "L2-benchmark" / "images" / "0" / name)
Image.new("RGB", (32, 32), (0, 0, 254)).save(".smoke/preds/0.png")
metadata = [{
    "id": 0,
    "image_id": 0,
    "category": "synthetic",
    "prompt": "a blue square",
    "ref_images": ["L2-benchmark/images/0/0.png", "L2-benchmark/images/0/1.png"],
    "gt_images": "L2-benchmark/images/0/2.png",
}]
(root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

compete = [
    {"id": 0, "model_A": "A", "model_B": "B", "image_A": "a.png", "image_B": "b.png"},
    {"id": 1, "model_A": "A", "model_B": "B", "image_A": "a.png", "image_B": "b.png"},
]
results = [
    {"id": 0, "flipped": False, "result": {"comparison_and_choice": {"final_decision": {"preferred_image": "img1", "confidence": 1.0}}}},
    {"id": 1, "flipped": False, "result": {"comparison_and_choice": {"final_decision": {"preferred_image": "tie", "confidence": 1.0}}}},
]
with open(".smoke/arena/compete_data.jsonl", "w", encoding="utf-8") as f:
    for row in compete:
        f.write(json.dumps(row) + "\n")
with open(".smoke/arena/results.jsonl", "w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row) + "\n")
PY

python -m compileall -q pipbench
python -m pipbench.data inspect --metadata .smoke/data/metadata.json --data-root .smoke/data
python -m pipbench.evaluate_outputs \
  --data-root .smoke/data \
  --metadata .smoke/data/metadata.json \
  --pred-dir .smoke/preds \
  --pred-pattern "{id}.png" \
  --out .smoke/results/metrics.jsonl \
  --metrics pixel \
  --device cpu \
  --missing error
python -m pipbench.elo \
  --compete-data .smoke/arena/compete_data.jsonl \
  --results .smoke/arena/results.jsonl \
  --out-dir .smoke/results/elo \
  --confidence-threshold 0.0
python -m pipbench.enrich_prompts_qwen2vl \
  --model-id dummy \
  --metadata .smoke/data/metadata.json \
  --data-root .smoke/data \
  --out .smoke/results/enriched_empty.jsonl \
  --limit 0
python -m pipbench.generate_qwen_image_edit --help >/dev/null
python -m pipbench.enrich_prompts_api --help >/dev/null
python -m pipbench.run_dreambooth_qwen_image --help >/dev/null
python -m pipbench.run_fabric --help >/dev/null

test -f .smoke/results/metrics.jsonl
test -f .smoke/results/elo/elo_scores.json
echo "smoke test passed"
