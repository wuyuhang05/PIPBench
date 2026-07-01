from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pipbench.data import load_metadata, resolve_data_path
from pipbench.enrich_prompts_qwen2vl import SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PE prompts with an OpenAI-compatible vision API.")
    parser.add_argument("--model", required=True, help="Model name on the selected API gateway.")
    parser.add_argument("--metadata", default="data/pipbench/metadata.json")
    parser.add_argument("--data-root", default="data/pipbench")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--output-field", default="enriched_prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def encode_image(path: Path) -> str:
    with path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def done_ids(path: Path, output_field: str) -> set[Any]:
    if not path.exists():
        return set()
    ids: set[Any] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get(output_field):
                    ids.add(row.get("id"))
    return ids


def call_api(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai to run API-based prompt enrichment.") from exc
    if not args.api_key:
        raise ValueError("OPENAI_API_KEY or --api-key is required")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    content: list[dict[str, Any]] = []
    for rel_path in row["ref_images"]:
        image_path = resolve_data_path(args.data_root, rel_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"missing reference image for row {row.get('id')}: {image_path}")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
            }
        )
    content.append({"type": "text", "text": f"Prompt to be enhanced:\n{row[args.prompt_field]}"})

    last_error: Exception | None = None
    for attempt in range(5):
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            enriched = completion.choices[0].message.content
            out = dict(row)
            out[args.output_field] = enriched.strip() if enriched else ""
            out["prompt_enricher_model"] = args.model
            return out
        except Exception as exc:  # pragma: no cover - network/API dependent
            last_error = exc
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"API failed for row {row.get('id')}") from last_error


def main() -> None:
    args = parse_args()
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed = done_ids(out_path, args.output_field) if args.resume else set()
    rows = [row for row in rows if row.get("id") not in completed]
    if not rows:
        out_path.open("a", encoding="utf-8").close()
        print(f"no rows to enrich; out={out_path}")
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool, out_path.open("a", encoding="utf-8") as handle:
        futures = [pool.submit(call_api, args, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"api PE {args.model}"):
            handle.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
