from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator, Sequence


DEFAULT_REPO_ID = "AirRain03/PIPBench"


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_metadata(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"metadata file does not exist: {path}")
    if path.suffix == ".jsonl":
        return list(iter_jsonl(path))
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def resolve_data_path(data_root: str | Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else Path(data_root) / path


def validate_metadata(rows: Sequence[dict], data_root: str | Path | None = None) -> dict:
    ids = [row.get("id") for row in rows]
    image_ids = {row.get("image_id") for row in rows}
    categories: dict[str, int] = {}
    referenced_images: set[str] = set()
    missing: list[str] = []

    for row in rows:
        for field in ("id", "image_id", "category", "prompt", "ref_images", "gt_images"):
            if field not in row:
                raise ValueError(f"missing field {field!r} in row: {row}")
        categories[row["category"]] = categories.get(row["category"], 0) + 1
        paths = [*row["ref_images"], row["gt_images"]]
        referenced_images.update(paths)
        if data_root is not None:
            for path in paths:
                if not resolve_data_path(data_root, path).is_file():
                    missing.append(path)

    return {
        "rows": len(rows),
        "ids_contiguous": ids == list(range(len(rows))),
        "image_ids_contiguous": image_ids == set(range(len(image_ids))),
        "categories": dict(sorted(categories.items())),
        "unique_images": len(referenced_images),
        "missing_images": missing,
    }


def download_snapshot(
    repo_id: str,
    local_dir: str | Path,
    revision: str | None = None,
    max_workers: int | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Please install huggingface_hub to download the dataset.") from exc

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        max_workers=max_workers or int(os.environ.get("PIPBENCH_HF_MAX_WORKERS", "2")),
    )
    return local_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PIPBench data helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    download = sub.add_parser("download", help="Download the Hugging Face dataset snapshot.")
    download.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download.add_argument("--local-dir", default="data/pipbench")
    download.add_argument("--revision", default=None)
    download.add_argument("--max-workers", type=int, default=None)

    inspect = sub.add_parser("inspect", help="Validate metadata and optional image files.")
    inspect.add_argument("--metadata", default="data/pipbench/metadata.json")
    inspect.add_argument("--data-root", default="data/pipbench")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "download":
        local_dir = download_snapshot(args.repo_id, args.local_dir, args.revision, args.max_workers)
        print(f"dataset_dir={local_dir}")
    elif args.cmd == "inspect":
        rows = load_metadata(args.metadata)
        summary = validate_metadata(rows, args.data_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
