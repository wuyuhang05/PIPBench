from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from pipbench.data import load_metadata, resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FABRIC with PIPBench reference images as positive feedback.")
    parser.add_argument("--metadata", default="data/pipbench/metadata.json")
    parser.add_argument("--data-root", default="data/pipbench")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="dreamlike-art/dreamlike-photoreal-2.0")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--ref-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from fabric.generator import AttentionBasedGenerator
    except ImportError as exc:
        raise ImportError("Install FABRIC first: pip install git+https://github.com/sd-fabric/fabric.git") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    generator = AttentionBasedGenerator(model_name=args.model_name, torch_dtype=dtype).to(device)
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    for row in tqdm(rows, desc="fabric"):
        out_path = out_dir / f"{row['id']}.png"
        if args.resume and out_path.exists():
            continue
        liked = [
            Image.open(resolve_data_path(data_root, path)).convert("RGB")
            for path in row["ref_images"][: args.ref_count]
        ]
        images = generator.generate(
            prompt=row[args.prompt_field],
            negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
            liked=liked,
            disliked=[],
            seed=args.seed + int(row["id"]),
            n_images=1,
            guidance_scale=args.guidance_scale,
            denoising_steps=args.steps,
        )
        images[0].save(out_path)


if __name__ == "__main__":
    main()
