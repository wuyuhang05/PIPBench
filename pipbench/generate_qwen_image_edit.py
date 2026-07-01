from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from pipbench.data import load_metadata, resolve_data_path


PROMPT_PREFIX = "Generate an image that matches the following prompt and fits the user's visual preferences shown in these images: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-Image-Edit on PIPBench with 1 or 2 reference images.")
    parser.add_argument("--metadata", default="data/pipbench/metadata.json")
    parser.add_argument("--data-root", default="data/pipbench")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=None, help="Optional HF repo ID or local pipeline path.")
    parser.add_argument("--ref-count", type=int, choices=[1, 2], required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Optional diffusers device_map, e.g. balanced.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_pipeline(ref_count: int, model_id: str | None, device: str, device_map: str | None):
    pipeline_kwargs = {}
    if device_map:
        pipeline_kwargs["device_map"] = device_map

    if ref_count == 1:
        from diffusers import QwenImageEditPipeline

        pipe = QwenImageEditPipeline.from_pretrained(
            model_id or "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
            **pipeline_kwargs,
        )
    else:
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id or "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.bfloat16,
            **pipeline_kwargs,
        )
    if device_map:
        return pipe
    return pipe.to(device)


def select_refs(row: dict, data_root: Path, ref_count: int, rng: random.Random) -> Image.Image | list[Image.Image]:
    refs = list(row["ref_images"])
    if len(refs) < ref_count:
        raise ValueError(f"row {row.get('id')} has {len(refs)} refs, need {ref_count}")
    selected = rng.sample(refs, k=ref_count)
    images = [Image.open(resolve_data_path(data_root, path)).convert("RGB") for path in selected]
    return images[0] if ref_count == 1 else images


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"no rows to generate; output_dir={out_dir}")
        return

    pipe = load_pipeline(args.ref_count, args.model_id, device, args.device_map)
    generator_device = "cpu" if args.device_map else device
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    for row in tqdm(rows, desc=f"qwen-image-edit-{args.ref_count}ref"):
        out_path = out_dir / f"{row['id']}.png"
        if args.resume and out_path.exists():
            continue
        ref_image = select_refs(row, data_root, args.ref_count, rng)
        image = pipe(
            image=ref_image,
            prompt=PROMPT_PREFIX + row[args.prompt_field],
            negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate",
            num_inference_steps=args.steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator,
        ).images[0]
        image.save(out_path)


if __name__ == "__main__":
    main()
