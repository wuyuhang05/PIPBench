from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from pipbench.data import load_metadata


def build_pipeline(model: str, model_id: str | None, device: str, device_map: str | None):
    from diffusers import DiffusionPipeline

    pipeline_kwargs = {}
    if device_map:
        pipeline_kwargs["device_map"] = device_map

    if model == "qwen-image":
        model_id = model_id or "Qwen/Qwen-Image"
        dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, **pipeline_kwargs)
    elif model == "sdxl":
        model_id = model_id or "stabilityai/stable-diffusion-xl-base-1.0"
        dtype = torch.float16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
            **pipeline_kwargs,
        )
    else:
        raise ValueError(f"unknown baseline model: {model}")
    if device_map:
        return pipe
    return pipe.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional prompt-only baseline generation.")
    parser.add_argument("--model", choices=["qwen-image", "sdxl"], required=True)
    parser.add_argument("--model-id", default=None, help="Optional HF repo ID or local pipeline path.")
    parser.add_argument("--metadata", default="data/pipbench/metadata.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None, help="Optional diffusers device_map, e.g. balanced.")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"no rows to generate; output_dir={output_dir}")
        return
    pipe = build_pipeline(args.model, args.model_id, device, args.device_map)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for row in tqdm(rows, desc=args.model):
        output_path = output_dir / f"{row['id']}.png"
        if args.resume and output_path.exists():
            continue
        if args.prompt_field not in row:
            raise KeyError(f"row {row.get('id')} does not contain prompt field {args.prompt_field!r}")
        kwargs = {"prompt": row[args.prompt_field], "generator": generator}
        if args.model == "qwen-image":
            kwargs.update(
                {
                    "negative_prompt": "lowres, bad anatomy, bad hands, cropped, worst quality",
                    "true_cfg_scale": args.true_cfg_scale,
                    "num_inference_steps": args.steps,
                }
            )
        image = pipe(**kwargs).images[0]
        image.save(output_path)


if __name__ == "__main__":
    main()
