from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one image with a Qwen-Image DreamBooth LoRA.")
    parser.add_argument("--model-id", default="Qwen/Qwen-Image")
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    from diffusers import DiffusionPipeline

    args = parse_args()
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(args.lora_dir)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
        num_inference_steps=args.steps,
        true_cfg_scale=args.true_cfg_scale,
        generator=generator,
    ).images[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


if __name__ == "__main__":
    main()
