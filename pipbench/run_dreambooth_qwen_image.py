from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm

from pipbench.data import load_metadata, resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-case DreamBooth LoRA on Qwen-Image.")
    parser.add_argument("--metadata", default="data/pipbench/metadata.json")
    parser.add_argument("--data-root", default="data/pipbench")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", default="outputs/dreambooth_work")
    parser.add_argument("--model-id", default="Qwen/Qwen-Image")
    parser.add_argument("--train-script", default="third_party/diffusers/train_dreambooth_lora_qwen_image.py")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.rank :: args.world_size]
    out_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    for row in tqdm(rows, desc="dreambooth-qwen-image"):
        out_path = out_dir / f"{row['id']}.png"
        if args.resume and out_path.exists():
            continue
        case_dir = work_dir / str(row["id"])
        instance_dir = case_dir / "instance"
        lora_dir = case_dir / "lora"
        if instance_dir.exists():
            shutil.rmtree(instance_dir)
        instance_dir.mkdir(parents=True, exist_ok=True)
        for ref in row["ref_images"]:
            src = resolve_data_path(args.data_root, ref)
            shutil.copy(src, instance_dir / Path(ref).name)
        prompt = row[args.prompt_field]
        train_cmd = [
            "accelerate",
            "launch",
            args.train_script,
            f"--pretrained_model_name_or_path={args.model_id}",
            f"--instance_data_dir={instance_dir}",
            f"--output_dir={lora_dir}",
            "--mixed_precision=bf16",
            "--instance_prompt=a photo of sks",
            "--resolution=1024",
            "--train_batch_size=1",
            "--gradient_accumulation_steps=4",
            "--learning_rate=2e-4",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            f"--max_train_steps={args.max_train_steps}",
            "--seed=0",
        ]
        infer_cmd = [
            "python",
            "-m",
            "pipbench.infer_dreambooth_qwen_image",
            f"--model-id={args.model_id}",
            f"--lora-dir={lora_dir}",
            f"--prompt=sks {prompt}",
            f"--output={out_path}",
        ]
        if args.dry_run:
            print(" ".join(map(str, train_cmd)))
            print(" ".join(map(str, infer_cmd)))
            continue
        subprocess.run(train_cmd, check=True)
        subprocess.run(infer_cmd, check=True)


if __name__ == "__main__":
    main()
