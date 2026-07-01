from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchvision as tv
from PIL import Image
from tqdm import tqdm

from pipbench.data import load_metadata, resolve_data_path
from pipbench.utils import ensure_same_size, format_prediction_path, open_rgb, psnr_from_mse, to_gray, write_jsonl


MetricFn = Callable[[dict, Image.Image, Image.Image, list[Image.Image]], dict[str, float]]


def pil_to_tensor_01(image: Image.Image) -> torch.Tensor:
    return tv.transforms.ToTensor()(image).unsqueeze(0)


def pixel_metrics(row: dict, pred: Image.Image, gt: Image.Image, refs: list[Image.Image]) -> dict[str, float]:
    pred = ensure_same_size(gt, pred)
    a = np.asarray(pred, dtype=np.float32) / 255.0
    b = np.asarray(gt, dtype=np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    return {"Pixel_MSE": mse, "Pixel_MAE": mae, "Pixel_PSNR": psnr_from_mse(mse)}


class LPIPSEvaluator:
    def __init__(self, device: str, input_size: int = 256, net: str = "vgg") -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Install lpips to compute LPIPS: pip install lpips") from exc
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.loss = lpips.LPIPS(net=net).to(self.device).eval()
        self.resize = tv.transforms.Compose([
            tv.transforms.Resize((input_size, input_size), interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])

    @torch.no_grad()
    def score(self, ref: Image.Image, pred: Image.Image) -> float:
        ref = ensure_same_size(ref, ref)
        pred = ensure_same_size(ref, pred)
        x = self.resize(ref).unsqueeze(0).to(self.device)
        y = self.resize(pred).unsqueeze(0).to(self.device)
        return float(self.loss(x, y).item())

    def __call__(self, row: dict, pred: Image.Image, gt: Image.Image, refs: list[Image.Image]) -> dict[str, float]:
        ref_scores = [self.score(ref, pred) for ref in refs]
        ref_gray_scores = [self.score(to_gray(ref), to_gray(pred)) for ref in refs]
        return {
            "LPIPS": self.score(gt, pred),
            "LPIPS_gray": self.score(to_gray(gt), to_gray(pred)),
            "LPIPS_ref": float(sum(ref_scores) / len(ref_scores)),
            "LPIPS_gray_ref": float(sum(ref_gray_scores) / len(ref_gray_scores)),
        }


class CLIPEvaluator:
    def __init__(self, device: str, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("Install open-clip-torch to compute CLIP metrics.") from exc
        self.open_clip = open_clip
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def image_embed(self, image: Image.Image) -> torch.Tensor:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(tensor)
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_embed(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)
        return emb / emb.norm(dim=-1, keepdim=True)

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(max(100.0 * (a * b).sum(dim=-1).item(), 0.0))

    def __call__(self, row: dict, pred: Image.Image, gt: Image.Image, refs: list[Image.Image]) -> dict[str, float]:
        pred_emb = self.image_embed(pred)
        pred_gray_emb = self.image_embed(to_gray(pred))
        gt_emb = self.image_embed(gt)
        gt_gray_emb = self.image_embed(to_gray(gt))
        text_emb = self.text_embed(row["prompt"])
        ref_scores = [self.cosine(self.image_embed(ref), pred_emb) for ref in refs]
        ref_gray_scores = [self.cosine(self.image_embed(to_gray(ref)), pred_gray_emb) for ref in refs]
        return {
            "CLIP_II": self.cosine(gt_emb, pred_emb),
            "CLIP_II_gray": self.cosine(gt_gray_emb, pred_gray_emb),
            "CLIP_TI": self.cosine(text_emb, pred_emb),
            "CLIP_TI_gray": self.cosine(text_emb, pred_gray_emb),
            "CLIP_II_ref": float(sum(ref_scores) / len(ref_scores)),
            "CLIP_II_gray_ref": float(sum(ref_gray_scores) / len(ref_gray_scores)),
        }


class DINOEvaluator:
    def __init__(self, device: str, model_name: str = "dinov2_vits14", input_size: int = 224) -> None:
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                model_name,
                trust_repo=True,
                skip_validation=True,
            ).to(self.device).eval()
        except TypeError:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.preprocess = tv.transforms.Compose([
            tv.transforms.Resize(input_size, interpolation=tv.transforms.InterpolationMode.BICUBIC),
            tv.transforms.CenterCrop(input_size),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, image: Image.Image) -> torch.Tensor:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        emb = self.model(tensor)
        return emb / emb.norm(dim=-1, keepdim=True)

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a * b).sum(dim=-1).item() * 100.0)

    def __call__(self, row: dict, pred: Image.Image, gt: Image.Image, refs: list[Image.Image]) -> dict[str, float]:
        pred_emb = self.embed(pred)
        pred_gray_emb = self.embed(to_gray(pred))
        gt_emb = self.embed(gt)
        gt_gray_emb = self.embed(to_gray(gt))
        ref_scores = [self.cosine(self.embed(ref), pred_emb) for ref in refs]
        ref_gray_scores = [self.cosine(self.embed(to_gray(ref)), pred_gray_emb) for ref in refs]
        return {
            "DINO_Cosine": self.cosine(gt_emb, pred_emb),
            "DINO_Cosine_gray": self.cosine(gt_gray_emb, pred_gray_emb),
            "DINO_Cosine_ref": float(sum(ref_scores) / len(ref_scores)),
            "DINO_Cosine_gray_ref": float(sum(ref_gray_scores) / len(ref_gray_scores)),
        }


def build_metric_fns(metric_names: list[str], device: str) -> list[MetricFn]:
    fns: list[MetricFn] = []
    for name in metric_names:
        if name == "pixel":
            fns.append(pixel_metrics)
        elif name == "lpips":
            fns.append(LPIPSEvaluator(device=device))
        elif name == "clip":
            fns.append(CLIPEvaluator(device=device))
        elif name == "dino":
            fns.append(DINOEvaluator(device=device))
        else:
            raise ValueError(f"unknown metric: {name}")
    return fns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated PIPBench output images.")
    parser.add_argument("--data-root", default="data/pipbench")
    parser.add_argument("--metadata", default="data/pipbench/metadata.json")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--pred-pattern", default="{id}.png")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--metrics", nargs="+", default=["lpips", "clip", "dino"], choices=["pixel", "lpips", "clip", "dino"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--missing", choices=["skip", "error"], default="skip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    pred_dir = Path(args.pred_dir)
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]
    metric_fns = build_metric_fns(args.metrics, args.device)

    results: list[dict] = []
    missing = 0
    for row in tqdm(rows, desc="evaluate"):
        pred_path = pred_dir / format_prediction_path(args.pred_pattern, row)
        if not pred_path.is_file():
            missing += 1
            if args.missing == "error":
                raise FileNotFoundError(f"missing prediction: {pred_path}")
            continue
        pred = open_rgb(pred_path)
        gt = open_rgb(resolve_data_path(data_root, row["gt_images"]))
        refs = [open_rgb(resolve_data_path(data_root, path)) for path in row["ref_images"]]
        record = {"id": row["id"], "image_id": row.get("image_id"), "category": row["category"]}
        for fn in metric_fns:
            record.update(fn(row, pred, gt, refs))
        results.append(record)

    write_jsonl(results, args.out)
    metrics = [key for key in results[0].keys() if key not in {"id", "image_id", "category"}] if results else []
    summary = {"evaluated": len(results), "missing": missing, "metrics": {}}
    for metric in metrics:
        values = [row[metric] for row in results if np.isfinite(row[metric])]
        summary["metrics"][metric] = float(np.mean(values)) if values else None
    summary_out = Path(args.summary_out) if args.summary_out else Path(args.out).with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
