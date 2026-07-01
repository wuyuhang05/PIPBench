from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image


def open_rgb(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


def to_gray(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")


def ensure_same_size(reference: Image.Image, prediction: Image.Image) -> Image.Image:
    if prediction.size == reference.size:
        return prediction
    return prediction.resize(reference.size, Image.Resampling.BICUBIC)


def format_prediction_path(pattern: str, row: dict) -> Path:
    return Path(
        pattern.format(
            id=row["id"],
            image_id=row.get("image_id", row["id"]),
            category=row.get("category", ""),
        )
    )


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def psnr_from_mse(mse: float) -> float:
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)
