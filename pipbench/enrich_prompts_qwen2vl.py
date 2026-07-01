from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from pipbench.data import load_metadata, resolve_data_path


SYSTEM_PROMPT = (
    "You are a preference inference model. Given several images representing a user's "
    "visual tastes, infer the user's aesthetic and semantic preferences (e.g., colors, "
    "composition, subjects, styles). When a new text prompt is provided, enrich it with "
    "additional details that align with these inferred preferences to make the output "
    "more personally fitting. Only output the enhanced prompt."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate enriched prompts with a PE-finetuned Qwen2.5-VL checkpoint."
    )
    parser.add_argument("--model-id", required=True, help="HF repo ID or local checkpoint path.")
    parser.add_argument(
        "--processor-id",
        default=None,
        help="Optional processor path/repo. Defaults to --model-id.",
    )
    parser.add_argument("--metadata", default="data/pipbench/metadata.json")
    parser.add_argument("--data-root", default="data/pipbench")
    parser.add_argument("--out", required=True, help="Output JSONL with an enriched_prompt field.")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--output-field", default="enriched_prompt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-pixels", type=int, default=784)
    parser.add_argument("--max-pixels", type=int, default=50176)
    return parser.parse_args()


def torch_dtype(name: str) -> str | torch.dtype:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def read_done_ids(path: Path, output_field: str) -> set[Any]:
    if not path.exists():
        return set()
    done: set[Any] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get(output_field):
                done.add(row.get("id"))
    return done


def build_messages(row: dict[str, Any], data_root: Path, prompt_field: str) -> list[dict[str, Any]]:
    prompt = row[prompt_field]
    content: list[dict[str, str]] = []
    for rel_path in row["ref_images"]:
        image_path = resolve_data_path(data_root, rel_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"missing reference image for row {row.get('id')}: {image_path}")
        content.append({"type": "image", "image": str(image_path.resolve())})
    content.append(
        {
            "type": "text",
            "text": f"Please enhance the following prompt in english:\n{prompt}",
        }
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def infer_qwen2_5_vl_processor_id(model_id: str) -> str | None:
    config_path = Path(model_id) / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_type") != "qwen2_5_vl":
        return None

    hidden_size = config.get("hidden_size")
    num_hidden_layers = config.get("num_hidden_layers")
    if hidden_size == 3584 and num_hidden_layers == 28:
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    if hidden_size == 5120 and num_hidden_layers == 64:
        return "Qwen/Qwen2.5-VL-32B-Instruct"
    return None


def load_processor(auto_processor: Any, args: argparse.Namespace):
    processor_id = args.processor_id or args.model_id
    try:
        return auto_processor.from_pretrained(
            processor_id,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    except OSError:
        if args.processor_id:
            raise
        fallback_id = infer_qwen2_5_vl_processor_id(args.model_id)
        if fallback_id is None:
            raise
        print(
            f"processor files not found in {args.model_id}; "
            f"falling back to {fallback_id}",
            flush=True,
        )
        return auto_processor.from_pretrained(
            fallback_id,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )


def load_model_and_processor(args: argparse.Namespace):
    try:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Qwen2.5-VL prompt enrichment requires recent transformers and qwen-vl-utils. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype(args.dtype)}
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    processor = load_processor(AutoProcessor, args)
    return model, processor, process_vision_info


def first_parameter_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_one(
    model: torch.nn.Module,
    processor: Any,
    process_vision_info: Any,
    messages: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(first_parameter_device(model))
    do_sample = args.temperature > 0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    generated_ids = model.generate(**inputs, **generate_kwargs)
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    rows = load_metadata(args.metadata)
    if args.limit is not None:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(out_path, args.output_field) if args.resume else set()
    mode = "a" if args.resume and out_path.exists() else "w"
    if not rows:
        out_path.open(mode, encoding="utf-8").close()
        print(f"no rows to enrich; out={out_path}")
        return

    model, processor, process_vision_info = load_model_and_processor(args)
    data_root = Path(args.data_root)
    with out_path.open(mode, encoding="utf-8") as handle:
        for row in tqdm(rows, desc="qwen2vl prompt enrichment"):
            if row.get("id") in done_ids:
                continue
            messages = build_messages(row, data_root, args.prompt_field)
            enriched_prompt = generate_one(model, processor, process_vision_info, messages, args)
            out_row = dict(row)
            out_row[args.output_field] = enriched_prompt
            out_row["prompt_enricher_model"] = args.model_id
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
