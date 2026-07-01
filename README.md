# PIPBench

**PIPBench: A Profile-Inclusive Framework for Personalized Image Generation Evaluation**

Accepted to **ECCV 2026**.

This repository hosts the PIPBench project page and the code for reproducing
the prompt-enrichment (PE) experiments and related image-generation baselines.
The benchmark data and released PE checkpoints are hosted on Hugging Face:

- Project page: https://wuyuhang05.github.io/PIPBench/
- Dataset: https://huggingface.co/datasets/AirRain03/PIPBench
- PE 7B checkpoint: https://huggingface.co/AirRain03/qwen2vl_7b
- PE 32B checkpoint: https://huggingface.co/AirRain03/qwen2vl_32b

The dataset uses `metadata.json` rows with:

```json
{
  "id": 0,
  "image_id": 0,
  "category": "synthetic",
  "prompt": "...",
  "ref_images": ["L2-benchmark/images/0/5.png"],
  "gt_images": "L2-benchmark/images/0/0.png"
}
```

## Setup

```bash
git clone <this-repo-url> PIPBench
cd PIPBench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Qwen-Image, Qwen-Image-Edit, and Qwen2.5-VL runs, use a CUDA environment
with enough GPU memory and access to the corresponding Hugging Face models.

## Download Data

```bash
bash scripts/download_data.sh
```

This downloads `AirRain03/PIPBench` into `data/pipbench`.

```bash
python -m pipbench.data inspect \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench
```

## Run Released PE Modules

Generate enriched prompts with the released 7B PE module:

```bash
bash scripts/enrich_prompts_qwen2vl.sh \
  --model-id AirRain03/qwen2vl_7b \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --out outputs/prompts/qwen2vl_7b.jsonl \
  --device-map auto \
  --dtype bfloat16 \
  --resume
```

Generate enriched prompts with the released 32B PE module:

```bash
bash scripts/enrich_prompts_qwen2vl.sh \
  --model-id AirRain03/qwen2vl_32b \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --out outputs/prompts/qwen2vl_32b.jsonl \
  --device-map auto \
  --dtype bfloat16 \
  --resume
```

Run Qwen-Image from the enriched prompts:

```bash
bash scripts/run_baseline_generation.sh \
  --model qwen-image \
  --metadata outputs/prompts/qwen2vl_7b.jsonl \
  --prompt-field enriched_prompt \
  --output-dir outputs/qwen_image_qwen2vl_7b \
  --device cuda \
  --resume
```

The same command works for `outputs/prompts/qwen2vl_32b.jsonl`.
Use `--model-id /local/path/to/Qwen-Image` to run from a local cache.
Use `--device-map balanced` when you want diffusers to shard the pipeline.

For local checkpoints, replace `--model-id` with the checkpoint directory. If the
checkpoint directory does not include processor files, the script falls back to
the matching base Qwen2.5-VL processor automatically; you can also pass it
explicitly:

```bash
export HF_HOME="$PWD/.hf_cache"

bash scripts/enrich_prompts_qwen2vl.sh \
  --model-id /path/to/qwen2vl_7b/checkpoint \
  --processor-id Qwen/Qwen2.5-VL-7B-Instruct \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --out outputs/prompts/qwen2vl_7b_local.jsonl \
  --device-map auto \
  --dtype bfloat16 \
  --resume
```

## Train PE Modules

The PE SFT code is included under `third_party/qwen-vl-finetune/`. It is the
Qwen2.5-VL supervised fine-tuning setup used for the released PE modules.

Prepare training data in the Qwen-VL conversation format. Each sample should
contain reference images and a target enriched prompt:

```json
{
  "image": ["data/images-L2/736/4.png", "data/images-L2/736/7.png"],
  "conversations": [
    {
      "from": "system",
      "value": "You are a preference inference model..."
    },
    {
      "from": "user",
      "value": "<image>\n<image>\nPlease enhance the following prompt in english:\n..."
    },
    {
      "from": "assistant",
      "value": "..."
    }
  ]
}
```

Register the dataset in
`third_party/qwen-vl-finetune/qwenvl/data/__init__.py`, then launch training.
Example 7B command:

```bash
cd third_party/qwen-vl-finetune
export USE_TF=0
export PIPBENCH_PE_DATA_ROOT=/path/to/project/root
export PIPBENCH_PE_L2_REAL_ANNOTATION=/path/to/vlm_train_imageonly-L2-real.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node=8 \
  --master_port=23868 \
  qwenvl/train/train_qwen.py \
  --deepspeed scripts/zero3.json \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset_use perpe_l2_real \
  --data_flatten True \
  --tune_mm_vision False \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 \
  --output_dir output/qwen2vl_pe_7b \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_pixels 50176 \
  --min_pixels 784 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 250 \
  --save_total_limit 5 \
  --learning_rate 2e-7 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --run_name qwen2vl-pe-7b \
  --report_to none
```

For 32B, switch `--model_name_or_path` to `Qwen/Qwen2.5-VL-32B-Instruct`,
reduce per-device batch size, and use 8x80GB GPUs with ZeRO-3 as in
`third_party/qwen-vl-finetune/scripts/sft_32b.sh`.

## GPT / Gemini PE

The API PE runner uses an OpenAI-compatible vision chat endpoint. Set your
gateway URL and API key, then pass the model name used by your gateway.

```bash
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_API_KEY="..."

bash scripts/enrich_prompts_api.sh \
  --model gpt-5-2025-08-07 \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --out outputs/prompts/gpt_pe.jsonl \
  --max-workers 16 \
  --resume
```

Gemini-style runs use the same script if exposed through the same compatible
gateway:

```bash
bash scripts/enrich_prompts_api.sh \
  --model gemini-3-pro \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --out outputs/prompts/gemini_pe.jsonl \
  --max-workers 8 \
  --resume
```

If your gateway names Gemini differently, replace `gemini-3-pro` with the
actual model id.

## Qwen-Image-Edit Baselines

Run the one-reference edit baseline:

```bash
bash scripts/run_qwen_image_edit.sh \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --ref-count 1 \
  --prompt-field prompt \
  --output-dir outputs/qwen_image_edit_1ref \
  --device cuda \
  --resume
```

Run the two-reference edit baseline:

```bash
bash scripts/run_qwen_image_edit.sh \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --ref-count 2 \
  --prompt-field prompt \
  --output-dir outputs/qwen_image_edit_2ref \
  --device cuda \
  --resume
```

Use `--prompt-field enriched_prompt` and a PE output JSONL if you want to run
Qwen-Image-Edit after prompt enrichment.
Use `--model-id /local/path/to/Qwen-Image-Edit` or
`--model-id /local/path/to/Qwen-Image-Edit-2509` to run from local caches.
Use `--device-map balanced` if the edit model cannot fit on a single GPU.

## DreamBooth On Qwen-Image

The release includes the Qwen-Image DreamBooth LoRA training script from
Diffusers at `third_party/diffusers/train_dreambooth_lora_qwen_image.py` and a
benchmark wrapper that trains one LoRA per PIPBench case.

Dry-run the commands for the first case:

```bash
bash scripts/run_dreambooth_qwen_image.sh \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --output-dir outputs/dreambooth_qwen_image \
  --work-dir outputs/dreambooth_work \
  --limit 1 \
  --dry-run
```

Run the benchmark:

```bash
bash scripts/run_dreambooth_qwen_image.sh \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --output-dir outputs/dreambooth_qwen_image \
  --work-dir outputs/dreambooth_work \
  --max-train-steps 400 \
  --resume
```

For multi-GPU sharding, launch multiple processes manually with different
`--rank` values and the same `--world-size`.
Use `--model-id /local/path/to/Qwen-Image` to run from a local cache.

## FABRIC Baseline

FABRIC depends on an older Diffusers API, while Qwen-Image needs a recent
Diffusers release. Run FABRIC in a separate environment:

```bash
python -m venv .venv-fabric
source .venv-fabric/bin/activate
pip install -r /path/to/fabric/requirements.txt
pip install git+https://github.com/sd-fabric/fabric.git
```

Then run PIPBench references as positive feedback images:

```bash
bash scripts/run_fabric.sh \
  --metadata data/pipbench/metadata.json \
  --data-root data/pipbench \
  --output-dir outputs/fabric \
  --model-name dreamlike-art/dreamlike-photoreal-2.0 \
  --ref-count 4 \
  --resume
```

## Evaluate Outputs

Generated images should be named `{id}.png` under the output directory.

```bash
bash scripts/evaluate_outputs.sh \
  --data-root data/pipbench \
  --metadata data/pipbench/metadata.json \
  --pred-dir outputs/qwen_image_qwen2vl_7b \
  --pred-pattern "{id}.png" \
  --out results/qwen_image_qwen2vl_7b_metrics.jsonl \
  --metrics lpips clip dino \
  --device cuda
```

For a lightweight CPU check:

```bash
bash scripts/evaluate_outputs.sh \
  --data-root data/pipbench \
  --metadata data/pipbench/metadata.json \
  --pred-dir outputs/MODEL_NAME \
  --pred-pattern "{id}.png" \
  --out results/MODEL_NAME_pixel_metrics.jsonl \
  --metrics pixel \
  --device cpu
```

## Elo / Bradley-Terry Ratings

Pairwise comparison metadata should be JSONL:

```json
{"id": 0, "model_A": "model1", "model_B": "model2", "image_A": "...", "image_B": "..."}
```

Judge result JSONL should contain `preferred_image` under
`result.comparison_and_choice.final_decision`.

```bash
bash scripts/compute_elo.sh \
  --compete-data arena/compete_data.jsonl \
  --results arena/compete_results_gpt.jsonl arena/compete_results_qwen.jsonl \
  --out-dir results/elo \
  --confidence-threshold 0.7
```

## Smoke Test

```bash
bash scripts/smoke_test.sh
```

The smoke test checks imports, metadata validation, pixel metrics, Elo, and
argument parsing for the PE/baseline entry points. It does not download or load
large models.

## License

The code release uses the MIT License. Third-party code under `third_party/`
keeps its upstream license headers where provided.
