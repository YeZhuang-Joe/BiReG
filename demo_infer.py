import argparse
import json
import os
from datetime import datetime

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel

from RegionalKolorsDiffusion_xl import RegionalDiffusionXLPipeline
from demo_examples import DEMO_CASES
from modeling_chatglm import ChatGLMModel
from tokenization_chatglm import ChatGLMTokenizer


# =========================
# 0. 默认 Demo 配置（可被 case 内 config 覆盖）
# =========================
DEFAULT_DEMO_CONFIG = {
    "batch_size": 1,
    "base_ratio": 0.2,
    "num_inference_steps": 40,
    "height": 768,
    "width": 1536,
    "seed": 1234,
    "guidance_scale": 7.5,
    "negative_prompt": "",
}

# =========================
# 1. 路径配置
# =========================
MODEL_ROOT = "/root/autodl-tmp/weights/Kolors"
OUTPUT_DIR = "outputs/demo"


# =========================
# 2. 加载模型
# =========================
def load_pipeline():
    text_encoder = ChatGLMModel.from_pretrained(
        f"{MODEL_ROOT}/text_encoder",
        torch_dtype=torch.float16,
        local_files_only=True
    ).half()

    tokenizer = ChatGLMTokenizer.from_pretrained(
        f"{MODEL_ROOT}/text_encoder",
        local_files_only=True
    )

    vae = AutoencoderKL.from_pretrained(
        f"{MODEL_ROOT}/vae",
        local_files_only=True
    ).half()

    scheduler = EulerDiscreteScheduler.from_pretrained(
        f"{MODEL_ROOT}/scheduler",
        local_files_only=True
    )

    unet = UNet2DConditionModel.from_pretrained(
        f"{MODEL_ROOT}/unet",
        local_files_only=True
    ).half()

    pipe = RegionalDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False
    )

    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


# =========================
# 3. 工具函数
# =========================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_case(case_name: str):
    if case_name not in DEMO_CASES:
        raise ValueError(
            f"Unknown case: {case_name}\n"
            f"Available cases: {list(DEMO_CASES.keys())}"
        )
    return DEMO_CASES[case_name]


def merge_config(case_config=None):
    if case_config is None:
        case_config = {}
    return {**DEFAULT_DEMO_CONFIG, **case_config}


def build_save_base_path(case_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_DIR, f"{case_name}_{timestamp}")


def save_metadata(save_base_path: str, metadata: dict):
    json_path = f"{save_base_path}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Metadata saved to: {json_path}")


def list_cases():
    print("[INFO] Available demo cases:")
    for case_name, case in DEMO_CASES.items():
        desc = case.get("desc", "")
        print(f"  - {case_name}: {desc}")


# =========================
# 4. Demo 推理主函数
# =========================
def run_demo(case_name: str):
    ensure_output_dir()

    case = get_case(case_name)

    prompt = case["prompt"]
    split_ratio = case["split_ratio"]
    regional_prompt = case["regional_prompt"]
    desc = case.get("desc", "")
    config = merge_config(case.get("config"))

    print("=" * 60)
    print(f"[INFO] Running case: {case_name}")
    if desc:
        print(f"[INFO] Description: {desc}")
    print(f"[INFO] Prompt: {prompt}")
    print(f"[INFO] Split ratio: {split_ratio}")
    print(f"[INFO] Config: {json.dumps(config, ensure_ascii=False)}")
    print("=" * 60)

    pipe = load_pipeline()

    image = pipe(
        prompt=regional_prompt,
        split_ratio=split_ratio,
        batch_size=config["batch_size"],
        base_ratio=config["base_ratio"],
        base_prompt=prompt,
        num_inference_steps=config["num_inference_steps"],
        height=config["height"],
        width=config["width"],
        negative_prompt=config["negative_prompt"],
        seed=config["seed"],
        guidance_scale=config["guidance_scale"],
    ).images[0]

    save_base_path = build_save_base_path(case_name)
    image_path = f"{save_base_path}.png"
    image.save(image_path)

    print(f"[INFO] Image saved to: {image_path}")

    metadata = {
        "case_name": case_name,
        "desc": desc,
        "prompt": prompt,
        "split_ratio": split_ratio,
        "regional_prompt": regional_prompt,
        "config": config,
        "model_root": MODEL_ROOT,
        "output_dir": OUTPUT_DIR,
        "image_path": image_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    save_metadata(save_base_path, metadata)


# =========================
# 5. 命令行入口
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Run demo inference by case name.")
    parser.add_argument(
        "--case",
        type=str,
        default="palace_two_maids",
        help="Case name defined in demo_examples.py"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available demo cases"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        list_cases()
    else:
        run_demo(args.case)


