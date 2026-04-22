import os
import json
import argparse
from datetime import datetime

import torch

from RegionalKolorsDiffusion_xl import RegionalDiffusionXLPipeline
from modeling_chatglm import ChatGLMModel
from tokenization_chatglm import ChatGLMTokenizer
from diffusers import EulerDiscreteScheduler, AutoencoderKL, UNet2DConditionModel

from planner import detect_language, build_para_dict, validate_para_dict


# =========================
# 0. 全流程默认配置
# =========================
MODEL_ROOT = os.getenv("KOLORS_PATH", "weights/Kolors")
OUTPUT_DIR = "outputs/full"

FULL_CONFIG = {
    "batch_size": 1,
    "base_ratio": 0.2,
    "num_inference_steps": 30,
    "height": 1024,
    "width": 1536,
    "guidance_scale": 4.5,
    "seed": 1234,
    "negative_prompt": "",
}


# =========================
# 1. 命令行参数
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="BiReG Full Pipeline Inference")

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User input prompt"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="auto",
        choices=["auto", "zh", "en"],
        help="Prompt language mode"
    )

    parser.add_argument(
        "--planner",
        type=str,
        default="gpt",
        help="Planner backend name, e.g. gpt / deepseek / mock"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path"
    )

    return parser.parse_args()


# =========================
# 2. 检查模型路径
# =========================
def check_model_paths():
    required_paths = {
        "text_encoder": f"{MODEL_ROOT}/text_encoder",
        "vae": f"{MODEL_ROOT}/vae",
        "scheduler": f"{MODEL_ROOT}/scheduler",
        "unet": f"{MODEL_ROOT}/unet",
    }

    missing = [name for name, path in required_paths.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing required model components under '{MODEL_ROOT}': {missing}"
        )


# =========================
# 3. 加载模型
# =========================
def load_pipeline():
    check_model_paths()

    text_encoder = ChatGLMModel.from_pretrained(
        f"{MODEL_ROOT}/text_encoder",
        torch_dtype=torch.float16
    ).half()

    tokenizer = ChatGLMTokenizer.from_pretrained(
        f"{MODEL_ROOT}/text_encoder"
    )

    vae = AutoencoderKL.from_pretrained(
        f"{MODEL_ROOT}/vae"
    ).half()

    scheduler = EulerDiscreteScheduler.from_pretrained(
        f"{MODEL_ROOT}/scheduler"
    )

    unet = UNet2DConditionModel.from_pretrained(
        f"{MODEL_ROOT}/unet"
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

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"[WARN] xformers is not enabled: {e}")

    return pipe


# =========================
# 4. 保存运行中间信息
# =========================
def save_run_metadata(
    output_image_path: str,
    prompt: str,
    lang: str,
    planner_name: str,
    para_dict: dict,
    config: dict
) -> str:
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    metadata = {
        "prompt": prompt,
        "lang": lang,
        "planner": planner_name,
        "para_dict": para_dict,
        "config": config,
        "output_image": output_image_path,
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

    json_path = os.path.splitext(output_image_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return json_path


# =========================
# 5. 主推理流程
# =========================
def run_full(prompt: str, lang: str = "auto", planner_name: str = "gpt", output: str = None):
    # 1) 语言选择
    if lang == "auto":
        lang = detect_language(prompt)

    print(f"[INFO] Prompt: {prompt}")
    print(f"[INFO] Language: {lang}")
    print(f"[INFO] Planner: {planner_name}")

    # 2) planner 生成 para_dict
    para_dict = build_para_dict(
        prompt=prompt,
        lang=lang,
        planner_name=planner_name,
    )

    # 3) 校验 para_dict
    validate_para_dict(para_dict)

    split_ratio = para_dict["Final split ratio"]
    regional_prompt = para_dict["Regional Prompt"]
    negative_prompt = FULL_CONFIG["negative_prompt"]

    print(f"[INFO] Final split ratio: {split_ratio}")
    print(f"[INFO] Regional prompt: {regional_prompt}")

    # 4) 加载 pipeline
    pipe = load_pipeline()

    # 5) 执行推理
    image = pipe(
        prompt=regional_prompt,
        split_ratio=split_ratio,
        batch_size=FULL_CONFIG["batch_size"],
        base_ratio=FULL_CONFIG["base_ratio"],
        base_prompt=prompt,
        num_inference_steps=FULL_CONFIG["num_inference_steps"],
        height=FULL_CONFIG["height"],
        width=FULL_CONFIG["width"],
        negative_prompt=negative_prompt,
        seed=FULL_CONFIG["seed"],
        guidance_scale=FULL_CONFIG["guidance_scale"],
    ).images[0]

    # 6) 输出路径
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if output is None:
        file_name = datetime.now().strftime("full_%Y%m%d_%H%M%S.png")
        output = os.path.join(OUTPUT_DIR, file_name)

    # 7) 保存图片
    image.save(output)

    # 8) 保存本次运行元信息
    json_path = save_run_metadata(
        output_image_path=output,
        prompt=prompt,
        lang=lang,
        planner_name=planner_name,
        para_dict=para_dict,
        config=FULL_CONFIG
    )

    print(f"[INFO] Image saved to: {output}")
    print(f"[INFO] Metadata saved to: {json_path}")


# =========================
# 6. 入口
# =========================
if __name__ == "__main__":
    args = parse_args()
    run_full(
        prompt=args.prompt,
        lang=args.lang,
        planner_name=args.planner,
        output=args.output
    )
