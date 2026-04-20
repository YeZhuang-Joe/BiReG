import torch
from RegionalKolorsDiffusion_xl import RegionalDiffusionXLPipeline
from modeling_chatglm import ChatGLMModel
from tokenization_chatglm import ChatGLMTokenizer
from diffusers import EulerDiscreteScheduler, AutoencoderKL, UNet2DConditionModel
from demo_examples import DEMO_CASES
# =========================
# 0. Demo 固定配置
# =========================
DEMO_CONFIG = {
    "batch_size": 1,
    "base_ratio": 0.2,
    "num_inference_steps": 30,
    "height": 1024,
    "width": 1536,
    "seed": 1234,
    "guidance_scale": 4.5,
    "negative_prompt": "",
}
# =========================
# 1. 加载模型
# =========================
def load_pipeline():
    text_encoder = ChatGLMModel.from_pretrained(
        '/root/autodl-tmp/weights/Kolors/text_encoder',
        torch_dtype=torch.float16
    ).half()

    tokenizer = ChatGLMTokenizer.from_pretrained(
        '/root/autodl-tmp/weights/Kolors/text_encoder'
    )

    vae = AutoencoderKL.from_pretrained(
        '/root/autodl-tmp/weights/Kolors/vae'
    ).half()

    scheduler = EulerDiscreteScheduler.from_pretrained(
        '/root/autodl-tmp/weights/Kolors/scheduler'
    )

    unet = UNet2DConditionModel.from_pretrained(
        '/root/autodl-tmp/weights/Kolors/unet'
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
# 2. Demo 推理
# =========================
def run_demo(case_name="traveler_cat_fire"):
    case = DEMO_CASES[case_name]
    prompt = case["prompt"]
    split_ratio = case["split_ratio"]
    regional_prompt = case["regional_prompt"]
    negative_prompt = DEMO_CONFIG["negative_prompt"]
    print(f"[INFO] Running case: {case_name}")
    print(f"[INFO] Prompt: {prompt}")
    print(f"[INFO] Split ratio: {split_ratio}")

    pipe = load_pipeline()

    image = pipe(
        prompt=regional_prompt,
        split_ratio=split_ratio,
        batch_size=DEMO_CONFIG["batch_size"],
        base_ratio=DEMO_CONFIG["base_ratio"],
        base_prompt=prompt,
        num_inference_steps=DEMO_CONFIG["num_inference_steps"],
        height=DEMO_CONFIG["height"],
        width=DEMO_CONFIG["width"],
        negative_prompt=negative_prompt,
        seed=DEMO_CONFIG["seed"],
        guidance_scale=DEMO_CONFIG["guidance_scale"],
    ).images[0]

    save_path = f"{case_name}.png"
    image.save(save_path)

    print(f"[INFO] Saved: {save_path}")
# =========================
# 3. 主入口
# =========================
if __name__ == "__main__":
    run_demo()
