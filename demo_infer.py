import torch

from RegionalKolorsDiffusion_xl import RegionalDiffusionXLPipeline
from modeling_chatglm import ChatGLMModel
from tokenization_chatglm import ChatGLMTokenizer
from diffusers import EulerDiscreteScheduler, AutoencoderKL, UNet2DConditionModel

from demo_examples import DEMO_CASES


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

    print(f"[INFO] Running case: {case_name}")
    print(f"[INFO] Prompt: {prompt}")

    pipe = load_pipeline()

    image = pipe(
        prompt=regional_prompt,
        base_prompt=prompt,
        num_inference_steps=50,
        height=1024,
        width=1024,
        split_ratio=split_ratio
    ).images[0]

    image.save(f"{case_name}.png")

    print(f"[INFO] Saved: {case_name}.png")


# =========================
# 3. 主入口
# =========================
if __name__ == "__main__":
    run_demo()
