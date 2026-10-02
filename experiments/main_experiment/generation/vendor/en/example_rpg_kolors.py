from __future__ import annotations

import hashlib
import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Any

from bireg_experiment.core import ratio_region_count
from bireg_experiment.adapters.base import AdapterError, GenerationAdapter
from example_kolors import _kolors_tokenization_record, _normalized_negative_prompt
from example_sdxl import _dtype_from_name


class RPGKolorsAdapter(GenerationAdapter):
    """Cached-plan RPG generation on the Kolors backbone."""

    method_id = "rpg_kolors"
    _source_files = (
        "RegionalKolorsDiffusion_xl.py",
        "cross_attention1.py",
        "matrix.py",
        "modeling_chatglm.py",
        "tokenization_chatglm.py",
        "configuration_chatglm.py",
    )

    def __init__(self, model_path: str | Path, rpg_source_path: str | Path, *,
                 device: str = "cuda", dtype: str = "float16",
                 enable_xformers: bool = True, enable_cpu_offload: bool = True,
                 max_sequence_length: int = 256) -> None:
        self.model_path = Path(model_path)
        self.rpg_source_path = Path(rpg_source_path)
        self.device = device
        self.dtype_name = dtype
        self.enable_xformers = enable_xformers
        self.enable_cpu_offload = enable_cpu_offload
        self.max_sequence_length = max_sequence_length
        self.pipe: Any | None = None
        self._modules: dict[str, Any] = {}
        self.model_config_sha256: dict[str, str] = {}
        self.source_sha256: dict[str, str] = {}

    def load(self) -> None:
        if self.pipe is not None:
            return
        missing = [n for n in self._source_files if not (self.rpg_source_path / n).is_file()]
        if missing:
            raise AdapterError(f"RPG-Kolors source path is missing required files: {missing}")
        if not self.model_path.exists():
            raise AdapterError(f"Kolors model path does not exist: {self.model_path}")
        try:
            import diffusers
            import torch
            import transformers
            from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, UNet2DConditionModel
        except ImportError as exc:
            raise AdapterError("RPG-Kolors dependencies are missing; activate the RPG environment") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise AdapterError("CUDA was requested but torch.cuda.is_available() is false")

        source_text = str(self.rpg_source_path.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            pipeline_class = importlib.import_module(
                "RegionalKolorsDiffusion_xl"
            ).RegionalDiffusionXLPipeline
            text_encoder_class = importlib.import_module("modeling_chatglm").ChatGLMModel
            tokenizer_class = importlib.import_module("tokenization_chatglm").ChatGLMTokenizer
        except Exception as exc:
            raise AdapterError(f"failed to import sanitized RPG-Kolors modules: {exc}") from exc

        dtype = _dtype_from_name(torch, self.dtype_name)
        common = {"local_files_only": True}
        text_encoder = text_encoder_class.from_pretrained(
            str(self.model_path / "text_encoder"), torch_dtype=dtype, **common
        )
        tokenizer = tokenizer_class.from_pretrained(
            str(self.model_path / "text_encoder"), **common
        )
        vae = AutoencoderKL.from_pretrained(
            str(self.model_path / "vae"), torch_dtype=dtype, **common
        )
        scheduler = DPMSolverMultistepScheduler.from_config(
            __import__("json").loads((self.model_path / "scheduler/scheduler_config.json").read_text()),
            use_karras_sigmas=True,
        )
        unet = UNet2DConditionModel.from_pretrained(
            str(self.model_path / "unet"), torch_dtype=dtype, **common
        )
        pipe = pipeline_class(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, force_zeros_for_empty_prompt=False,
        )
        if self.enable_cpu_offload and self.device.startswith("cuda"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        if self.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                raise AdapterError(f"failed to enable xFormers attention: {exc}") from exc

        tracked = {
            "model_index.json": self.model_path / "model_index.json",
            "scheduler/scheduler_config.json": self.model_path / "scheduler/scheduler_config.json",
            "text_encoder/config.json": self.model_path / "text_encoder/config.json",
            "text_encoder/pytorch_model.bin.index.json": self.model_path / "text_encoder/pytorch_model.bin.index.json",
            "tokenizer/tokenizer.model": self.model_path / "tokenizer/tokenizer.model",
            "unet/config.json": self.model_path / "unet/config.json",
            "vae/config.json": self.model_path / "vae/config.json",
        }
        absent = [n for n, p in tracked.items() if not p.is_file()]
        if absent:
            raise AdapterError(f"Kolors model is missing tracked configuration files: {absent}")
        self.model_config_sha256 = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in tracked.items()}
        self.source_sha256 = {n: hashlib.sha256((self.rpg_source_path / n).read_bytes()).hexdigest() for n in self._source_files}
        self.pipe = pipe
        self._modules = {"torch": torch, "diffusers": diffusers, "transformers": transformers}

    def _validate_task(self, task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
        if task.get("method_id") != self.method_id or task.get("uses_regions") is not True:
            raise AdapterError("RPG-Kolors adapter requires regional rpg_kolors tasks")
        if task.get("language") not in {"en", "zh"}:
            raise AdapterError("RPG-Kolors language must be 'en' or 'zh'")
        generation, plan, bireg = task.get("generation"), task.get("plan"), task.get("bireg")
        if not all(isinstance(x, dict) for x in (generation, plan, bireg)):
            raise AdapterError("RPG-Kolors task requires generation, plan, and bireg objects")
        required = ("height", "width", "guidance_scale", "num_inference_steps")
        missing = [n for n in required if generation.get(n) is None]
        if missing:
            raise AdapterError(f"RPG-Kolors task is missing generation fields: {missing}")
        if generation.get("batch_size", 1) != 1:
            raise AdapterError("RPG-Kolors runner supports batch_size=1 only")
        if generation.get("max_sequence_length", self.max_sequence_length) != self.max_sequence_length:
            raise AdapterError("RPG-Kolors max_sequence_length does not match the adapter")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise AdapterError("RPG-Kolors base prompt is empty")
        if not isinstance(task.get("seed"), int) or task["seed"] < 0:
            raise AdapterError("RPG-Kolors seed must be a non-negative integer")
        for name in ("split_ratio", "regional_prompt", "region_count"):
            if plan.get(name) in (None, ""):
                raise AdapterError(f"RPG-Kolors plan is missing {name}")
        from plan_check import check_plan
        native=check_plan('Final split ratio: '+plan['split_ratio']+'\nRegional Prompt: '+plan['regional_prompt'])
        if native['region_count']!=plan['region_count']:
            raise AdapterError('Frozen region count differs from native execution')
        base_ratio = bireg.get("base_ratio")
        if not isinstance(base_ratio, (int, float)) or not 0 < float(base_ratio) <= 1:
            raise AdapterError("bireg.base_ratio must be in (0, 1]")
        return generation, plan, float(base_ratio)

    def generate(self, task: dict[str, Any], output_path: Path) -> dict[str, Any]:
        generation, plan, base_ratio = self._validate_task(task)
        self.load()
        assert self.pipe is not None
        torch = self._modules["torch"]
        expected_scheduler = generation.get("scheduler")
        if expected_scheduler and expected_scheduler != self.pipe.scheduler.__class__.__name__:
            raise AdapterError("RPG-Kolors scheduler mismatch")
        actual_karras = bool(getattr(self.pipe.scheduler.config, "use_karras_sigmas", False))
        if generation.get("use_karras_sigmas") is not None and bool(generation["use_karras_sigmas"]) != actual_karras:
            raise AdapterError("RPG-Kolors Karras-sigma mismatch")

        segments = [x.strip() for x in plan["regional_prompt"].split(" BREAK ")]
        tokenization = {
            "max_sequence_length": self.max_sequence_length,
            "base_prompt": _kolors_tokenization_record(self.pipe.tokenizer, task["prompt"], self.max_sequence_length),
            "regional_segments": [_kolors_tokenization_record(self.pipe.tokenizer, x, self.max_sequence_length) for x in segments],
        }
        generator_device = "cuda" if self.device.startswith("cuda") else self.device
        generator = torch.Generator(device=generator_device).manual_seed(task["seed"])
        call_kwargs = {
            "prompt": plan["regional_prompt"], "split_ratio": plan["split_ratio"],
            "base_prompt": task["prompt"], "base_ratio": base_ratio, "batch_size": 1,
            "height": int(generation["height"]), "width": int(generation["width"]),
            "guidance_scale": float(generation["guidance_scale"]),
            "num_inference_steps": int(generation["num_inference_steps"]),
            "negative_prompt": _normalized_negative_prompt(generation.get("negative_prompt")),
            "seed": task["seed"], "generator": generator,
        }
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = self.pipe(**call_kwargs).images[0]
        elapsed = time.perf_counter() - started
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
        image.save(temporary)
        temporary.replace(output_path)
        return {
            "adapter": "RPGKolorsAdapter", "model_path": str(self.model_path),
            "model_config_sha256": self.model_config_sha256,
            "rpg_source_path": str(self.rpg_source_path), "rpg_source_sha256": self.source_sha256,
            "device": self.device, "dtype": self.dtype_name,
            "xformers_enabled": self.enable_xformers, "cpu_offload_enabled": self.enable_cpu_offload,
            "max_sequence_length": self.max_sequence_length, "base_ratio": base_ratio,
            "region_count": plan["region_count"], "scheduler_class": self.pipe.scheduler.__class__.__name__,
            "scheduler_config": dict(getattr(self.pipe.scheduler, "config", {}) or {}),
            "tokenization": tokenization, "generation_seconds": elapsed,
            "image_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "image_bytes": output_path.stat().st_size, "image_size": list(image.size),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None,
            "software": {"python": platform.python_version(), "torch": torch.__version__,
                         "diffusers": self._modules["diffusers"].__version__,
                         "transformers": self._modules["transformers"].__version__},
        }

    def close(self) -> None:
        self.pipe = None
        torch = self._modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
