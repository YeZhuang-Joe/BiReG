from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path
from typing import Any

from .base import AdapterError, GenerationAdapter
from .sdxl import _dtype_from_name


def _normalized_negative_prompt(value: Any) -> str:
    """Avoid the pinned Kolors pipeline's negative pooled-embedding None bug."""
    return value if isinstance(value, str) and value else ""


def _kolors_tokenization_record(
    tokenizer: Any, prompt: str, max_sequence_length: int
) -> dict[str, Any]:
    """Record the explicit Kolors limit; ChatGLM exposes a sentinel model_max_length."""
    encoded = tokenizer(prompt, truncation=False, add_special_tokens=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    count = len(input_ids)
    return {
        "token_count": count,
        "max_sequence_length": max_sequence_length,
        "was_truncated": count > max_sequence_length,
        "truncated_token_count": max(0, count - max_sequence_length),
    }


class KolorsAdapter(GenerationAdapter):
    """Pure Kolors baseline using Diffusers' native KolorsPipeline."""

    method_id = "kolors"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        enable_xformers: bool = True,
        max_sequence_length: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.dtype_name = dtype
        self.enable_xformers = enable_xformers
        self.max_sequence_length = max_sequence_length
        self.pipe: Any | None = None
        self._modules: dict[str, Any] = {}
        self.model_config_sha256: dict[str, str] = {}

    def load(self) -> None:
        if self.pipe is not None:
            return
        if not self.model_path.exists():
            raise AdapterError(f"Kolors model path does not exist: {self.model_path}")
        try:
            import diffusers
            import torch
            import transformers
            from diffusers import KolorsPipeline
        except ImportError as exc:
            raise AdapterError("Kolors dependencies are missing; activate the RPG environment") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise AdapterError("CUDA was requested but torch.cuda.is_available() is false")

        # Do not pass variant='fp16': the pinned Diffusers development snapshot
        # has a variant scanner bug on extensionless files in the Kolors root.
        pipe = KolorsPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=_dtype_from_name(torch, self.dtype_name),
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).to(self.device)
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
        missing = [name for name, path in tracked.items() if not path.is_file()]
        if missing:
            raise AdapterError(f"Kolors model is missing tracked configuration files: {missing}")
        self.model_config_sha256 = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in tracked.items()
        }
        self.pipe = pipe
        self._modules = {"torch": torch, "diffusers": diffusers, "transformers": transformers}

    def _validate_task(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("method_id") != self.method_id:
            raise AdapterError(f"Kolors adapter received method_id={task.get('method_id')!r}")
        if task.get("uses_regions"):
            raise AdapterError("pure Kolors baseline must not consume a regional plan")
        if task.get("language") not in {"zh", "en"}:
            raise AdapterError("Kolors baseline language must be 'zh' or 'en'")
        generation = task.get("generation")
        if not isinstance(generation, dict):
            raise AdapterError("task.generation must be an object")
        required = ("height", "width", "guidance_scale", "num_inference_steps")
        missing = [name for name in required if generation.get(name) is None]
        if missing:
            raise AdapterError(f"Kolors task is missing generation fields: {missing}")
        if generation.get("batch_size", 1) != 1:
            raise AdapterError("Kolors reproducibility runner supports batch_size=1 only")
        if generation.get("max_sequence_length", self.max_sequence_length) != self.max_sequence_length:
            raise AdapterError("Kolors max_sequence_length does not match the adapter")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise AdapterError("Kolors task prompt is empty")
        if not isinstance(task.get("seed"), int) or task["seed"] < 0:
            raise AdapterError("Kolors task seed must be a non-negative integer")
        task_dtype = generation.get("dtype")
        accepted = {self.dtype_name.lower(), self.dtype_name.lower().replace("float", "fp")}
        if task_dtype and str(task_dtype).lower() not in accepted:
            raise AdapterError(
                f"dtype mismatch: task expects {task_dtype}, adapter uses {self.dtype_name}"
            )
        return generation

    def generate(self, task: dict[str, Any], output_path: Path) -> dict[str, Any]:
        generation = self._validate_task(task)
        self.load()
        assert self.pipe is not None
        torch = self._modules["torch"]

        actual_scheduler = self.pipe.scheduler.__class__.__name__
        expected_scheduler = generation.get("scheduler")
        if expected_scheduler and expected_scheduler != actual_scheduler:
            raise AdapterError(
                f"scheduler mismatch: task expects {expected_scheduler}, model provides {actual_scheduler}"
            )
        actual_karras = bool(getattr(self.pipe.scheduler.config, "use_karras_sigmas", False))
        expected_karras = generation.get("use_karras_sigmas")
        if expected_karras is not None and bool(expected_karras) != actual_karras:
            raise AdapterError(
                f"Karras-sigma mismatch: task expects {bool(expected_karras)}, model provides {actual_karras}"
            )

        tokenization = _kolors_tokenization_record(
            self.pipe.tokenizer, task["prompt"], self.max_sequence_length
        )
        generator = torch.Generator(device=self.device).manual_seed(task["seed"])
        call_kwargs: dict[str, Any] = {
            "prompt": task["prompt"],
            # The pinned Diffusers 0.33 development snapshot leaves
            # negative_pooled_prompt_embeds=None when negative_prompt is None
            # and force_zeros_for_empty_prompt is enabled, then calls repeat()
            # on it. Historical Kolors scripts explicitly passed "".
            "negative_prompt": _normalized_negative_prompt(
                generation.get("negative_prompt")
            ),
            "height": int(generation["height"]),
            "width": int(generation["width"]),
            "guidance_scale": float(generation["guidance_scale"]),
            "num_inference_steps": int(generation["num_inference_steps"]),
            "num_images_per_prompt": 1,
            "max_sequence_length": self.max_sequence_length,
            "generator": generator,
        }
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = self.pipe(**call_kwargs).images[0]
        elapsed = time.perf_counter() - started

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
        image.save(temporary_path)
        temporary_path.replace(output_path)
        return {
            "adapter": "KolorsAdapter",
            "model_path": str(self.model_path),
            "model_config_sha256": self.model_config_sha256,
            "device": self.device,
            "dtype": self.dtype_name,
            "xformers_enabled": self.enable_xformers,
            "max_sequence_length": self.max_sequence_length,
            "scheduler_class": self.pipe.scheduler.__class__.__name__,
            "scheduler_config": dict(getattr(self.pipe.scheduler, "config", {}) or {}),
            "tokenization": tokenization,
            "generation_seconds": elapsed,
            "image_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "image_bytes": output_path.stat().st_size,
            "image_size": list(image.size),
            "cuda_max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None
            ),
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "diffusers": self._modules["diffusers"].__version__,
                "transformers": self._modules["transformers"].__version__,
            },
        }

    def close(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        torch = self._modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
