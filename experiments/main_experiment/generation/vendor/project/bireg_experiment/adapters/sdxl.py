from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path
from typing import Any

from .base import AdapterError, GenerationAdapter


def _dtype_from_name(torch_module: Any, name: str) -> Any:
    normalized = str(name).lower().replace("torch.", "")
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise AdapterError(f"unsupported dtype: {name!r}") from exc


def _tokenization_record(tokenizer: Any, prompt: str) -> dict[str, Any]:
    """Describe tokenizer truncation without changing the prompt."""
    encoded = tokenizer(prompt, truncation=False, add_special_tokens=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    token_ids = list(input_ids)
    max_length = int(tokenizer.model_max_length)
    was_truncated = len(token_ids) > max_length
    # Match Diffusers' SDXL warning logic: the final token is normally EOS and
    # the token at max_length - 1 is displaced by truncation.
    truncated_ids = token_ids[max_length - 1:-1] if was_truncated else []
    try:
        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()
    except Exception:
        truncated_text = ""
    return {
        "token_count": len(token_ids),
        "model_max_length": max_length,
        "was_truncated": was_truncated,
        "truncated_token_count": max(0, len(token_ids) - max_length),
        "truncated_text": truncated_text,
    }


class SDXLAdapter(GenerationAdapter):
    """Pure SDXL baseline implemented with StableDiffusionXLPipeline."""

    method_id = "sdxl"

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        use_safetensors: bool = True,
        enable_xformers: bool = True,
        variant: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.dtype_name = dtype
        self.use_safetensors = use_safetensors
        self.enable_xformers = enable_xformers
        self.variant = variant
        self.pipe: Any | None = None
        self._modules: dict[str, Any] = {}
        self.model_index_sha256: str | None = None

    def load(self) -> None:
        if self.pipe is not None:
            return
        if not self.model_path.exists():
            raise AdapterError(f"SDXL model path does not exist: {self.model_path}")
        try:
            import diffusers
            import torch
            import transformers
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise AdapterError(
                "SDXL dependencies are missing; activate the historical sdxl environment"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise AdapterError("CUDA was requested but torch.cuda.is_available() is false")

        kwargs: dict[str, Any] = {
            "torch_dtype": _dtype_from_name(torch, self.dtype_name),
            "use_safetensors": self.use_safetensors,
        }
        if self.variant:
            kwargs["variant"] = self.variant
        pipe = StableDiffusionXLPipeline.from_pretrained(str(self.model_path), **kwargs)
        pipe = pipe.to(self.device)
        if self.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                raise AdapterError(f"failed to enable xFormers attention: {exc}") from exc

        self.pipe = pipe
        self._modules = {
            "torch": torch,
            "diffusers": diffusers,
            "transformers": transformers,
        }
        model_index = self.model_path / "model_index.json"
        if model_index.exists():
            self.model_index_sha256 = hashlib.sha256(model_index.read_bytes()).hexdigest()

    def _validate_task(self, task: dict[str, Any]) -> dict[str, Any]:
        if task.get("method_id") != self.method_id:
            raise AdapterError(
                f"SDXL adapter received method_id={task.get('method_id')!r}"
            )
        if task.get("uses_regions"):
            raise AdapterError("pure SDXL baseline must not consume a regional plan")
        generation = task.get("generation")
        if not isinstance(generation, dict):
            raise AdapterError("task.generation must be an object")
        required = ("height", "width", "guidance_scale", "num_inference_steps")
        missing = [name for name in required if generation.get(name) is None]
        if missing:
            raise AdapterError(f"SDXL task is missing generation fields: {missing}")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise AdapterError("SDXL task prompt is empty")
        if not isinstance(task.get("seed"), int) or task["seed"] < 0:
            raise AdapterError("SDXL task seed must be a non-negative integer")
        task_dtype = generation.get("dtype")
        if task_dtype and str(task_dtype).lower() not in {
            self.dtype_name.lower(),
            self.dtype_name.lower().replace("float", "fp"),
        }:
            raise AdapterError(
                f"dtype mismatch: task expects {task_dtype}, adapter uses {self.dtype_name}"
            )
        return generation

    def generate(self, task: dict[str, Any], output_path: Path) -> dict[str, Any]:
        generation = self._validate_task(task)
        self.load()
        assert self.pipe is not None
        torch = self._modules["torch"]

        expected_scheduler = generation.get("scheduler")
        actual_scheduler = self.pipe.scheduler.__class__.__name__
        if expected_scheduler and expected_scheduler != actual_scheduler:
            raise AdapterError(
                f"scheduler mismatch: task expects {expected_scheduler}, model provides {actual_scheduler}"
            )
        expected_karras = generation.get("use_karras_sigmas")
        actual_karras = bool(getattr(self.pipe.scheduler.config, "use_karras_sigmas", False))
        if expected_karras is not None and bool(expected_karras) != actual_karras:
            raise AdapterError(
                f"Karras-sigma mismatch: task expects {bool(expected_karras)}, "
                f"model provides {actual_karras}"
            )

        tokenizers: dict[str, Any] = {"tokenizer": self.pipe.tokenizer}
        if getattr(self.pipe, "tokenizer_2", None) is not None:
            tokenizers["tokenizer_2"] = self.pipe.tokenizer_2
        tokenization = {
            name: _tokenization_record(tokenizer, task["prompt"])
            for name, tokenizer in tokenizers.items()
        }

        generator = torch.Generator(device=self.device).manual_seed(task["seed"])
        call_kwargs: dict[str, Any] = {
            "prompt": task["prompt"],
            "height": int(generation["height"]),
            "width": int(generation["width"]),
            "guidance_scale": float(generation["guidance_scale"]),
            "num_inference_steps": int(generation["num_inference_steps"]),
            "generator": generator,
        }
        if "negative_prompt" in generation and generation["negative_prompt"] not in (None, ""):
            call_kwargs["negative_prompt"] = generation["negative_prompt"]

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        image = self.pipe(**call_kwargs).images[0]
        elapsed = time.perf_counter() - started

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
        image.save(temporary_path)
        temporary_path.replace(output_path)
        image_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

        scheduler = self.pipe.scheduler
        scheduler_config = dict(getattr(scheduler, "config", {}) or {})
        return {
            "adapter": "SDXLAdapter",
            "model_path": str(self.model_path),
            "model_index_sha256": self.model_index_sha256,
            "device": self.device,
            "dtype": self.dtype_name,
            "use_safetensors": self.use_safetensors,
            "xformers_enabled": self.enable_xformers,
            "scheduler_class": scheduler.__class__.__name__,
            "scheduler_config": scheduler_config,
            "tokenization": tokenization,
            "generation_seconds": elapsed,
            "image_sha256": image_sha256,
            "image_bytes": output_path.stat().st_size,
            "image_size": list(image.size),
            "cuda_max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if self.device.startswith("cuda") else None
            ),
            "software": {
                "python": platform.python_version(),
                "torch": self._modules["torch"].__version__,
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
