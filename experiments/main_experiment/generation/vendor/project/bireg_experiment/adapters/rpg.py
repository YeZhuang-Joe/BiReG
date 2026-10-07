from __future__ import annotations

import hashlib
import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Any

from ..core import ratio_region_count
from .base import AdapterError, GenerationAdapter
from .sdxl import _dtype_from_name, _tokenization_record


class RPGAdapter(GenerationAdapter):
    """Cached-plan RPG generation on the same SDXL backbone as the baseline."""

    method_id = "rpg"
    _source_files = ("RegionalDiffusion_xl.py", "cross_attention.py", "matrix.py")

    def __init__(
        self,
        model_path: str | Path,
        rpg_source_path: str | Path,
        *,
        device: str = "cuda",
        dtype: str = "float16",
        use_safetensors: bool = True,
        enable_xformers: bool = True,
        variant: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.rpg_source_path = Path(rpg_source_path)
        self.device = device
        self.dtype_name = dtype
        self.use_safetensors = use_safetensors
        self.enable_xformers = enable_xformers
        self.variant = variant
        self.pipe: Any | None = None
        self._modules: dict[str, Any] = {}
        self.model_index_sha256: str | None = None
        self.source_sha256: dict[str, str] = {}

    def load(self) -> None:
        if self.pipe is not None:
            return
        if not self.model_path.exists():
            raise AdapterError(f"SDXL model path does not exist: {self.model_path}")
        missing = [name for name in self._source_files if not (self.rpg_source_path / name).is_file()]
        if missing:
            raise AdapterError(f"RPG source path is missing required files: {missing}")
        try:
            import diffusers
            import torch
            import transformers
        except ImportError as exc:
            raise AdapterError("RPG dependencies are missing; activate the RPG environment") from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise AdapterError("CUDA was requested but torch.cuda.is_available() is false")

        source_text = str(self.rpg_source_path.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            module = importlib.import_module("RegionalDiffusion_xl")
            pipeline_class = module.RegionalDiffusionXLPipeline
        except Exception as exc:
            raise AdapterError(f"failed to import sanitized RPG pipeline: {exc}") from exc

        kwargs: dict[str, Any] = {
            "torch_dtype": _dtype_from_name(torch, self.dtype_name),
            "use_safetensors": self.use_safetensors,
            "local_files_only": True,
        }
        if self.variant:
            kwargs["variant"] = self.variant
        pipe = pipeline_class.from_pretrained(str(self.model_path), **kwargs).to(self.device)
        if self.enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                raise AdapterError(f"failed to enable xFormers attention: {exc}") from exc

        self.pipe = pipe
        self._modules = {"torch": torch, "diffusers": diffusers, "transformers": transformers}
        model_index = self.model_path / "model_index.json"
        if model_index.is_file():
            self.model_index_sha256 = hashlib.sha256(model_index.read_bytes()).hexdigest()
        self.source_sha256 = {
            name: hashlib.sha256((self.rpg_source_path / name).read_bytes()).hexdigest()
            for name in self._source_files
        }

    def _validate_task(self, task: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
        if task.get("method_id") != self.method_id:
            raise AdapterError(f"RPG adapter received method_id={task.get('method_id')!r}")
        if task.get("uses_regions") is not True:
            raise AdapterError("RPG tasks must consume a frozen regional plan")
        generation = task.get("generation")
        plan = task.get("plan")
        bireg = task.get("bireg")
        if not isinstance(generation, dict) or not isinstance(plan, dict) or not isinstance(bireg, dict):
            raise AdapterError("RPG task requires generation, plan, and bireg objects")
        required = ("height", "width", "guidance_scale", "num_inference_steps")
        missing = [name for name in required if generation.get(name) is None]
        if missing:
            raise AdapterError(f"RPG task is missing generation fields: {missing}")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise AdapterError("RPG base prompt is empty")
        if not isinstance(task.get("seed"), int) or task["seed"] < 0:
            raise AdapterError("RPG task seed must be a non-negative integer")
        if generation.get("batch_size", 1) != 1:
            raise AdapterError("RPG reproducibility runner supports batch_size=1 only")
        task_dtype = generation.get("dtype")
        accepted_dtypes = {
            self.dtype_name.lower(),
            self.dtype_name.lower().replace("float", "fp"),
        }
        if task_dtype and str(task_dtype).lower() not in accepted_dtypes:
            raise AdapterError(
                f"dtype mismatch: task expects {task_dtype}, adapter uses {self.dtype_name}"
            )
        for name in ("split_ratio", "regional_prompt", "region_count"):
            if plan.get(name) in (None, ""):
                raise AdapterError(f"RPG plan is missing {name}")
        segments = [value.strip() for value in plan["regional_prompt"].split(" BREAK ")]
        if any(not value for value in segments) or len(segments) != plan["region_count"]:
            raise AdapterError("RPG regional prompt segments do not match region_count")
        if ratio_region_count(plan["split_ratio"]) != plan["region_count"]:
            raise AdapterError("RPG split_ratio does not match region_count")
        base_ratio = bireg.get("base_ratio")
        if not isinstance(base_ratio, (int, float)) or not 0 < float(base_ratio) <= 1:
            raise AdapterError("bireg.base_ratio must be in (0, 1]")
        return generation, plan, float(base_ratio)

    def generate(self, task: dict[str, Any], output_path: Path) -> dict[str, Any]:
        generation, plan, base_ratio = self._validate_task(task)
        self.load()
        assert self.pipe is not None
        torch = self._modules["torch"]

        actual_scheduler = self.pipe.scheduler.__class__.__name__
        expected_scheduler = generation.get("scheduler")
        if expected_scheduler and expected_scheduler != actual_scheduler:
            raise AdapterError(f"scheduler mismatch: task expects {expected_scheduler}, model provides {actual_scheduler}")
        actual_karras = bool(getattr(self.pipe.scheduler.config, "use_karras_sigmas", False))
        expected_karras = generation.get("use_karras_sigmas")
        if expected_karras is not None and bool(expected_karras) != actual_karras:
            raise AdapterError(
                f"Karras-sigma mismatch: task expects {bool(expected_karras)}, model provides {actual_karras}"
            )

        segments = [value.strip() for value in plan["regional_prompt"].split(" BREAK ")]
        tokenizers = {"tokenizer": self.pipe.tokenizer}
        if getattr(self.pipe, "tokenizer_2", None) is not None:
            tokenizers["tokenizer_2"] = self.pipe.tokenizer_2
        tokenization = {
            name: {
                "base_prompt": _tokenization_record(tokenizer, task["prompt"]),
                "regional_segments": [_tokenization_record(tokenizer, segment) for segment in segments],
            }
            for name, tokenizer in tokenizers.items()
        }

        generator = torch.Generator(device=self.device).manual_seed(task["seed"])
        call_kwargs: dict[str, Any] = {
            "prompt": plan["regional_prompt"],
            "split_ratio": plan["split_ratio"],
            "base_prompt": task["prompt"],
            "base_ratio": base_ratio,
            "batch_size": 1,
            "height": int(generation["height"]),
            "width": int(generation["width"]),
            "guidance_scale": float(generation["guidance_scale"]),
            "num_inference_steps": int(generation["num_inference_steps"]),
            "seed": task["seed"],
            "generator": generator,
        }
        if generation.get("negative_prompt") not in (None, ""):
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
        scheduler_config = dict(getattr(self.pipe.scheduler, "config", {}) or {})
        return {
            "adapter": "RPGAdapter",
            "model_path": str(self.model_path),
            "model_index_sha256": self.model_index_sha256,
            "rpg_source_path": str(self.rpg_source_path),
            "rpg_source_sha256": self.source_sha256,
            "device": self.device,
            "dtype": self.dtype_name,
            "use_safetensors": self.use_safetensors,
            "xformers_enabled": self.enable_xformers,
            "base_ratio": base_ratio,
            "region_count": plan["region_count"],
            "scheduler_class": self.pipe.scheduler.__class__.__name__,
            "scheduler_config": scheduler_config,
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
