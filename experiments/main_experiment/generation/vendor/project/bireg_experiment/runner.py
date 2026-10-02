from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .adapters.base import GenerationAdapter
from .core import stable_hash


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sidecar_path(image_path: Path) -> Path:
    return image_path.with_suffix(".json")


def resolve_image_path(task: dict[str, Any], workspace_root: Path) -> Path:
    raw = Path(task["image_path"])
    return raw if raw.is_absolute() else workspace_root / raw


def select_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    method_id: str,
    task_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row for row in tasks
        if row.get("method_id") == method_id
        and (task_ids is None or row.get("task_id") in task_ids)
    ]
    return selected[:limit] if limit is not None else selected


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    temporary.replace(path)


def run_tasks(
    adapter: GenerationAdapter,
    tasks: Iterable[dict[str, Any]],
    *,
    workspace_root: Path,
    overwrite: bool = False,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    loaded = False
    try:
        for index, task in enumerate(tasks, 1):
            image_path = resolve_image_path(task, workspace_root)
            metadata_path = sidecar_path(image_path)
            if image_path.exists() and metadata_path.exists() and not overwrite:
                records.append({
                    "task_id": task.get("task_id"),
                    "status": "skipped_existing",
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                })
                print(f"[{index}] SKIP {task.get('task_id')} -> {image_path}", flush=True)
                continue

            started_at = utc_now()
            wall_start = time.perf_counter()
            print(
                f"[{index}] RUN {task.get('task_id')} method={task.get('method_id')} "
                f"prompt={task.get('prompt_id')} seed={task.get('seed')}",
                flush=True,
            )
            try:
                if not loaded:
                    adapter.load()
                    loaded = True
                method_metadata = adapter.generate(task, image_path)
                record = {
                    "schema_version": "1.0",
                    "status": "completed",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "wall_seconds": time.perf_counter() - wall_start,
                    "hostname": socket.gethostname(),
                    "task": task,
                    "task_hash": stable_hash(task),
                    "image_path": str(image_path),
                    "method_metadata": method_metadata,
                }
                _atomic_json(metadata_path, record)
                records.append({
                    "task_id": task.get("task_id"),
                    "status": "completed",
                    "image_path": str(image_path),
                    "metadata_path": str(metadata_path),
                })
                print(f"[{index}] DONE {task.get('task_id')} -> {image_path}", flush=True)
            except Exception as exc:
                failure = {
                    "schema_version": "1.0",
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "wall_seconds": time.perf_counter() - wall_start,
                    "hostname": socket.gethostname(),
                    "task": task,
                    "task_hash": stable_hash(task),
                    "image_path": str(image_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failure_path = metadata_path.with_name(metadata_path.stem + ".failed.json")
                _atomic_json(failure_path, failure)
                records.append({
                    "task_id": task.get("task_id"),
                    "status": "failed",
                    "failure_path": str(failure_path),
                    "error": str(exc),
                })
                print(f"[{index}] FAILED {task.get('task_id')}: {exc}", flush=True)
                if fail_fast:
                    raise
    finally:
        if loaded:
            adapter.close()
    return records
