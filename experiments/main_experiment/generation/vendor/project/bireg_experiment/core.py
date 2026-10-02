from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LANGUAGES = {"zh", "en"}
RUNTIME_PLAN_FIELDS = ("split_ratio", "regional_prompt", "region_count")


class ValidationError(ValueError):
    pass


def runtime_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return only fields consumed by a regional image-generation pipeline.

    Provenance fields are deliberately excluded so adding planner/model/template
    metadata cannot alter inference behavior.
    """
    _require(plan, RUNTIME_PLAN_FIELDS, "plan")
    return {name: plan[name] for name in RUNTIME_PLAN_FIELDS}


def ratio_region_count(split_ratio: str) -> int:
    """Count regions using the historical matrix.py split-ratio grammar."""
    try:
        rows = [[float(value.strip()) for value in row.split(",")] for row in split_ratio.split(";")]
    except (AttributeError, ValueError) as exc:
        raise ValidationError(f"invalid split_ratio: {split_ratio!r}") from exc
    if not rows or any(not row or any(value <= 0 for value in row) for row in rows):
        raise ValidationError(f"split_ratio values must be positive: {split_ratio!r}")
    if len(rows) == 1:
        return len(rows[0])
    return sum(max(1, len(row) - 1) for row in rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"{path}: expected a JSON object")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValidationError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _require(row: dict[str, Any], names: Iterable[str], label: str) -> None:
    missing = [name for name in names if row.get(name) in (None, "")]
    if missing:
        raise ValidationError(f"{label}: missing {', '.join(missing)}")


def validate_prompts(prompts: list[dict[str, Any]]) -> None:
    if not prompts:
        raise ValidationError("prompt set is empty")
    prompt_ids: set[str] = set()
    bilingual_pairs: set[tuple[str, str]] = set()
    for row in prompts:
        _require(row, ["prompt_id", "zh", "en", "category", "source", "split", "version"], "prompt")
        prompt_id = row["prompt_id"]
        if not ID_RE.fullmatch(prompt_id) or prompt_id in prompt_ids:
            raise ValidationError(f"invalid or duplicate prompt_id: {prompt_id}")
        if row.get("split") != "test" or row.get("frozen") is not True:
            raise ValidationError(f"{prompt_id}: main benchmark prompts must be frozen test records")
        pair = (row["zh"].strip(), row["en"].strip())
        if pair in bilingual_pairs:
            raise ValidationError(f"duplicate bilingual prompt pair: {prompt_id}")
        prompt_ids.add(prompt_id)
        bilingual_pairs.add(pair)


def validate(config: dict[str, Any], prompts: list[dict[str, Any]], plans: list[dict[str, Any]]) -> None:
    _require(config, ["experiment_id", "planning_mode", "languages", "seeds", "methods", "common_generation", "backbones", "bireg"], "config")
    if config["planning_mode"] != "cached":
        raise ValidationError("main experiment must use planning_mode='cached'")
    if not ID_RE.fullmatch(config["experiment_id"]):
        raise ValidationError("experiment_id contains unsafe path characters")
    languages = config["languages"]
    if not languages or set(languages) - LANGUAGES:
        raise ValidationError("languages must be a non-empty subset of ['zh', 'en']")
    seeds = config["seeds"]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)) or not all(isinstance(x, int) and x >= 0 for x in seeds):
        raise ValidationError("seeds must contain at least three distinct non-negative integers")
    enabled_methods = [m for m in config["methods"] if m.get("enabled")]
    if not enabled_methods:
        raise ValidationError("at least one method must be enabled")
    method_ids = [m.get("id") for m in enabled_methods]
    if any(not isinstance(x, str) or not ID_RE.fullmatch(x) for x in method_ids) or len(method_ids) != len(set(method_ids)):
        raise ValidationError("enabled method ids must be unique safe identifiers")
    backbones = config["backbones"]
    if not isinstance(backbones, dict) or not backbones:
        raise ValidationError("backbones must be a non-empty object")
    for method in enabled_methods:
        _require(method, ["id", "backend", "languages", "uses_regions"], f"method {method.get('id')}")
        if method["backend"] not in backbones:
            raise ValidationError(f"{method['id']}: unknown backend {method['backend']}")
        if not method["languages"] or set(method["languages"]) - set(languages):
            raise ValidationError(f"{method['id']}: unsupported languages")
        if "generation" in method and not isinstance(method["generation"], dict):
            raise ValidationError(f"{method['id']}: generation override must be an object")

    validate_prompts(prompts)
    prompt_ids = {row["prompt_id"] for row in prompts}

    plan_keys: set[tuple[str, str]] = set()
    for row in plans:
        _require(row, ["prompt_id", "language", "planner_id", "template_id", "split_ratio", "regional_prompt", "region_count", "status", "plan_version"], "plan")
        key = (row["prompt_id"], row["language"])
        if key in plan_keys:
            raise ValidationError(f"duplicate plan: {key}")
        if row["prompt_id"] not in prompt_ids or row["language"] not in languages:
            raise ValidationError(f"orphan or unsupported plan: {key}")
        if row["status"] != "frozen" or not isinstance(row["region_count"], int) or row["region_count"] < 1:
            raise ValidationError(f"{key}: plan must be frozen with a positive region_count")
        if len(row["regional_prompt"].split(" BREAK ")) != row["region_count"]:
            raise ValidationError(f"{key}: region_count does not match BREAK segments")
        if ratio_region_count(row["split_ratio"]) != row["region_count"]:
            raise ValidationError(f"{key}: region_count does not match split_ratio")
        plan_keys.add(key)

    planned_languages = {
        language
        for method in enabled_methods if method.get("uses_regions")
        for language in method["languages"]
    }
    expected = {(prompt_id, language) for prompt_id in prompt_ids for language in planned_languages}
    missing_plans = sorted(expected - plan_keys)
    if missing_plans:
        raise ValidationError(f"missing frozen plans: {missing_plans}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_tasks(config: dict[str, Any], prompts: list[dict[str, Any]], plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate(config, prompts, plans)
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    plan_by_key = {(row["prompt_id"], row["language"]): row for row in plans}
    tasks: list[dict[str, Any]] = []
    for method in [m for m in config["methods"] if m.get("enabled")]:
        generation = {
            **config["common_generation"],
            **config["backbones"][method["backend"]],
            **method.get("generation", {}),
        }
        for language in method["languages"]:
            for prompt_id in sorted(prompt_by_id):
                prompt = prompt_by_id[prompt_id][language]
                plan_record = plan_by_key[(prompt_id, language)] if method.get("uses_regions") else None
                for seed in config["seeds"]:
                    relative_image = Path(config.get("output_root", "outputs")) / config["experiment_id"] / method["id"] / language / prompt_id / f"seed_{seed}.png"
                    task = {
                        "schema_version": config.get("schema_version", "1.0"),
                        "experiment_id": config["experiment_id"],
                        "prompt_set_version": config.get("prompt_set_version"),
                        "plan_set_version": config.get("plan_set_version"),
                        "method_id": method["id"],
                        "backend": method["backend"],
                        "uses_regions": bool(method.get("uses_regions")),
                        "prompt_id": prompt_id,
                        "language": language,
                        "prompt": prompt,
                        "seed": seed,
                        "generation": generation,
                        "model_revision": config.get("model_revisions", {}).get(method["backend"]),
                        "bireg": config["bireg"] if method.get("uses_regions") else None,
                        "plan": runtime_plan(plan_record) if plan_record else None,
                        "plan_provenance": {
                            key: value for key, value in (plan_record or {}).items()
                            if key not in RUNTIME_PLAN_FIELDS
                        } if plan_record else None,
                        "image_path": relative_image.as_posix(),
                    }
                    task["task_id"] = stable_hash(task)[:16]
                    tasks.append(task)
    paths = [row["image_path"] for row in tasks]
    if len(paths) != len(set(paths)):
        raise ValidationError("generated image paths collide")
    return tasks


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
