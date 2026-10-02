#!/usr/bin/env python3
"""Check or score unified main-experiment generation outputs; no generation/planning."""
import argparse
import fcntl
import json
from pathlib import Path
from common import ROOT, METHODS, SEEDS, read, sha, need, freeze, digest, package_check, inputs, select, validate_image, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['check', 'run', 'aggregate'])
    p.add_argument('--language', choices=['en', 'zh'], required=True)
    p.add_argument('--method', choices=[*METHODS, 'all'], default='all')
    p.add_argument('--local-config', type=Path, default=ROOT / 'configs/local.json')
    p.add_argument('--generation-dir', type=Path)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs')
    p.add_argument('--prompt-id', action='append')
    p.add_argument('--seed', type=int)
    p.add_argument('--limit', type=int)
    p.add_argument('--check-environment', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    a = p.parse_args()
    need(a.limit is None or a.limit > 0, '--limit must be positive')
    need(a.seed is None or a.seed in SEEDS[a.language], 'Seed outside frozen experiment')
    need(a.command != 'run' or a.execute, 'Actual scoring requires run --execute')
    package_check()
    config = read(a.local_config) if a.local_config.is_file() else {}
    for key in ('generation_dir', 'image_root', 'english_repo', 'chinese_repo', 'chinese_model_dir'):
        if key in config:
            value = Path(config[key]).expanduser()
            config[key] = str((a.local_config.resolve().parent / value).resolve() if not value.is_absolute() else value.resolve())
    generation = (a.generation_dir or Path(config.get('generation_dir', ROOT.parent / 'generation'))).resolve()
    prompts, cfg, tasks = inputs(generation, a.language)
    byid = {r['prompt_id']:r for r in prompts}
    need(not a.prompt_id or set(a.prompt_id) <= set(byid), 'Unknown prompt ID')
    methods = METHODS if a.method == 'all' else [a.method]
    batches = {m:select(tasks[m], a.prompt_id, a.seed, a.limit) for m in methods}
    for m, batch in batches.items():
        print('PASS INPUT', a.language, m, len(batch), 'selected of', len(tasks[m]), flush=True)
    image_root = a.image_root or (Path(config['image_root']) if config.get('image_root') else None)
    if image_root is None:
        need(a.command == 'check', 'Set --image-root or image_root in local config')
        records = None
    else:
        image_root = image_root.resolve()
        records = []
        for m, batch in batches.items():
            for task in batch:
                r = validate_image(image_root, a.language, m, task, cfg['methods'][m])
                prompt = byid[r['prompt_id']]
                r['caption'] = prompt['prompt']
                if a.language == 'en':
                    r.update(category=prompt['category'], complex_kind=prompt.get('complex_kind'))
                else:
                    r.update(source_index=prompt['source_index'], annotations=prompt['annotations'])
                records.append(r)
        print('PASS IMAGES', len(records), 'PNG/sidecar/task/plan/config bindings', flush=True)
    if a.check_environment:
        from backends import english_provenance, chinese_provenance
        if a.language == 'en':
            english_provenance(config)
        else:
            chinese_provenance(config)
        print('PASS ENVIRONMENT EVIDENCE; no inference or service request', flush=True)
    if a.command == 'check':
        print('CHECK COMPLETE; no scoring requests or GPU inference')
        return
    need(records is not None, 'Images required')
    output = a.output.resolve()
    need(output != image_root and not output.is_relative_to(image_root), 'Choose a separate evaluation output directory')
    for protected in (ROOT.parent / 'en1920', ROOT.parent / 'zh150', generation):
        need(not output.is_relative_to(protected.resolve()), 'Output overlaps released data or generation package')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from backends import english_score, chinese_score
        all_scores = []
        for m in methods:
            selected = [r for r in records if r['method'] == m]
            complete = len(selected) == len(tasks[m])
            tag = 'full' if complete else 'subset_' + digest([(r['prompt_id'],r['seed']) for r in selected])[:16]
            folder = output / a.language / m / tag
            freeze(folder / 'run.frozen.json', {'evaluation_package_sha256':sha(ROOT / 'CHECKSUMS.json'), 'selection':selected, 'complete_method':complete})
            scored = (english_score if a.language == 'en' else chinese_score)(config, selected, folder, a.command, a.retry_failed)
            need({(r['prompt_id'],r['method'],r['seed']) for r in scored} == {(r['prompt_id'],r['method'],r['seed']) for r in selected} and len(scored) == len(selected), 'Incomplete score coverage')
            freeze(folder / 'summary.json', summarize(scored, a.language, complete))
            dest = folder / 'scores.jsonl'
            text = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in scored)
            if dest.exists():
                need(dest.read_text(encoding='utf-8') == text, 'Existing score export differs')
            else:
                dest.write_text(text, encoding='utf-8')
            all_scores.extend(scored)
            print('RESULT', dest, flush=True)
        tag = digest([(r['method'],r['prompt_id'],r['seed']) for r in all_scores])[:16]
        complete = all(len(batches[m]) == len(tasks[m]) for m in methods)
        freeze(output / a.language / ('summary_' + tag + '.json'), summarize(all_scores, a.language, complete))

if __name__ == '__main__':
    main()
