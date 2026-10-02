"""Input binding and aggregation for retained EN1920/ZH150 main experiments."""
import copy
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ('sdxl', 'rpg', 'kolors', 'rpg_kolors', 'ragd', 'bireg')
SEEDS = {'en': (2026, 3407, 5678), 'zh': (1234, 2468, 42)}
CATS = ('color', 'shape', 'texture', 'spatial', 'non_spatial', 'complex')

def need(ok, message):
    if not ok:
        raise ValueError(message)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(obj):
    # Exactly the generation runner's canonicalization.
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def freeze(p, obj):
    p = Path(p)
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if p.exists():
        need(p.read_text(encoding='utf-8') == data, 'Existing record differs: ' + str(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        f.write(data)

def package_check():
    for name, expected in read(ROOT / 'CHECKSUMS.json').items():
        need(sha(ROOT / name) == expected, 'Evaluation package changed: ' + name)

def inputs(generation, lang):
    hashes = read(ROOT / 'references/generation_input_hashes.json')
    for name, expected in hashes.items():
        if name.startswith('data/' + lang + '/') or name == 'configs/' + lang + '.json':
            need(sha(generation / name) == expected, 'Frozen generation input changed: ' + name)
    prompts = rows(ROOT / ('data/' + lang + '.prompts.jsonl'))
    n = 1920 if lang == 'en' else 150
    need(len(prompts) == len({r['prompt_id'] for r in prompts}) == n, 'Prompt count/IDs')
    if lang == 'en':
        need(Counter(r['category'] for r in prompts) == Counter(dict.fromkeys(CATS, 320)), 'Category counts')
    cfg = read(generation / ('configs/' + lang + '.json'))
    need(cfg['seeds'] == list(SEEDS[lang]), 'Seed configuration differs')
    tasks = {}
    for method in METHODS:
        batch = rows(generation / f'data/{lang}/{method}.tasks.jsonl')
        need(len(batch) == n * 3, 'Task count: ' + method)
        need(Counter((t['prompt_id'], t['seed']) for t in batch) == Counter((r['prompt_id'], s) for r in prompts for s in SEEDS[lang]), 'Task coverage: ' + method)
        family = 'rpg' if lang == 'en' and method == 'rpg_kolors' else method
        path = generation / f'data/{lang}/{family}.plans.frozen.jsonl'
        plans = {r['prompt_id']: r for r in rows(path)} if path.exists() else {}
        for t in batch:
            if 'plan_ref' in t:
                r = plans[t['plan_ref']]
                t['plan'] = r['plan']
                t['plan_provenance'] = r.get('provenance', r.get('planning_provenance'))
            elif method == 'ragd':
                r = plans[t['prompt_id']]
                t['plan'] = r['plan']
                t['plan_provenance'] = r.get('planning_provenance')
        tasks[method] = batch
    return prompts, cfg, tasks

def select(batch, prompt_ids, seed, limit):
    chosen = [t for t in batch if (not prompt_ids or t['prompt_id'] in prompt_ids) and (seed is None or t['seed'] == seed)]
    if limit is not None:
        chosen = chosen[:limit]
    need(chosen, 'Empty task selection')
    return chosen

def validate_image(image_root, lang, method, expected, settings):
    from PIL import Image
    image = image_root / lang / method / expected['prompt_id'] / f"seed_{expected['seed']}.png"
    side = image.with_suffix('.json')
    need(image.is_file() and side.is_file(), 'Missing image/sidecar: ' + str(image))
    record = read(side)
    need(record.get('status') == 'completed', 'Image is not completed')
    runtime = record['runtime']
    need(runtime['settings'] == settings, 'Generation settings differ')
    task = copy.deepcopy(expected)
    if 'generation' in task:
        task['generation']['model_path'] = runtime['model_path']
    need(record['task'] == task, 'Task/plan differs: ' + str(side))
    need(record['fingerprint'] == digest({'task': task, 'runtime': runtime}), 'Generation fingerprint differs')
    image_hash = sha(image)
    need(record['image_sha256'] == image_hash, 'Image hash differs')
    with Image.open(image) as im:
        im.load()
        need(im.format == 'PNG' and im.size == (settings['width'], settings['height']), 'Image format/size differs')
    return {'prompt_id': task['prompt_id'], 'method': method, 'seed': task['seed'], 'image_path': str(image.resolve()), 'image_sha256': image_hash, 'sidecar_sha256': sha(side), 'generation_release_sha256': runtime['release_sha256']}

def english_value(category, kind, values):
    keys = ['a'] if category in CATS[:3] else ['s'] if category == 'spatial' else ['c'] if category == 'non_spatial' else {'spatial':['a','s'], 'action':['a','c'], 'both':['a','s','c']}[kind]
    need(all(type(values[k]) in (int, float) and math.isfinite(values[k]) for k in keys), 'Nonfinite branch score')
    return statistics.mean(values[k] for k in keys)

def valid_zh(result, task):
    obj = result.get('result_json')
    need(isinstance(obj, dict), 'No parsed judge result')
    need(result.get('prompt') == task['prompt'] and result.get('img_path') == task['img_path'], 'Judge input differs')
    need(obj.get('testpoint') == task['testpoint'], 'Judge checkpoint labels differ')
    scores = obj.get('score')
    need(isinstance(scores, list) and len(scores) == len(task['testpoint']), 'Judge score count differs')
    need(all(type(v) is int and v in (0, 1) for v in scores), 'Non-binary checkpoint score')
    return obj

def summarize(records, lang, complete):
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    seen = set()
    for r in records:
        key = (r['prompt_id'], r['method'], r['seed'])
        need(key not in seen, 'Duplicate score record')
        seen.add(key)
        if lang == 'en':
            items = [(r['category'], r['score'])]
        else:
            need(len(r['testpoint']) == len(r['score']), 'Checkpoint alignment')
            items = [(dim, value) for label, value in zip(r['testpoint'], r['score']) for dim in {label, label.split('-')[0]}]
        for dim, value in items:
            cell = groups[(r['method'], dim)][r['seed']]
            cell[0] += value
            cell[1] += 1
    report = []
    for (method, dim), seeds in sorted(groups.items()):
        means = {str(s): v[0] / v[1] for s, v in sorted(seeds.items())}
        report.append({'method': method, 'dimension': dim, 'counts': {str(s):v for s,v in seeds.items()}, 'seed_means': means, 'mean': statistics.mean(means.values()), 'sample_sd': statistics.stdev(means.values()) if len(means) == 3 else None})
    return {'language': lang, 'complete_selected_methods': complete, 'n_images': len(records), 'scale': 'archived metric' if lang == 'en' else '0–1 checkpoint accuracy', 'results': report}
