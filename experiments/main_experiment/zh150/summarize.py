"""Validate ZH150 coverage and aggregate archived checkpoint scores (stdlib only)."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHODS = ['sdxl', 'rpg', 'kolors', 'rpg_kolors', 'ragd', 'bireg']
NAMES = ['SDXL', 'RPG', 'Kolors', 'RPG+Kolors', 'RAGD', 'BiReG']
SEEDS = [1234, 2468, 42]
DIMENSIONS = ['属性', '动作', '关系', '实体布局', '复合考点', '风格', '世界知识', '语法', '逻辑推理', '文本生成']
ENGLISH = ['Attribute', 'Action', 'Relation', 'Entity layout', 'Composite', 'Style', 'World knowledge', 'Grammar', 'Logical reasoning', 'Text rendering']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    prompts = read_jsonl(ROOT / 'data/prompts.frozen.jsonl')
    ids = {r['prompt_id'] for r in prompts}
    require(len(prompts) == len(ids) == 150, 'Expected 150 unique prompts')
    rows = read_jsonl(ROOT / 'data/scores.jsonl')
    seen, bindings = set(), {}
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    for row in rows:
        pid, method, seed = row['prompt_id'], row['method'], row['seed']
        key = (pid, method, seed)
        require(pid in ids and method in METHODS and type(seed) is int and seed in SEEDS, f'Unexpected record: {key}')
        require(key not in seen, f'Duplicate record: {key}')
        seen.add(key)
        labels, scores = row['testpoint'], row['score']
        require(isinstance(labels, list) and isinstance(scores, list) and len(labels) == len(scores) > 0, f'Invalid checkpoint arrays: {key}')
        require(all(isinstance(label, str) and label.split('-')[0] in DIMENSIONS for label in labels), f'Invalid labels: {key}')
        require(all(type(score) is int and score in (0, 1) for score in scores), f'Non-binary scores: {key}')
        require(bindings.setdefault(pid, labels) == labels, f'Checkpoint mismatch: {key}')
        for label, score in zip(labels, scores):
            for dimension in {label, label.split('-')[0]}:
                counts[method][dimension][seed][0] += score
                counts[method][dimension][seed][1] += 1
    expected = {(pid, method, seed) for pid in ids for method in METHODS for seed in SEEDS}
    require(seen == expected, f'Incomplete coverage: {len(expected - seen)} missing records')
    results = {}
    for method in METHODS:
        results[method] = {}
        require(set(DIMENSIONS) <= set(counts[method]), f'Missing dimensions: {method}')
        for dimension, by_seed in sorted(counts[method].items()):
            rates = [by_seed[s][0] / by_seed[s][1] for s in SEEDS]
            results[method][dimension] = {
                'mean': statistics.mean(rates),
                'sample_sd': statistics.stdev(rates),
                'counts': dict(by_seed),
                'seed_scores': dict(zip(SEEDS, rates)),
            }
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps({'n_prompts': len(ids), 'n_images': len(rows), 'seeds': SEEDS, 'scale': '0–1', 'results': results}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    table = ['# ZH150 automatic evaluation', '', 'Checkpoint accuracy (%), mean ± sample SD across three generation seeds.', '', '| Dimension | ' + ' | '.join(NAMES) + ' |', '|---|' + '---:|' * len(METHODS)]
    for dimension, english in zip(DIMENSIONS, ENGLISH):
        cells = [f"{results[m][dimension]['mean'] * 100:.2f} ± {results[m][dimension]['sample_sd'] * 100:.2f}" for m in METHODS]
        table.append('| ' + english + ' | ' + ' | '.join(cells) + ' |')
    (output / 'table.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    print(f'Validated {len(ids)} prompts and {len(rows)} image records; wrote {output}')


if __name__ == '__main__':
    main()
