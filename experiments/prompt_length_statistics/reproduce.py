#!/usr/bin/env python3
"""Compute BiReG prompt-length statistics using only the Python standard library."""
import argparse
import csv
import hashlib
import json
import statistics
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECS = [
    ('development_zh', 'Development (zh)', 'development', 'prompt_zh', 'characters', 50),
    ('development_en', 'Development (en)', 'development', 'prompt_en', 'words', 50),
    ('main_en', 'English main', 'english', 'prompt', 'words', 1920),
    ('main_zh', 'Chinese main', 'chinese', 'prompt', 'characters', 150),
    ('main_zh_translation_en', 'English translations', 'chinese', 'translation_en', 'words', 150),
]
REPO_PATHS = {
    'development': 'data/development_prompts/bireg_development_pairs_50_v1.jsonl',
    'english': 'experiments/main_experiment/en1920/data/prompts.frozen.jsonl',
    'chinese': 'experiments/main_experiment/zh150/data/prompts.frozen.jsonl',
}
SNAPSHOT_PATHS = {
    'development': 'inputs/development_pairs_50.jsonl',
    'english': 'inputs/en1920.prompts.jsonl',
    'chinese': 'inputs/zh150.prompts.jsonl',
}


def count_characters(text):
    """Count Unicode code points other than whitespace and punctuation."""
    return sum(not c.isspace() and not unicodedata.category(c).startswith('P') for c in text)


def count_words(text):
    """Count whitespace-delimited tokens containing a Unicode letter or number."""
    return sum(any(c.isalnum() for c in token) for token in text.split())


def read_collection(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['prompt_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate prompt identifiers in ' + str(path))
    return rows


def format_number(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def make_latex(summary):
    row_lines = []
    for row in summary:
        row_lines.append(f"{row['label']} & {row['n']:,} & {row['mean']:.2f} & {format_number(row['median'])} & {row['min']}--{row['max']} " + r'\\')
    return '\n'.join([
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Prompt-length statistics for the development and main-experiment collections. Chinese lengths are measured in characters excluding whitespace and punctuation; English lengths count whitespace-delimited tokens containing letters or numbers.}',
        r'\label{tab:prompt_length_statistics}',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{2pt}',
        r'\renewcommand{\arraystretch}{1.12}',
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrr@{}}',
        r'\toprule',
        r'\textbf{Collection} & \textbf{$N$} & \textbf{Mean} & \textbf{Median} & \textbf{Range} \\',
        r'\midrule',
        *row_lines,
        r'\bottomrule',
        r'\end{tabular*}',
        r'\end{table}',
        '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, help='Read the three original files from a local BiReG checkout instead of bundled snapshots.')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    paths = {k: args.repo_root / p for k, p in REPO_PATHS.items()} if args.repo_root else {k: HERE / p for k, p in SNAPSHOT_PATHS.items()}
    collections = {k: read_collection(p) for k, p in paths.items()}
    manifest = json.loads((HERE / 'source_manifest.json').read_text(encoding='utf-8')) if (HERE / 'source_manifest.json').exists() else None
    if args.repo_root is None and manifest:
        for source in manifest['files']:
            data = (HERE / source['local_path']).read_bytes()
            git_blob = hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0' + data).hexdigest()
            if git_blob != source['git_blob_sha']:
                raise ValueError('Bundled snapshot differs from the recorded Git blob: ' + source['local_path'])
    summary, details = [], []
    for key, label, collection, field, unit, expected in SPECS:
        rows = collections[collection]
        if len(rows) != expected:
            raise ValueError(f'{key}: expected {expected} records, got {len(rows)}')
        values = []
        for row in rows:
            text = row[field]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'{key}: empty or invalid prompt for {row["prompt_id"]}')
            length = count_characters(text) if unit == 'characters' else count_words(text)
            values.append(length)
            details.append({'collection': key, 'prompt_id': row['prompt_id'], 'field': field, 'unit': unit, 'length': length})
        summary.append({'collection': key, 'label': label, 'n': len(values), 'unit': unit, 'mean': statistics.mean(values), 'median': statistics.median(values), 'min': min(values), 'max': max(values)})
    result = {
        'schema_version': 'bireg-prompt-length-statistics-v1',
        'source_repository': 'YeZhuang-Joe/BiReG',
        'source_commit': manifest['commit'] if manifest and args.repo_root is None else None,
        'unicode_database_version': unicodedata.unidata_version,
        'counting_rules': {
            'characters': 'Count Unicode code points other than whitespace or punctuation (Unicode General Category beginning with P); no text normalization.',
            'words': 'Split on Unicode whitespace and count tokens containing at least one Unicode letter or number (str.isalnum); no text normalization. Hyphenated or apostrophized forms without whitespace count as one token.',
        },
        'input_files': {k: {'path': REPO_PATHS[k], 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()},
        'summary': summary,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / 'per_prompt_lengths.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in details), encoding='utf-8')
    (args.output / 'table.tex').write_text(make_latex(summary), encoding='utf-8')
    markdown = ['| Collection | N | Unit | Mean | Median | Min–max |', '| --- | ---: | --- | ---: | ---: | --- |']
    for row in summary:
        markdown.append(f"| {row['label']} | {row['n']:,} | {row['unit']} | {row['mean']:.2f} | {format_number(row['median'])} | {row['min']}–{row['max']} |")
    (args.output / 'table.md').write_text('\n'.join(markdown) + '\n', encoding='utf-8')
    print('\n'.join(markdown))


if __name__ == '__main__':
    main()
