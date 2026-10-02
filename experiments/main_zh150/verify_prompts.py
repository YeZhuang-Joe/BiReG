"""Validate this released prompt snapshot, without API/GPU/dependencies."""
from pathlib import Path
from collections import Counter
import hashlib,json
root=Path(__file__).resolve().parent
data=root/'data'
p=data/'prompts.frozen.jsonl'
meta=json.loads((data/'prompts.provenance.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['released_file_sha256'], 'Manifest hash mismatch'
a=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
assert len(a)==meta['n_prompts']
assert len({r['prompt_id'] for r in a})==len(a)
assert len({r['prompt_original'] for r in a})==len(a)
assert all(r['prompt_original'].strip() and r['source_record_id'] is None for r in a)
assert [r['source_manifest_line_1based'] for r in a]==list(range(1,len(a)+1))
if a[0]['language']=='en':
 assert len(a)==1920 and Counter(r['category'] for r in a)==Counter(dict.fromkeys(['color','shape','texture','spatial','non_spatial','complex'],320))
else:
 assert len(a)==150 and all(r['translation_en'].strip() for r in a)
 assert Counter(r['translation_review_status'] for r in a)==Counter(meta['translation_review_counts'])
print(f'PASS: {len(a)} prompts; unique IDs/texts; frozen hash; counts and required fields verified.')
