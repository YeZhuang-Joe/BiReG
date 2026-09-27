#!/usr/bin/env python3
"""Verify distributed source/document hashes; local private/output files are outside the manifest."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'MANIFEST.sha256.json').read_text(encoding='utf-8'))
failed=[]
for name,expected in manifest.items():
    path=ROOT/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:failed.append(name)
if failed:raise SystemExit('Mismatch: '+', '.join(failed))
print('PASS:',len(manifest),'distributed files match their hashes. No API/GPU called.')
