import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'vendor/historical_kolors'

def now(): return datetime.now(timezone.utc).isoformat()
def digest(data): return hashlib.sha256(data).hexdigest()
def filehash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def encoded(obj): return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=encoded(obj)
    if path.exists():
        if path.read_bytes()!=b: raise ValueError(f'Refusing to overwrite different record: {path}')
        return
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f: f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def jsonlines(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def save_lines(path,rows):
    b=''.join(json.dumps(x,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for x in rows).encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=b:raise ValueError(f'Refusing to replace {path}')
    if not path.exists():
        temp=path.with_name(path.name+'.tmp');temp.write_bytes(b);os.replace(temp,path)
def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,150}',value):raise ValueError('Unsafe/empty identifier')
    return value

def source_hashes():
    expected=load(ROOT/'source_hashes.json')
    for name,h in expected.items():
        if filehash(ROOT/name)!=h:raise ValueError(f'Vendored source/template hash mismatch: {name}')
    return expected

def code_hashes():
    return {str(p.relative_to(ROOT)):filehash(p) for p in sorted((ROOT/'bireg').glob('*.py'))}

@contextlib.contextmanager
def lock(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def environment():
    names=['torch','torchvision','diffusers','transformers','accelerate','xformers','numpy','Pillow','sentencepiece','safetensors','huggingface-hub','opencv-python']
    versions={}
    for name in names:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return dict(python=platform.python_version(),platform=platform.platform(),executable=sys.executable,packages=versions)
