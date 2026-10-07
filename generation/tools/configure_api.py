#!/usr/bin/env python3
"""Create a local private API configuration; never prints the credential."""
import getpass
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bireg.workflow import read_private

def main():
    target=ROOT/'private/api_config.json'
    if target.exists():
        print('Configuration already exists:',target)
        print('Edit it locally to change endpoint/model/key. Existing file was not overwritten.')
        return
    data=json.loads((ROOT/'configs/api.example.json').read_text(encoding='utf-8'))
    for field in ('endpoint','model'):
        value=input(field+' ['+data[field]+']: ').strip()
        if value:data[field]=value
    data['api_key']=getpass.getpass('API key (hidden): ').strip()
    if not data['api_key']:raise ValueError('Empty API key; no file written')
    target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(str(target),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as out:json.dump(data,out,ensure_ascii=False,indent=2);out.write('\n')
        read_private(target)
    except Exception:
        target.unlink()
        raise
    print('Saved local private configuration:',target)
    print('For other API providers, remove the optional thinking field if unsupported.')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as exc:print('ERROR:',str(exc),file=sys.stderr);sys.exit(2)
