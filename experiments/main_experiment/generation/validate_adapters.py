#!/usr/bin/env python3
"""Validate frozen adapter inputs without importing torch or loading models."""
import sys,subprocess
from pathlib import Path
import run_generation as r
if len(sys.argv)==1:
 for lang in ['en','zh']:subprocess.run([sys.executable,__file__,lang],check=True)
else:
 lang=sys.argv[1];r.check(lang)
 for method in r.METHODS:
  tasks=r.load_tasks(lang,method)
  if tasks[0]['backend']=='adapter':
   adapter=r.setup_adapter(lang,method,Path('/unused/model/path'))
   for t in tasks:adapter._validate_task(t)
  else:
   for t in tasks:
    p=t.get('plan')
    if method=='ragd':
     for k in ['SR_hw_split_ratio','SR_prompt','HB_prompt_list','HB_m_offset_list','HB_n_offset_list','HB_m_scale_list','HB_n_scale_list']:assert k in p
     n=len(p['HB_prompt_list']);assert n>0
     assert all(len(p[k])==n for k in ['HB_m_offset_list','HB_n_offset_list','HB_m_scale_list','HB_n_scale_list'])
    elif t['generation_mode']=='regional':assert p['split_ratio'] and p['regional_prompt']
  print('VALIDATED',lang,method,len(tasks))
