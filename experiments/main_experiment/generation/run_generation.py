#!/usr/bin/env python3
"""Unified offline replay of EN1920/ZH150; no planner or evaluator calls."""
import argparse, copy, hashlib, importlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
METHODS=('sdxl','rpg','kolors','rpg_kolors','ragd','bireg')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n');tmp.replace(p)
def load_tasks(lang,method):
 tasks=rows(ROOT/f'data/{lang}/{method}.tasks.jsonl')
 family=('rpg' if method=='rpg_kolors' and lang=='en' else method)
 p=ROOT/f'data/{lang}/{family}.plans.frozen.jsonl'
 plans={r['prompt_id']:r for r in rows(p)} if p.exists() else {}
 for t in tasks:
  if 'plan_ref' in t:
   r=plans[t['plan_ref']];t['plan']=r['plan'];t['plan_provenance']=r.get('provenance',r.get('planning_provenance'))
  elif method=='ragd':
   r=plans[t['prompt_id']];t['plan']=r['plan'];t['plan_provenance']=r.get('planning_provenance')
 return tasks

def setup_adapter(lang,method,model):
 sys.path.insert(0,str(ROOT/'vendor/project'));sys.path.insert(0,str(ROOT/'vendor'/lang))
 regional=ROOT/'vendor/regional'
 if method=='bireg':
  cls=importlib.import_module('benchmark_adapter').BenchmarkBiReGAdapter
 else:
  module=('example_'+method) if lang=='en' else ('bireg_experiment.adapters.'+method)
  cls=getattr(importlib.import_module(module),{'sdxl':'SDXLAdapter','rpg':'RPGAdapter','kolors':'KolorsAdapter','rpg_kolors':'RPGKolorsAdapter'}[method])
 return cls(model,regional) if method in ('rpg','rpg_kolors','bireg') else cls(model)

def check(lang):
 checks=read(ROOT/'CHECKSUMS.json')
 for name,expected in checks.items():
  if file_sha(ROOT/name)!=expected:raise ValueError('Release file changed: '+name)
 prompts=rows(ROOT/f'data/{lang}/prompts.frozen.jsonl');byid={r['prompt_id']:r for r in prompts};n=1920 if lang=='en' else 150
 assert len(prompts)==len(byid)==n
 cfg=read(ROOT/f'configs/{lang}.json');expected={(p,s) for p in byid for s in cfg['seeds']}
 for m in METHODS:
  ts=load_tasks(lang,m);actual={(t['prompt_id'],t['seed']) for t in ts}
  assert len(ts)==len(actual)==n*3 and actual==expected,(lang,m,'coverage')
  for t in ts:
   pid=t['prompt_id'];assert t['language']==lang and t['method']==m
   original=byid[pid].get('prompt',byid[pid].get('prompt_original'))
   if lang=='en' or m in ('kolors','rpg_kolors','bireg'):assert t['prompt']==original
   else:assert t['prompt_zh']==original and t['prompt']==t['prompt_en']
   if m in ('rpg','rpg_kolors','ragd','bireg') and t['generation_mode']=='regional':assert isinstance(t['plan'],dict)
  print(f'PASS {lang}/{m}: {len(ts)} tasks, {n} prompts',flush=True)
 return cfg

def worker(a):
 import time,fcntl,traceback
 local=read(a.local_config);cfg=read(ROOT/f'configs/{a.language}.json')['methods'][a.method]
 tasks=load_tasks(a.language,a.method)
 if a.prompt_id:tasks=[t for t in tasks if t['prompt_id'] in a.prompt_id]
 if a.seed is not None:tasks=[t for t in tasks if t['seed']==a.seed]
 if a.limit:tasks=tasks[:a.limit]
 if a.mode:tasks=[t for t in tasks if t['generation_mode']==a.mode]
 if not tasks:return
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 modelkey='flux' if a.method=='ragd' else ('sdxl' if a.method in ('sdxl','rpg') else 'kolors')
 model=Path(local['models'][modelkey]).expanduser().resolve()
 if not model.is_dir():raise ValueError('Model directory missing: '+str(model))
 runtime={'settings':cfg,'model_path':str(model),'release_sha256':file_sha(ROOT/'CHECKSUMS.json'),'python':sys.version}
 pending=[]
 for t in tasks:
  t=copy.deepcopy(t)
  if 'generation' in t:t['generation']['model_path']=str(model)
  image=out/a.language/a.method/t['prompt_id']/f"seed_{t['seed']}.png";side=image.with_suffix('.json');fp=digest({'task':t,'runtime':runtime})
  if image.exists() or side.exists():
   if not (image.is_file() and side.is_file()):raise ValueError('Incomplete existing output: '+str(image))
   old=read(side)
   if old.get('fingerprint')!=fp or old.get('image_sha256')!=file_sha(image):raise ValueError('Existing output differs: '+str(image))
   continue
  failed=image.with_suffix('.failed.json')
  if failed.exists() and not a.retry_failed:raise ValueError('Prior failure; inspect and use --retry-failed: '+str(failed))
  pending.append((t,image,side,fp))
 if not pending:print('All selected outputs verified; nothing to generate.');return
 with (out/'generation.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  adapter=None;pipe=None
  if pending[0][0]['backend']=='adapter':
   adapter=setup_adapter(a.language,a.method,model);adapter.load()
  else:
   sys.path.insert(0,str(ROOT/'vendor/zh_baselines'))
   hr=importlib.import_module('historical_runner')
   from types import SimpleNamespace
   args=SimpleNamespace(ragd_repo=ROOT/'vendor/ragd',flux_model=model,sdxl_model=model)
   pipe=hr.load_pipeline(a.method,pending[0][0]['generation_mode'],args,cfg)
   if a.method=='rpg' and a.language=='zh' and a.mode=='regional':
    module=sys.modules['native_source.RegionalDiffusion_xl'];matrix=sys.modules['matrix'];original=module.matrixdealer
    def compatible(state,ratio,base):
     if ratio=='1.0':
      if base!=.5:raise ValueError('Unexpected single-region base ratio')
      state.split_ratio=[matrix.Row(0.,1.,[matrix.Region(0.,1.,base,0)])];state.baseratio=[[base]]
     else:original(state,ratio,base)
    module.matrixdealer=compatible
  try:
   import torch
   from importlib import metadata as package_metadata
   env={n:package_metadata.version(n) for n in ('torch','diffusers','transformers','accelerate')}
   for i,(t,image,side,fp) in enumerate(pending,1):
    if (out/'STOP').exists():print('STOP requested');break
    image.parent.mkdir(parents=True,exist_ok=True);start=time.time()
    try:
     if adapter:
      if a.language=='en' and a.method in ('rpg','rpg_kolors') and t['plan']['region_count']==1:
       from regional_compat import single_region_compatibility
       with single_region_compatibility(adapter.pipe):meta=adapter.generate(t,image)
      else:meta=adapter.generate(t,image)
     else:
      runtime_task=dict(t,prompt_en=t['prompt'])
      result=hr.generate_one(pipe,runtime_task,cfg);im=result.images[0]
      if im.size!=(cfg['width'],cfg['height']):raise ValueError('Unexpected image dimensions')
      im.save(image);meta={'scheduler_class':type(pipe.scheduler).__name__,'scheduler_config':dict(pipe.scheduler.config)}
     write(side,{'status':'completed','task':t,'runtime':runtime,'environment':env,'fingerprint':fp,'image_sha256':file_sha(image),'method_metadata':meta,'elapsed_seconds':time.time()-start})
     print(f'{i}/{len(pending)} {a.language}/{a.method}/{t["prompt_id"]}/{t["seed"]}',flush=True)
    except Exception:
     write(image.with_suffix('.failed.json'),{'task':t,'runtime':runtime,'error':traceback.format_exc()})
     # Retain orphan image for inspection. Resume refuses ambiguous output.
     raise
  finally:
   if adapter:adapter.close()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--language',choices=['en','zh'],required=True);p.add_argument('--method',choices=[*METHODS,'all'],default='all')
 p.add_argument('--check',action='store_true');p.add_argument('--local-config',type=Path,default=ROOT/'configs/local.json');p.add_argument('--output',type=Path,default=ROOT/'outputs')
 p.add_argument('--prompt-id',action='append');p.add_argument('--seed',type=int);p.add_argument('--limit',type=int)
 p.add_argument('--retry-failed',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--mode',help=argparse.SUPPRESS)
 a=p.parse_args()
 if a.limit is not None and a.limit<1:p.error('--limit must be positive')
 if a.worker:worker(a);return
 cfg=check(a.language)
 if a.check:return
 if a.seed is not None and a.seed not in cfg['seeds']:p.error('Seed not in frozen experiment')
 if a.prompt_id:
  ids={r['prompt_id'] for r in rows(ROOT/f'data/{a.language}/prompts.frozen.jsonl')}
  if not set(a.prompt_id)<=ids:p.error('Prompt ID outside main experiment')
 local=read(a.local_config)
 for m in METHODS if a.method=='all' else [a.method]:
  executable=local.get('python',{}).get('ragd' if m=='ragd' else 'default',sys.executable)
  modes=['regional','baseline_fallback'] if a.language=='zh' and m=='rpg' else [None]
  # Use exact archived mode spelling.
  if a.language=='zh' and m=='rpg':modes=sorted({t['generation_mode'] for t in load_tasks('zh','rpg')})
  for mode in modes:
   cmd=[executable,str(Path(__file__).resolve()),'--worker','--language',a.language,'--method',m,'--local-config',str(a.local_config.resolve()),'--output',str(a.output.resolve())]
   if mode:cmd+=['--mode',mode]
   if a.seed is not None:cmd+=['--seed',str(a.seed)]
   if a.limit:cmd+=['--limit',str(a.limit)]
   for pid in a.prompt_id or []:cmd+=['--prompt-id',pid]
   if a.retry_failed:cmd+=['--retry-failed']
   env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1')
   subprocess.run(cmd,check=True,env=env)
if __name__=='__main__':main()
