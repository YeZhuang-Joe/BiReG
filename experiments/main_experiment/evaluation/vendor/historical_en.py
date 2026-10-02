#!/usr/bin/env python3
"""Score the frozen 1920-prompt RAGD main experiment with retained T2I-CompBench branches."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODELS = ['ragd']
CATS = ['color', 'shape', 'texture', 'spatial', 'non_spatial', 'complex']
SEEDS = [2026, 3407, 5678]
REV = '4aa404212eb5d06e5adbcd9cee696c750d0d25a5'
BRANCH = {
 'a': ('BLIPvqa_eval', 'BLIP_vqa.py', '--out_dir', 'annotation_blip/vqa_result.json'),
 's': ('UniDet_eval', '2D_spatial_eval.py', '--outpath', 'labels/annotation_obj_detection_2d/vqa_result.json'),
 'c': ('CLIPScore_eval', 'CLIP_similarity.py', '--outpath', 'annotation_clip/vqa_result.json'),
}
def enc(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def obj(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def need(x,msg):
    if not x: raise ValueError(msg)
def freeze(p,x):
    p=Path(p); data=enc(x); p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): need(p.read_bytes()==data, 'Frozen content differs: '+str(p))
    else:
        with p.open('xb') as f:f.write(data)
def now():return datetime.now(timezone.utc).isoformat()
def jsonl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]
def fingerprint(t,plan,g):
    value={'source_task':t,'plan':plan,'generation':g}
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def package():
    for name,h in obj(ROOT/'PACKAGE.json').items():need(sha(ROOT/name)==h,'Package changed: '+name)
    rows=jsonl(ROOT/'prompts_en1920.jsonl');tasks=jsonl(ROOT/'tasks.frozen.jsonl')
    need(len(rows)==1920 and len({r['prompt_id'] for r in rows})==1920,'Prompt IDs/count')
    need(Counter(r['category'] for r in rows)==Counter({c:320 for c in CATS}),'Category quotas')
    need(Counter((t['prompt_id'],t['seed']) for t in tasks)==Counter((r['prompt_id'],s) for r in rows for s in SEEDS),'Task coverage')
    need(len({t['task_id'] for t in tasks})==5760,'Duplicate task IDs')
    byid={r['prompt_id']:r for r in rows}
    for t in tasks:
        need(t['method']=='ragd','Wrong method')
        need(all(t[k]==byid[t['prompt_id']][k] for k in ['prompt','category','complex_kind','source_index','source_cohort']),'Task/prompt differs')
        if t['category']=='complex':need(t['complex_kind'] in ['spatial','action','both'],'Invalid complex kind')
    return rows

def source_images(a,rows,model):
    from PIL import Image
    run=a.run_dir;cfg=obj(run/'configuration.json');g=cfg['generation']
    need(sha(run/'configuration.json')==sha(ROOT/'configuration.json'),'Generation config changed')
    need(sha(run/'tasks.frozen.jsonl')==sha(ROOT/'tasks.frozen.jsonl')==cfg['tasks_sha256'],'Frozen tasks changed')
    need(sha(run/'plans.frozen.jsonl')==cfg['plans_sha256'],'Frozen plans changed')
    plans=jsonl(run/'plans.frozen.jsonl');byid={r['prompt_id']:r for r in plans}
    need(len(plans)==1920 and set(byid)=={r['prompt_id'] for r in rows},'Plan coverage')
    selected={next(r for r in rows if r['category']==c)['prompt_id'] for c in CATS} if a.smoke else {r['prompt_id'] for r in rows}
    tasks=[t for t in jsonl(run/'tasks.frozen.jsonl') if t['prompt_id'] in selected and (not a.smoke or t['seed']==2026)]
    out=[];cfgsha=sha(run/'configuration.json')
    for t in tasks:
        plan=byid[t['prompt_id']];need(plan['prompt']==t['prompt'],'Plan prompt changed')
        image=run/'images'/t['prompt_id']/('seed_'+str(t['seed'])+'.png');p=image.with_suffix('.json');rec=obj(p)
        fp=fingerprint(t,plan,g)
        need(rec.get('status')=='completed' and rec['task']==t,'Incomplete/wrong task: '+t['task_id'])
        need(rec['configuration_sha256']==cfgsha and rec['task_fingerprint']==fp,'Completion binding changed')
        need(rec['plan']==plan['plan'] and rec['planning_provenance']==plan['planning_provenance'],'Plan completion mismatch')
        need(rec['generation']==dict(g,seed=t['seed']),'Per-image generation settings changed')
        need(sha(image)==rec['image_sha256'] and image.stat().st_size==rec['image_size_bytes'],'Image hash/size mismatch: '+str(image))
        with Image.open(image) as im:
            need(im.size==(1024,1024) and im.mode=='RGB' and im.format=='PNG','Image properties differ')
            im.load()
        out.append(dict(planner=model,prompt_id=t['prompt_id'],seed=t['seed'],category=t['category'],complex_kind=t.get('complex_kind'),caption=t['prompt'],source_index=t['source_index'],source_cohort=t['source_cohort'],task_id=t['task_id'],task_sha256=fp,image_sha256=rec['image_sha256'],image_path=str(image.resolve()),result_path=str(p.resolve()),result_sha256=sha(p),frozen_plans_sha256=cfg['plans_sha256']))
    return out,g

def export(a,records,model):
    result={}
    for cat in CATS:
        d=a.output/('smoke' if a.smoke else 'full')/model/cat
        mapping=[]; samples=d/'samples';samples.mkdir(parents=True,exist_ok=True)
        for r in (r for r in records if r['category']==cat):
            qid=len(mapping);caption=r['caption']; name=caption+f'_{qid:06d}.png'
            need(not any(c in caption for c in ['_','/','\\','\n','\r','\x00']) and len(name.encode())<=255,'Unsupported caption filename: '+r['prompt_id'])
            target=samples/name
            if target.is_symlink():need(target.resolve()==Path(r['image_path']),'Wrong export link')
            else:
                need(not target.exists(),'Export collision');target.symlink_to(r['image_path'])
            mapping.append(dict(r,question_id=qid,filename=name))
        need({p.name for p in samples.iterdir()}=={r['filename'] for r in mapping},'Extra exported images')
        freeze(d/'image_map.json',mapping);result[cat]=(d,mapping)
    return result

def branches(cat):return ['a'] if cat in CATS[:3] else ['s'] if cat=='spatial' else ['c'] if cat=='non_spatial' else ['a','s','c']
def read_scores(path,mapping):
    data=obj(path); need(isinstance(data,list),'Score output must be a JSON list')
    scores={}
    for r in data:
        k=int(r['question_id']);v=float(r['answer'])
        need(k not in scores and math.isfinite(v),'Duplicate/nonfinite score')
        scores[k]=v
    need(set(scores)=={r['question_id'] for r in mapping},'Score IDs do not match images')
    return scores

def provenance(a):
    def git(*args):return subprocess.check_output(['git','-C',str(a.repo),*args],text=True).strip()
    need(git('rev-parse','HEAD')==REV,'Evaluator revision mismatch')
    folders=[v[0] for v in BRANCH.values()]
    need(not git('diff','HEAD','--',*folders),'Tracked evaluator changes found')
    source={}
    for folder in folders:
        for p in sorted((a.repo/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.yaml','.yml','.json') and '__pycache__' not in p.parts:
                source[str(p.relative_to(a.repo))]=sha(p)
    # Record actual local weights, including files outside Git tracking.
    weights={}
    roots=[a.repo/'UniDet_eval/experts/expert_weights',Path('/root/.cache/torch/hub/checkpoints'),Path('/root/.cache/clip')]
    for root in roots:
        if root.exists():
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.suffix in ('.pth','.pt','.pkl','.bin','.safetensors'):weights[str(p.resolve())]=sha(p)
    need(any('model_base_vqa_capfilt_large.pth' in p for p in weights),'BLIP weight not found')
    need(any('Unified_learned_OCIM_RS200_6x+2x.pth' in p for p in weights),'UniDet weight not found')
    need(any('ViT-B-32.pt' in p for p in weights),'CLIP weight not found')
    packages={}
    for name in ['torch','torchvision','transformers','timm','fairscale','detectron2','numpy','Pillow','spacy']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return dict(revision=REV,source_sha256=source,weight_sha256=weights,python=sys.version,executable=sys.executable,packages=packages,adapter_sha256=sha(Path(__file__)),prompt_release_sha256=sha(ROOT/'prompts_en1920.jsonl'),offline_hf=True)

def branch_result(d,b,mapping,prov):
    receipt=d/('completed_'+b+'.json')
    if not receipt.exists():return None
    r=obj(receipt)
    need(r['input_sha256']==sha(d/'image_map.json'),'Cached input mismatch')
    need(r['provenance_sha256']==hashlib.sha256(enc(prov)).hexdigest(),'Cached evaluator changed')
    p=d/r['score_path'];need(sha(p)==r['score_sha256'],'Cached score changed')
    return read_scores(p,mapping)

def run_branch(a,d,b,cat,mapping,prov):
    if branch_result(d,b,mapping,prov) is not None:
        print('REUSE',d.parent.name,cat,b,flush=True);return
    folder,script,flag,rel=BRANCH[b]
    attempts=d/('branch_'+b);attempts.mkdir(exist_ok=True)
    n=1
    while (attempts/f'attempt_{n:03d}').exists():n+=1
    work=attempts/f'attempt_{n:03d}';work.mkdir()
    (work/'samples').symlink_to((d/'samples').resolve(),target_is_directory=True)
    cmd=[sys.executable,script,flag,str(work)]
    if cat=='complex' and b!='a':cmd+=['--complex','True']
    freeze(work/'invocation.json',dict(command=cmd,cwd=str(a.repo/folder),at=now(),input_sha256=sha(d/'image_map.json'),provenance_sha256=hashlib.sha256(enc(prov)).hexdigest()))
    print('SCORE',d.parent.name,cat,b,'LOG',work/'process.log',flush=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    with (work/'process.log').open('wb') as log:
        p=subprocess.run(cmd,cwd=a.repo/folder,env=env,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:
        print((work/'process.log').read_text(errors='replace')[-7000:],flush=True)
        raise RuntimeError('Scorer failed; original log retained. No dependency/config changes made.')
    score=work/rel
    read_scores(score,mapping)
    freeze(d/('completed_'+b+'.json'),dict(input_sha256=sha(d/'image_map.json'),provenance_sha256=hashlib.sha256(enc(prov)).hexdigest(),score_path=str(score.relative_to(d)),score_sha256=sha(score),at=now()))
    print('ACCEPTED',d.parent.name,cat,b,len(mapping),flush=True)

def aggregate(a,exports,prov):
    reports={}; allrows=[]
    for model,cats in exports.items():
        report={};seed_by_cat={}
        for cat,(d,mapping) in cats.items():
            values={b:branch_result(d,b,mapping,prov) for b in branches(cat)}
            need(all(v is not None for v in values.values()),'Missing branch scores: '+model+'/'+cat)
            out=[]
            for r in mapping:
                keys=branches(cat)
                if cat=='complex':keys={'spatial':['a','s'],'action':['a','c'],'both':['a','s','c']}[r['complex_kind']]
                out.append(dict(r,score=statistics.mean(values[k][r['question_id']] for k in keys),branch_scores={k:values[k][r['question_id']] for k in values},aggregation_branches=keys))
            seeds={str(s):statistics.mean(r['score'] for r in out if r['seed']==s) for s in (SEEDS[:1] if a.smoke else SEEDS)}
            seed_by_cat[cat]=seeds
            report[cat]=dict(prompt_count=len({r['prompt_id'] for r in out}),image_count=len(out),seed_scores=seeds,mean=statistics.mean(seeds.values()),sample_sd=statistics.stdev(seeds.values()) if len(seeds)>1 else None)
            allrows.extend(out)
        macro={s:statistics.mean(seed_by_cat[c][s] for c in CATS) for s in seed_by_cat[CATS[0]]}
        report['auxiliary_macro']=dict(seed_scores=macro,mean=statistics.mean(macro.values()),sample_sd=statistics.stdev(macro.values()) if len(macro)>1 else None,official_overall=False)
        reports[model]=report
    result=dict(smoke_only=a.smoke,models=reports,provenance_sha256=hashlib.sha256(enc(prov)).hexdigest(),definition='Category mean per generation seed, then mean and sample SD across three seed-level means; smoke has only one seed and no SD. Complex follows the retained BiReG main-experiment adapter, not an unmodified official complex score.')
    dest=a.output/('smoke' if a.smoke else 'full')/'summaries';dest.mkdir(exist_ok=True)
    # Separate subsets can be scored independently without overwriting each other.
    tag='__'.join(exports)
    freeze(dest/(tag+'_summary.json'),result)
    freeze(dest/(tag+'_image_scores.json'),allrows)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','run','aggregate'])
    p.add_argument('--run-dir',type=Path,default=Path('/root/autodl-tmp/ragd_workspace/english1920_ragd_generation_v1'))
    p.add_argument('--repo',type=Path,default=Path('/root/autodl-tmp/T2I-CompBench-eval'))
    p.add_argument('--output',type=Path,default=Path('/root/autodl-tmp/ragd_workspace/english1920_ragd_eval_v1'))
    p.add_argument('--smoke',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.run_dir=a.run_dir.resolve();a.repo=a.repo.resolve();a.output=a.output.resolve()
    need(a.command!='run' or a.execute,'Scoring requires --execute')
    rows=package();models=MODELS
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prov=provenance(a);freeze(a.output/'scorer_provenance.json',prov)
        exports={};gens=[]
        for model in models:
            records,g=source_images(a,rows,model);gens.append(g);exports[model]=export(a,records,model)
            print('VERIFIED',model,len(records),'English PNGs and frozen task bindings',flush=True)
        need(all(g==gens[0] for g in gens),'Generation config differs between planners')
        if a.command=='check':print('CHECK PASS; no GPU inference, no API calls');return
        if a.command=='run':
            import torch
            need(torch.cuda.is_available(),'CUDA unavailable in evaluator environment')
        for model,cats in exports.items():
            if a.command=='run':
                for cat,(d,mapping) in cats.items():
                    for b in branches(cat):run_branch(a,d,b,cat,mapping,prov)
        aggregate(a,exports,prov)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('ERROR:',type(e).__name__,str(e),file=sys.stderr,flush=True);sys.exit(1)
