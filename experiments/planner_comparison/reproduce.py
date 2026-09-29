#!/usr/bin/env python3
"""Recompute Section 4.6 tables from retained scores; Python standard library only."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ['deepseek-flash', 'deepseek-v4-pro', 'qwen3.8-flash', 'gpt-5.5']
CATS = ['color', 'shape', 'texture', 'spatial', 'non_spatial', 'complex']
def require(ok, message):
    if not ok:
        raise ValueError(message)
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def stats(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
def close(a,b):
    require(math.isclose(a,b,rel_tol=0,abs_tol=1e-12),'Summary mismatch')
def save(name, obj):
    (ROOT / 'results' / name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    for name,digest in read('data_sha256.json').items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Checksum mismatch: '+name)
    prompts=read('data/prompts_frozen.json')
    require(len(prompts)==152,'Expected 152 prompts')
    lookup={p['prompt_id']:p for p in prompts}
    require(len(lookup)==152,'Duplicate prompt ID')
    ens=read('data/english_image_scores.json'); zhs=read('data/chinese_checkpoint_scores.json')
    require(len(ens)==1224 and len(zhs)==600,'Incorrect total score count')
    eref=read('data/reference_english_summary.json'); zref=read('data/reference_chinese_summary.json')
    out={'english':{},'chinese':{}}
    bindings={}
    for model in MODELS:
        for lang,rows,seeds,n in [('en',ens,[2026,3407,5678],306),('zh',zhs,[1234,2468,42],150)]:
            rr=[r for r in rows if r['planner']==model]
            require(len(rr)==n,'Incorrect per-model count')
            pairs={(r['prompt_id'],r['seed']) for r in rr}
            expected={(p['prompt_id'],seed) for p in prompts if p['language']==lang for seed in seeds}
            require(pairs==expected and len(pairs)==len(rr),'Missing or duplicate prompt/seed')
            for r in rr:
                p=lookup[r['prompt_id']]
                require(r.get('caption',r.get('prompt'))==p['prompt'],'Prompt text mismatch')
                if lang=='en': require(r['category']==p['category'],'Category mismatch')
                else:
                    old=bindings.setdefault(r['prompt_id'],r['testpoints'])
                    require(old==r['testpoints'],'Chinese testpoint mismatch')
        er=[r for r in ens if r['planner']==model]
        out['english'][model]={}
        for cat in CATS:
            values=[]; seed_scores={}
            for seed in [2026,3407,5678]:
                cr=[r for r in er if r['category']==cat and r['seed']==seed]
                require(len(cr)==17,'Expected 17 English prompts per category/seed')
                require(all(math.isfinite(r['score']) for r in cr),'Nonfinite score')
                value=statistics.mean(r['score'] for r in cr); values.append(value);seed_scores[str(seed)]=value
            result=stats(values);result['seed_scores']=seed_scores
            for k in ['mean','sample_sd']:close(result[k],eref[model][cat][k])
            out['english'][model][cat]=result
        counts=defaultdict(lambda:[0,0])
        for r in zhs:
            if r['planner']!=model:continue
            require(len(r['testpoints'])==len(r['scores']),'Checkpoint length mismatch')
            for point,score in zip(r['testpoints'],r['scores']):
                require(score in [0,1],'Nonbinary checkpoint score')
                for level,dim in [('primary_dimensions',point.split('-')[0]),('sub_dimensions',point)]:
                    c=counts[(level,dim,r['seed'])];c[0]+=score;c[1]+=1
        out['chinese'][model]={}
        for level,dims in zref[model].items():
            out['chinese'][model][level]={}
            for dim,ref in dims.items():
                cs=[counts[(level,dim,s)] for s in [1234,2468,42]]
                require(all(c[1]>0 for c in cs),'Empty dimension')
                vals=[a/b for a,b in cs]; result=stats(vals)
                result['seed_scores']={str(s):v for s,v in zip([1234,2468,42],vals)}
                result['total_checkpoints']=sum(c[1] for c in cs)
                for k in ['mean','sample_sd']:close(result[k],ref[k])
                require(result['total_checkpoints']==ref['total_checkpoints'],'Denominator mismatch')
                out['chinese'][model][level][dim]=result
    (ROOT/'results').mkdir(exist_ok=True)
    save('summary.json',out)
    for lang,dims in [('english',CATS),('chinese',list(out['chinese'][MODELS[0]]['primary_dimensions']))]:
        scale=1 if lang=='english' else 100
        def val(model,dim):return out[lang][model][dim] if lang=='english' else out[lang][model]['primary_dimensions'][dim]
        with (ROOT/'results'/('table_'+lang+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['dimension','planner','mean','sample_sd','unit'])
            for dim in dims:
                for model in MODELS:
                    v=val(model,dim);w.writerow([dim,model,v['mean']*scale,v['sample_sd']*scale,'score' if scale==1 else 'percent'])
        lines=['| Dimension | '+' | '.join(MODELS)+' |','|---|'+ '---:|'*4]
        for dim in dims:
            digits=3 if scale==1 else 2
            lines.append('| '+dim+' | '+' | '.join(f"{val(m,dim)['mean']*scale:.{digits}f} ± {val(m,dim)['sample_sd']*scale:.{digits}f}" for m in MODELS)+' |')
        (ROOT/'results'/('table_'+lang+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save('audit_summary.json',{'status':'PASS','english_prompts':102,'chinese_prompts':50,'english_images':1224,'chinese_images':600,'planners':MODELS,'generation_seeds_per_prompt':3,'API_calls':0,'GPU_calls':0,'scope':'Stored score aggregation and bindings; no image re-evaluation or PNG-byte verification.'})
    print('PASS: 1224 English + 600 Chinese score records; tables reproduced; no API/GPU calls.')
if __name__=='__main__':main()
