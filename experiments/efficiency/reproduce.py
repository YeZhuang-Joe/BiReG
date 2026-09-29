#!/usr/bin/env python3
"""Recompute the paper table from frozen request/image timings; standard library only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
MODELS=['deepseek-flash','deepseek-v4-pro','qwen3.8-flash','gpt-5.5']
def read(name): return json.loads((ROOT/'data'/name).read_text(encoding='utf-8'))
def require(ok,msg):
    if not ok:raise ValueError(msg)
def same(a,b):require(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),'Statistics mismatch')
def main():
    hashes=json.loads((ROOT/'data_sha256.json').read_text())
    for n,h in hashes.items():
        require(hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()==h,'Changed data: '+n)
    requests=read('per_request_timings.json');images=read('per_image_timings.json')
    prompts=read('per_prompt_timings.json');reference=read('summary.json')
    require((len(requests),len(images),len(prompts))==(631,1824,608),'Record count mismatch')
    req=defaultdict(list);img=defaultdict(list)
    for r in requests:req[(r['model'],r['language'],r['prompt_id'])].append(r)
    for r in images:img[(r['model'],r['language'],r['prompt_id'])].append(r)
    keys={(r['model'],r['language'],r['prompt_id']) for r in prompts}
    require(len(keys)==608 and keys==set(req)==set(img),'Prompt coverage mismatch')
    vals=defaultdict(list)
    for r in prompts:
        k=(r['model'],r['language'],r['prompt_id']);rs=req[k];ims=img[k]
        seeds={2026,3407,5678} if r['language']=='en' else {1234,2468,42}
        require(len(ims)==3 and {i['seed'] for i in ims}==seeds,'Seed coverage mismatch')
        require(len({x['path'] for x in rs})==len(rs),'Duplicate request')
        p=sum(x['request_seconds']+x.get('parse_seconds',0) for x in rs)
        d=st.mean(x['generation_stage_seconds'] for x in ims)
        require(all(math.isfinite(x) and x>=0 for x in [p,d]),'Invalid duration')
        same(p,r['planning_active_seconds']);same(d,r['generation_stage_mean3_seconds'])
        vals[k[:2]].append((p,d))
    rows=[]
    for lang,n in [('en',102),('zh',50)]:
        ids=[{k[2] for k in keys if k[:2]==(m,lang)} for m in MODELS]
        require(all(s==ids[0] and len(s)==n for s in ids),'Cross-planner prompt mismatch')
        for m in MODELS:
            ps,ds=zip(*vals[(m,lang)])
            row=dict(language=lang,planner=m,prompts=n,planning_mean=st.mean(ps),planning_sd=st.stdev(ps),
                     planning_median=st.median(ps),generation_mean=st.mean(ds),generation_sd=st.stdev(ds))
            old=reference['models'][m]['languages'][lang]
            for prefix,key in [('planning','planning_active_seconds'),('generation','generation_stage_mean3_seconds')]:
                same(row[prefix+'_mean'],old[key]['mean']);same(row[prefix+'_sd'],old[key]['sample_sd'])
            rows.append(row)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    with (out/'table47.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=[r'\begin{table*}[t]',r'\centering',
      r'\caption{Observed planning and generation times (seconds). Values are means $\pm$ sample standard deviations across 102 English or 50 Chinese prompts. Generation time is first averaged over three seeds per prompt. Planning includes failed requests, retries, and supplemental requests; explicit waiting and model loading are excluded. The GPT identifier is provided by a third-party service.}',
      r'\label{tab:planner_efficiency}',r'\begin{tabular}{llcc}',r'\hline',
      r'Language & Planner identifier & Planning time & Generation time \\',r'\hline']
    for r in rows:
        lines.append(f"{'English' if r['language']=='en' else 'Chinese'} & "+r'\texttt{'+r['planner']+'} & '+
          f"${r['planning_mean']:.3f} \\pm {r['planning_sd']:.3f}$ & ${r['generation_mean']:.3f} \\pm {r['generation_sd']:.3f}$ \\")
    lines += [r'\hline',r'\end{tabular}',r'\end{table*}']
    # Ensure row terminators are two literal backslashes.
    lines=[s+'\\' if s.endswith('\\') and not s.endswith('\\\\') else s for s in lines]
    (out/'table47.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=dict(requests=len(requests),images=len(images),prompt_model_pairs=len(prompts),
        recorded_explicit_wait_seconds=sum(r['observed_wait_seconds'] for r in read('explicit_wait_records.json')),
        per_model={m:dict(requests=sum(r['model']==m for r in requests),
            http_errors=sum(r['model']==m and r['status']=='http_error' for r in requests),
            supplemental_requests=sum(r['model']==m and r['phase']=='supplemental' for r in requests)) for m in MODELS})
    (out/'audit_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 631 requests, 1824 images, 608 prompt-model pairs; table matches frozen summary.')
    print(out)
if __name__=='__main__':main()
