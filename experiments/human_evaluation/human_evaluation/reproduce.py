"""Reproduce the submitted V4 human-rating statistics; requires NumPy only."""
from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
METHODS=['sdxl','rpg','kolors','rpg_kolors','ragd','bireg']
DIMS=['attribute','spatial','quality']
B=20000; SEED=20260930

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_rows(name):
    return [json.loads(line) for line in (ROOT/'data'/name).read_text(encoding='utf-8').splitlines() if line.strip()]

prompt_rows=read_rows('prompts.jsonl')
samples={s['task_id']:s for s in prompt_rows}
require(len(prompt_rows)==len(samples)==89, 'Expected 89 unique tasks')
require(Counter(s['language'] for s in samples.values())=={'en':60,'zh':29}, 'Language counts differ')
rows=read_rows('ratings.jsonl')
values=defaultdict(dict); apps=defaultdict(set)
images={}; seen=set(); groups={}; workloads=defaultdict(set)
for r in rows:
    task, method, rater=r['task_id'],r['method'],r['rater_id']
    require(task in samples and method in METHODS, 'Unknown task or method')
    s=samples[task]
    require(r['prompt_id']==s['prompt_id'] and r['seed']==s['seed'], 'Prompt/seed mismatch')
    require(groups.setdefault(rater,r['group_id'])==r['group_id'], 'Inconsistent evaluator group')
    binding=(task,method,r['prompt_id'],r['seed'],r['group_id'])
    require(images.setdefault(r['image_code'],binding)==binding, 'Inconsistent image mapping')
    key=(task,method,rater)
    require(key not in seen, 'Duplicate assessment')
    seen.add(key); workloads[rater].add(task)
    require(set(r['ratings'])==set(DIMS), 'Unexpected dimensions')
    for dim in DIMS:
        score=r['ratings'][dim]; app=score is not None
        require((type(score) is int and 1<=score<=5) if app else dim!='quality', 'Invalid score')
        apps[task,dim].add(app)
        values[task,method,dim][rater]=score
raters=sorted(groups)
require(len(raters)==21 and len(images)==534 and len(rows)==1602, 'Unexpected coverage')
require(set(values)=={(t,m,d) for t in samples for m in METHODS for d in DIMS}, 'Missing task/method/dimension')
require(all(len(v)==1 for v in apps.values()), 'Inconsistent applicability')
require(all(len(v)==3 for v in values.values()), 'Expected three raters per image')
require(len({(v[0],v[1]) for v in images.values()})==534, 'Image IDs are not one-to-one')
for task in samples:
    panels={tuple(sorted(values[task,m,d])) for m in METHODS for d in DIMS}
    require(len(panels)==1, 'Evaluator panel differs across methods')
means=[]; pairs=[]; agreement=[]; strata_counts=[]
for li,lang in enumerate(['en','zh']):
    for di,dim in enumerate(DIMS):
        tasks=sorted(t for t,s in samples.items() if s['language']==lang and next(iter(apps[t,dim])))
        x=np.array([[np.mean(list(values[t,m,dim].values())) for m in METHODS] for t in tasks])
        rng=np.random.default_rng(np.random.SeedSequence([SEED,li,di]))
        boot=np.zeros((B,len(METHODS)))
        for stratum in sorted({samples[t]['sampling_stratum'] for t in tasks}):
            idx=np.array([i for i,t in enumerate(tasks) if samples[t]['sampling_stratum']==stratum])
            draws=rng.choice(idx,size=(B,len(idx)),replace=True)
            boot+=x[draws].sum(axis=1)/len(tasks)
            strata_counts.append(dict(language=lang,dimension=dim,stratum=stratum,n=len(idx)))
        for j,m in enumerate(METHODS):
            lo,hi=np.quantile(boot[:,j],[.025,.975],method='linear')
            means.append(dict(language=lang,dimension=dim,method=m,n=len(tasks),mean=float(x[:,j].mean()),sd=float(x[:,j].std(ddof=1)),lower=float(lo),upper=float(hi)))
            if m!='bireg':
                lo,hi=np.quantile(boot[:,-1]-boot[:,j],[.025,.975],method='linear')
                pairs.append(dict(language=lang,dimension=dim,baseline=m,n=len(tasks),delta=float((x[:,-1]-x[:,j]).mean()),lower=float(lo),upper=float(hi)))
        # Preserve E01--E21 identities, not a fabricated three-row panel.
        units=[(t,m) for t in tasks for m in METHODS]
        matrix=np.full((len(raters),len(units)),np.nan)
        for u,(t,m) in enumerate(units):
            for r,v in values[t,m,dim].items(): matrix[raters.index(r),u]=v
        # Independent coincidence-matrix cross-check of ordinal alpha.
        counts=np.array([(matrix.T==k).sum(axis=1) for k in range(1,6)]).T
        coincidences=sum((np.outer(c,c)-np.diag(c))/(sum(c)-1) for c in counts)
        marg=coincidences.sum(axis=0); expected=(np.outer(marg,marg)-np.diag(marg))/(marg.sum()-1)
        ranks=np.cumsum(marg)-marg/2; distance=(ranks[:,None]-ranks[None,:])**2
        a_check=1-(coincidences*distance).sum()/(expected*distance).sum()
        a=float(a_check)
        exact=sum(len(set(values[t,m,dim].values()))==1 for t,m in units)
        agreement.append(dict(language=lang,dimension=dim,alpha=a,n_images=len(units),n_scores=int(np.isfinite(matrix).sum()),unanimous_images=exact))


result=dict(snapshot='Submitted V4 rating values',bootstrap=dict(resamples=B,seed=SEED,method='stratified paired prompt-level percentile; pointwise 95% intervals',rng='PCG64, SeedSequence([seed, language_index, dimension_index])'),validation=dict(prompts=89,images=534,raters=21,image_assessments=len(rows),valid_scores=sum(a['n_scores'] for a in agreement),workloads=dict(Counter(len(v) for v in workloads.values()))),means=means,paired=pairs,agreement=agreement,strata_counts=strata_counts)
OUT.mkdir(exist_ok=True)
(OUT/'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
table=['# Human evaluation results', '', 'Submitted V4 values. Scores: 1–5; higher is better. SD is across image means; CI is the stratified prompt-bootstrap 95% interval.', '', '| Language | Dimension | Method | Prompts | Mean ± SD | 95% CI |', '|---|---|---|---:|---:|---|']
for r in means:
    table.append(f"| {r['language']} | {r['dimension']} | {r['method']} | {r['n']} | {r['mean']:.3f} ± {r['sd']:.3f} | [{r['lower']:.3f}, {r['upper']:.3f}] |")
table+=['', '## Inter-rater agreement', '', '| Language | Dimension | Ordinal Krippendorff alpha | Images | Applicable ratings |', '|---|---|---:|---:|---:|']
for r in agreement:
    table.append(f"| {r['language']} | {r['dimension']} | {r['alpha']:.6f} | {r['n_images']} | {r['n_scores']} |")
table+=['', '## Paired differences: BiReG minus baseline', '', '| Language | Dimension | Baseline | Mean difference | 95% CI |', '|---|---|---|---:|---|']
for r in pairs:
    table.append(f"| {r['language']} | {r['dimension']} | {r['baseline']} | {r['delta']:.3f} | [{r['lower']:.3f}, {r['upper']:.3f}] |")
(OUT/'table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
print(json.dumps(result['validation'],ensure_ascii=False))
print('Wrote results/statistics.json and results/table.md')
