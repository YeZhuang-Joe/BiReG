import getpass
import json
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, build_opener
from .common import ROOT, code_hashes, digest, encoded, environment, filehash, jsonlines, load, lock, now, safe_id, save, source_hashes
from .layout import check_layout

def read_experiment(path):
    path=Path(path).resolve();cfg=load(path)
    # All relative paths in a config are relative to the repository root.
    prompts=Path(cfg['prompts']);prompts=prompts if prompts.is_absolute() else ROOT/prompts
    rows=jsonlines(prompts);ids=[]
    for row in rows:
        ids.append(safe_id(row['prompt_id']))
        if row['language'] not in ('en','zh') or not isinstance(row['prompt'],str) or not row['prompt'].strip():raise ValueError('Invalid language/prompt')
        if 'BREAK' in row['prompt']:raise ValueError('Reserved BREAK in input caption')
        g=cfg['generation'][row['language']]
        for field in ('width','height'):
            if type(g[field]) is not int or g[field]<64 or g[field]%64:raise ValueError('Image dimensions must be positive multiples of 64')
        if g['scheduler'] not in ('EulerDiscreteScheduler','DPMSolverMultistepScheduler'):raise ValueError('Unsupported scheduler')
        if not math.isfinite(g['lambda_global']) or not 0<=g['lambda_global']<=1:raise ValueError('Invalid global fusion weight')
        if type(g['steps']) is not int or g['steps']<1 or not math.isfinite(g['guidance_scale']) or g['guidance_scale']<=1:raise ValueError('Require steps>=1 and CFG>1 for this backend')
        if not g['seeds'] or len(g['seeds'])!=len(set(g['seeds'])) or any(type(s) is not int or s<=0 or s>=2**32 for s in g['seeds']):raise ValueError('Unique positive uint32 seeds required')
        if g.get('dtype')!='float16':raise ValueError('This workflow supports the retained float16 renderer configuration')
    if not rows or len(set(ids))!=len(ids):raise ValueError('Empty or duplicate prompt IDs')
    p=cfg['planner']
    u=urlparse(p['endpoint'])
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:raise ValueError('Require HTTPS endpoint without credentials/query')
    allowed={'endpoint','model','key_env','temperature','max_tokens','thinking','repeats','max_attempts','timeout_seconds','interval_seconds'}
    if set(p)-allowed:raise ValueError('Unknown planner configuration fields')
    for name in ('repeats','max_attempts','max_tokens'):
        if type(p[name]) is not int or p[name]<1:raise ValueError('Invalid planner budget')
    if p['timeout_seconds']<=0 or p['interval_seconds']<0:raise ValueError('Invalid timeout/interval')
    if not math.isfinite(p['temperature']) or p['temperature']<0:raise ValueError('Invalid temperature')
    if not isinstance(p['model'],str) or not p['model'].strip():raise ValueError('Require an exact model identifier')
    if cfg.get('max_regions') is not None and (type(cfg['max_regions']) is not int or cfg['max_regions']<1):raise ValueError('Invalid region budget')
    run=Path(cfg['run_dir']);run=run if run.is_absolute() else ROOT/run
    templates={lang:(ROOT/'templates'/f'template_{lang}.txt').read_text(encoding='utf-8') for lang in ('en','zh')}
    snapshot=dict(schema='bireg_generation_v1',config=cfg,prompts=rows,templates=templates,source_hashes=source_hashes(),code_hashes=code_hashes(),prompts_sha256=filehash(prompts))
    if (run/'experiment.json').exists() and load(run/'experiment.json')!=snapshot:raise ValueError('Run configuration/code/input changed: use a new run_dir')
    return cfg,rows,templates,run,snapshot

def record_dir(run,pid,repeat):return run/'planning'/pid/f'repeat_{repeat:02d}'

def response_result(raw,row,cfg):
    data=json.loads(raw);choice=data['choices'][0];msg=choice['message']
    content=msg.get('content');base=dict(returned_model=data.get('model'),response_id=data.get('id'),usage=data.get('usage'),finish_reason=choice.get('finish_reason'),content=content)
    if choice.get('finish_reason')!='stop' or msg.get('refusal') or not isinstance(content,str):return dict(base,status='response_rejected',error='Incomplete, refused, or non-text response')
    g=cfg['generation'][row['language']]
    try:parsed=check_layout(content,row['prompt'],g['width'],g['height'],cfg['max_regions'])
    except (ValueError,IndexError,KeyError,ZeroDivisionError) as e:return dict(base,status='parse_failed',error=str(e))
    return dict(base,status='accepted',parsed=parsed)

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def inspect_attempt(directory):
    started=directory/'started.json';result=directory/'result.json';raw=directory/'response.bin'
    if result.exists():
        r=load(result)
        if filehash(directory/'request.json')!=r['request_sha256']:raise ValueError('Saved request hash mismatch')
        if r.get('response_sha256') and (not raw.exists() or filehash(raw)!=r['response_sha256']):raise ValueError('Saved response hash mismatch')
        return r
    if started.exists():return {'status':'interrupted_unknown'}
    return None

def attempt(row,repeat,index,run,cfg,templates,key,previous=None):
    d=record_dir(run,row['prompt_id'],repeat)/f'attempt_{index:02d}';p=cfg['planner']
    messages=[{'role':'system','content':templates[row['language']]},{'role':'user','content':row['prompt']}]
    if previous and previous.get('content'):
        messages += [{'role':'assistant','content':previous['content']},{'role':'user','content':'The previous response could not be executed: '+previous.get('error','invalid response')+'. Return the complete two-line plan again, preserving the original caption. Do not include commentary.'}]
    body={k:p[k] for k in ('model','temperature','max_tokens')};body.update(messages=messages,stream=False)
    if 'thinking' in p:body['thinking']=p['thinking']
    payload=encoded(body);save(d/'request.json',body)
    save(d/'started.json',dict(at=now(),request_sha256=digest(payload),prompt_id=row['prompt_id'],repeat=repeat,attempt=index))
    req=Request(p['endpoint'],data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'},method='POST')
    opener=build_opener(NoRedirect());raw=None;http=None
    start=time.perf_counter()
    try:
        with opener.open(req,timeout=p['timeout_seconds']) as response:
            raw=response.read();http=response.status
        elapsed=time.perf_counter()-start
    except HTTPError as e:
        elapsed=time.perf_counter()-start;http=e.code
        result=dict(status='http_error',http_status=http,error='HTTP request failed; no automatic transport retry',request_seconds=elapsed)
    except Exception as e:
        elapsed=time.perf_counter()-start
        result=dict(status='transport_unknown',error=type(e).__name__,request_seconds=elapsed)
    else:
        # Bodies are retained separately; authorization headers are never serialized.
        (d/'response.bin').write_bytes(raw)
        parse_start=time.perf_counter()
        try:result=response_result(raw,row,cfg)
        except (ValueError,KeyError,IndexError,TypeError) as e:result=dict(status='response_rejected',error=type(e).__name__)
        result.update(parse_seconds=time.perf_counter()-parse_start,request_seconds=elapsed,http_status=http,response_sha256=digest(raw))
    result.update(at=now(),prompt_id=row['prompt_id'],repeat=repeat,attempt=index,requested_model=p['model'],request_sha256=digest(payload))
    save(d/'result.json',result)
    print(f"{row['prompt_id']} repeat {repeat} attempt {index}: {result['status']} | HTTP {elapsed:.3f}s",flush=True)
    return result

def planning(cfg,rows,templates,run,snapshot,execute=False,limit=None):
    budget=len(rows)*cfg['planner']['repeats']*cfg['planner']['max_attempts']
    if not execute:
        print(json.dumps(dict(check='PASS',prompts=len(rows),max_total_requests=budget,run_dir=str(run),GPU_calls=0),indent=2));return
    key=os.environ.get(cfg['planner']['key_env']) or getpass.getpass('DeepSeek/API key (hidden): ')
    if not key.strip():raise ValueError('API key is empty')
    with lock(run):
        save(run/'experiment.json',snapshot)
        if not (run/'planning_environment.json').exists():save(run/'planning_environment.json',environment())
        count=0
        # Interleave language records within each repeat; no cached replies count as new measurements.
        for repeat in range(1,cfg['planner']['repeats']+1):
            for row in rows:
                previous=None
                for index in range(1,cfg['planner']['max_attempts']+1):
                    d=record_dir(run,row['prompt_id'],repeat)/f'attempt_{index:02d}'
                    r=inspect_attempt(d)
                    if r is None:
                        if limit is not None and count>=limit:return
                        if (run/'STOP').exists():print('STOP file found.');return
                        if count:time.sleep(cfg['planner']['interval_seconds'])
                        r=attempt(row,repeat,index,run,cfg,templates,key,previous);count+=1
                    if r['status']=='accepted':break
                    if r['status'] in ('http_error','transport_unknown','interrupted_unknown'):
                        print('Uncertain/network failure retained; stopped. Inspect records before choosing a new run.');return
                    previous=r

def frozen_plans(cfg,rows,run):
    plans=[];failed=[]
    # Repeat 1 alone supplies the generation plan. Later repeats only measure repeated planning.
    for row in rows:
        selected=None
        for index in range(1,cfg['planner']['max_attempts']+1):
            d=record_dir(run,row['prompt_id'],1)/f'attempt_{index:02d}';r=inspect_attempt(d)
            if r and r['status']=='accepted':
                checked=response_result((d/'response.bin').read_bytes(),row,cfg)
                if checked['status']!='accepted' or checked['parsed']!=r['parsed']:raise ValueError('Saved plan cannot be revalidated')
                selected=dict(prompt=row,plan=r['parsed'],raw_response=str((d/'response.bin').relative_to(run)),raw_sha256=r['response_sha256'],requested_model=r['requested_model'],returned_model=r.get('returned_model'),repeat=1,attempt=index);break
        if selected:plans.append(selected)
        else:failed.append(row['prompt_id'])
    if failed:raise ValueError('Cannot freeze a complete generation set; unresolved IDs: '+', '.join(failed))
    return dict(schema='bireg_frozen_plans_v1',experiment_sha256=filehash(run/'experiment.json'),plans=plans,semantic_validation=False,selection='first executable response from repeat 1; no posthoc repair/fallback')
