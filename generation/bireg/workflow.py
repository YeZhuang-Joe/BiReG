#!/usr/bin/env python3
"""Auto language routing -> API plan -> BiReG image. No scoring.

Private credentials are read from private/api_config.json.
Use a separate output directory for each prompt/configuration.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]

def read_private(path):
    try:data=json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:raise ValueError('Private API config missing. Copy configs/api.example.json to private/api_config.json and fill in your key.')
    except (ValueError,UnicodeError):raise ValueError('Invalid API configuration JSON; check formatting locally.')
    if not isinstance(data,dict):raise ValueError('API configuration must be an object')
    allowed={'api_key','endpoint','model','temperature','max_tokens','thinking','max_attempts','timeout_seconds','interval_seconds'}
    if set(data)-allowed:raise ValueError('Unknown API configuration field; see example')
    key=data.get('api_key')
    if not isinstance(key,str) or not key.strip() or key.strip()=='REPLACE_WITH_YOUR_KEY':raise ValueError('Fill api_key in the local private configuration first.')
    p={k:v for k,v in data.items() if k!='api_key'}
    for name in ('endpoint','model','temperature','max_tokens','max_attempts','timeout_seconds','interval_seconds'):
        if name not in p:raise ValueError('API configuration is missing required fields')
    u=urlparse(p['endpoint'])
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:raise ValueError('Require an HTTPS endpoint without credentials or URL query')
    if type(p['temperature']) not in (int,float) or not math.isfinite(p['temperature']) or p['temperature']<0:raise ValueError('Invalid temperature')
    for n in ('max_tokens','max_attempts'):
        if type(p[n]) is not int or p[n]<1:raise ValueError('Invalid request budget')
    for n in ('timeout_seconds','interval_seconds'):
        if type(p[n]) not in (int,float) or not math.isfinite(p[n]) or p[n]<0:raise ValueError('Invalid timeout/interval')
    if p['timeout_seconds']==0:raise ValueError('Timeout must be positive')
    if not isinstance(p['model'],str) or not p['model'].strip():raise ValueError('Exact model identifier required')
    # No credential or credential hash appears in the persisted public configuration.
    p.update(key_env='BIREG_WORKFLOW_PRIVATE_API_KEY',repeats=1)
    return key.strip(),p

def run(args):
    from bireg.common import save,save_lines,filehash,digest,lock,load
    from bireg.planning import read_experiment,planning,frozen_plans,inspect_attempt,record_dir
    from bireg.rendering import tasks_for,completed_task
    import bireg.rendering as rendering
    prompt=args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding='utf-8')
    if not prompt.strip():raise ValueError('Prompt is empty')
    from .language import detect
    routing=detect(prompt,args.language)
    if args.detect_only:
        print(json.dumps(routing,ensure_ascii=False,indent=2));return
    if routing['requires_manual_language']:
        raise ValueError('Mixed or insufficient language evidence: specify --language zh or --language en. No API/GPU called.')
    language=routing['selected_language']
    from .common import source_hashes
    from .rendering import checkpoint_info
    source_hashes()
    checkpoint_info(args.model_path)
    key,planner=read_private(args.api_config)
    if any(t in prompt.upper() for t in ('BREAK','ADDBASE','ADDCOL','ADDROW','ADDCOMM')):raise ValueError('Prompt contains a reserved renderer control token')
    if args.seed is not None and (type(args.seed) is not int or not 0<args.seed<2**32):raise ValueError('Seed must be a positive uint32 integer')
    defaults=load(ROOT/'configs/generation.json')
    generation=copy.deepcopy(defaults['generation'])
    seed=args.seed if args.seed is not None else generation[language]['seeds'][0]
    generation[language]['seeds']=[seed]
    output=Path(args.output).resolve()
    # Keep generated artifacts separate from source files.
    if output==ROOT or ROOT in output.parents and output.parts[len(ROOT.parts)] in ('bireg','vendor','templates','configs','private','tests','docs','licenses'):
        raise ValueError('Use a separate output folder, e.g. outputs/demo_zh')
    pid='user_'+digest(prompt.encode())[:20]
    row={'prompt_id':pid,'prompt':prompt,'language':language,'language_routing':routing}
    config=dict(experiment_id='bireg_generation_v1',purpose='single prompt to image; no scoring',
        prompts=str(output/'input.jsonl'),run_dir=str(output/'run'),max_regions=7,planner=planner,generation=generation)
    launch=dict(workflow_sha256=filehash(__file__),
        model_path=str(Path(args.model_path).resolve()),language_routing=routing,
        detector_sha256=filehash(ROOT/'bireg/language.py'),
        output_schema='bireg-generation-v1',scoring_enabled=False)
    if args.check:
        if (output/'workflow.json').exists() and load(output/'workflow.json')!=launch:raise ValueError('Existing workflow differs; use a new output folder')
        print(json.dumps(dict(check='PASS',language=language,language_routing=routing,seed=seed,width=generation[language]['width'],height=generation[language]['height'],output=str(output),credential_loaded=True,API_calls=0,GPU_calls=0),ensure_ascii=False,indent=2));return
    print('LANGUAGE:',language,'|',routing['selection_source'],'| H=',routing['han_characters'],'E=',routing['english_words'])
    # Validate existing files before any API/GPU call. Credentials never enter these files.
    save(output/'workflow.json',launch);save_lines(output/'input.jsonl',[row]);save(output/'config.json',config)
    cfg,rows,templates,run_dir,snapshot=read_experiment(output/'config.json')
    if (run_dir/'STOP').exists():raise ValueError('STOP file exists; workflow paused')
    first_accepted=any((inspect_attempt(record_dir(run_dir,pid,1)/f'attempt_{i:02d}') or {}).get('status')=='accepted' for i in range(1,planner['max_attempts']+1))
    if not first_accepted:
        # Compatibility with the retained planner: inject only into this process,
        # then restore it before model loading; no shell export or file write.
        envname=planner['key_env'];old=os.environ.get(envname)
        try:
            os.environ[envname]=key
            planning(cfg,rows,templates,run_dir,snapshot,execute=True)
        finally:
            if old is None:os.environ.pop(envname,None)
            else:os.environ[envname]=old
    del key
    with lock(run_dir):
        save(run_dir/'plans.frozen.json',frozen_plans(cfg,rows,run_dir))
    rendering.generate(cfg,rows,run_dir,Path(args.model_path),execute=True,retry_failed=args.retry_failed)
    task=tasks_for(cfg,rows,run_dir)[0];result=completed_task(run_dir,task)
    if result is None:raise ValueError('No completed image; inspect retained run records')
    src=run_dir/result['image_path'];dst=output/'image.png'
    if dst.exists():
        if filehash(dst)!=result['image_sha256']:raise ValueError('Existing image.png differs; refusing to overwrite')
    else:
        # Exclusive creation, leaving the canonical generation image unchanged.
        with dst.open('xb') as f:
            f.write(src.read_bytes());f.flush();os.fsync(f.fileno())
    save(output/'output.json',dict(image='image.png',image_sha256=filehash(dst),task_id=task['task_id'],task_sha256=task['task_sha256'],canonical_image=str(src),language_routing=routing,record=result))
    print('IMAGE:',str(dst))
    print('RECORD:',str(output/'output.json'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    prompts=ap.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt');prompts.add_argument('--prompt-file',type=Path)
    ap.add_argument('--language',choices=['auto','en','zh'],default='auto')
    ap.add_argument('--detect-only',action='store_true',help='Only inspect language routing; no API config or GPU needed')
    ap.add_argument('--seed',type=int)
    ap.add_argument('--output',type=Path,help='Required for generation/check; use a separate folder per prompt/configuration')
    ap.add_argument('--api-config',type=Path,default=ROOT/'private/api_config.json')
    ap.add_argument('--model-path',type=Path,default=Path('/root/autodl-tmp/weights/Kolors'))
    ap.add_argument('--check',action='store_true',help='Validate local configuration only; no writes or API/GPU')
    ap.add_argument('--retry-failed',action='store_true',help='Explicitly retry a previous failed/incomplete image attempt')
    args=ap.parse_args()
    if not args.detect_only and args.output is None:ap.error('--output is required unless --detect-only is used')
    run(args)

if __name__=='__main__':
    try:main()
    except (ValueError,RuntimeError,OSError,KeyError) as e:
        print('ERROR:',str(e),file=sys.stderr);sys.exit(2)
