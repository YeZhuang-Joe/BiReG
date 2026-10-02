#!/usr/bin/env python3
"""Offline translated-ZH baseline generation. One process/pipeline per execution mode."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
SEEDS = [1234, 2468, 42]
METHODS = ['sdxl', 'rpg', 'ragd']
PYTHONS = {'sdxl': '/root/miniconda3/envs/RPG/bin/python',
           'rpg': '/root/miniconda3/envs/RPG/bin/python',
           'ragd': '/root/autodl-tmp/ragd_workspace/.venv/bin/python'}
FIELDS = ('HB_m_offset_list', 'HB_n_offset_list', 'HB_m_scale_list', 'HB_n_scale_list')

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def canonical(v): return json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def clean(v):
    if isinstance(v, float) and not math.isfinite(v): return {'nonfinite_float': str(v)}
    if isinstance(v, dict): return {str(k): clean(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(clean(v), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)
def freeze(p, v):
    if p.exists():
        if read(p) != clean(v): raise ValueError('Frozen configuration changed: ' + str(p))
    else: write(p, v)
def require(ok, msg):
    if not ok: raise ValueError(msg)

def check_bundle():
    for name, expected in read(HERE/'package_sha256.json').items():
        require(sha(HERE/name) == expected, 'Package file changed: ' + name)
    subprocess.run([sys.executable, str(HERE/'inputs/verify_release.py')], check=True)

def grid_check(plan, width, height):
    split = plan.get('SR_hw_split_ratio', plan.get('split_ratio'))
    rr = [[float(x) for x in r.split(',')] for r in split.split(';')]
    heights = [1.] if len(rr) == 1 else [r[0] for r in rr]
    widths = [rr[0]] if len(rr) == 1 else [r[1:] or [1.] for r in rr]
    def extents(ws, n):
        require(all(math.isfinite(x) and x > 0 for x in ws), 'Invalid ratio')
        cursor=0.; edges=[0]
        for x in ws:
            cursor += x/sum(ws); edges.append(int(n*cursor))
        edges[-1] = n
        return [b-a for a,b in zip(edges, edges[1:])]
    # Include coarsest SDXL cross-attention grid and FLUX packed latent grid.
    scales = [16] if 'SR_hw_split_ratio' in plan else [8,16,32,64]
    for scale in scales:
        require(min(extents(heights,height//scale)) > 0, 'Empty row at grid scale '+str(scale))
        for ws in widths: require(min(extents(ws,width//scale)) > 0, 'Empty column')
    hb=[]
    if 'HB_prompt_list' in plan:
        for x,y,w,h in zip(*(plan[f] for f in FIELDS)):
            gx,gy,gw,gh = int(x*width//16),int(y*height//16),int(w*width//16),int(h*height//16)
            require(gw>0 and gh>0 and gx+gw<=width//16 and gy+gh<=height//16, 'Invalid rounded HB box')
            hb.append([gx,gy,gw,gh])
    return {'HB_grid_boxes_xywh': hb, 'overlap_policy': 'preserve original list order; no geometry edits'}

def tasks_for(method):
    records=rows(HERE/'inputs'/('ragd_instructions_150.frozen.jsonl' if method=='ragd' else 'rpg_instructions_150.frozen.jsonl'))
    tasks=[]
    for r in records:
        mode = 'plain' if method=='sdxl' else r['generation_mode']
        plan = None if method=='sdxl' else r['plan']
        for seed in SEEDS:
            tasks.append(dict(task_id='%s_%s_seed_%s'%(method,r['prompt_id'],seed), method=method,
                prompt_id=r['prompt_id'], prompt_en=r['prompt_en'], prompt_zh=r['prompt_zh'],
                annotations=r['annotations'], seed=seed, generation_mode=mode, plan=plan,
                frozen_instruction_sha256=digest(r),
                planning_provenance=r['planning_provenance'] if method!='sdxl' else None,
                fallback=r['fallback'] if method=='rpg' else None))
    require(len(tasks)==450 and len({t['task_id'] for t in tasks})==450, 'Task coverage mismatch')
    return tasks

def inventory(model):
    require((model/'model_index.json').is_file(), 'Missing model: '+str(model))
    configs={str(p.relative_to(model)):sha(p) for p in model.rglob('*.json') if '.cache' not in p.parts}
    files=[{'path':str(p.relative_to(model)),'bytes':p.stat().st_size} for p in sorted(model.rglob('*'))
           if p.is_file() and '.cache' not in p.parts and p.suffix in ('.safetensors','.bin','.model','.txt')]
    require(any(x['path'].endswith(('.safetensors','.bin')) for x in files), 'No model weights')
    return {'config_hashes':configs,'files':files,'weight_content_hashes_verified':False}

def configuration(method, a, tasks):
    settings=read(HERE/'generation_settings.json')[method]
    model=(a.flux_model if method=='ragd' else a.sdxl_model).resolve()
    for t in tasks:
        if t['plan']: grid_check(t['plan'],settings['width'],settings['height'])
    provenance=read(HERE/'source_provenance.json')
    if method=='ragd':
        for name,h in provenance['ragd_source_sha256'].items():
            require(sha(a.ragd_repo/name)==h, 'RAGD source mismatch: '+name)
    require(Path(PYTHONS[method]).is_file(), 'Missing interpreter: '+PYTHONS[method])
    return {'schema':'bireg_zh150_generation_v1','method':method,'tasks_sha256':digest(tasks),
        'generation':settings,'seeds':SEEDS,'task_count':450,'model_path':str(model),
        'model_inventory':inventory(model),'source_provenance':provenance,
        'runner_sha256':sha(Path(__file__).resolve()),'API_calls':0,
        'timing_definition':'CUDA-synchronized pipeline call including encoding and decoding; excludes model loading and image/record writes',
        'failure_policy':'Stop and retain attempts. At most one explicit retry with unchanged configuration.'}

def paths(out,t):
    image=out/'images'/t['prompt_id']/('seed_%s.png'%t['seed'])
    return image,image.with_suffix('.json'),out/'attempts'/t['task_id']

def completed(out,t,conf):
    image,sidecar,_=paths(out,t)
    if not sidecar.exists(): return False
    meta=read(sidecar)
    require(meta.get('status')=='completed' and meta.get('task_sha256')==digest(t) and meta.get('configuration_sha256')==digest(conf), 'Metadata mismatch: '+str(sidecar))
    require(image.is_file() and sha(image)==meta['image_sha256'],'Missing/changed PNG: '+str(image))
    from PIL import Image
    with Image.open(image) as im:
        require(im.size==(conf['generation']['width'],conf['generation']['height']) and im.mode=='RGB','PNG size/mode mismatch')
        im.verify()
    return True

def select_smoke(tasks,method):
    ids={tasks[0]['prompt_id']}
    if method=='rpg': ids.update(['UGB-ZH-110','UGB-ZH-278'])
    if method=='ragd': ids.update(['UGB-ZH-004','UGB-ZH-493'])
    return [t for t in tasks if t['seed']==1234 and t['prompt_id'] in ids]

def load_pipeline(method,mode,a,settings):
    import torch
    require(torch.cuda.is_available(),'CUDA unavailable')
    torch.set_num_threads(8)
    if method=='ragd':
        require(torch.cuda.is_bf16_supported(),'BF16 unavailable')
        os.chdir(a.ragd_repo);sys.path.insert(0,str(a.ragd_repo))
        from RAG_pipeline_flux import RAG_FluxPipeline, FluxTransformer2DModel
        transformer=FluxTransformer2DModel.from_pretrained(str(a.flux_model),subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,use_safetensors=True)
        pipe=RAG_FluxPipeline.from_pretrained(str(a.flux_model),transformer=transformer,torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,use_safetensors=True)
        pipe.enable_model_cpu_offload(gpu_id=0)
    else:
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        cls=StableDiffusionXLPipeline
        if mode=='regional':
            sys.path.insert(0,str(HERE/'native_source'))
            from diffusers.utils import is_invisible_watermark_available
            if is_invisible_watermark_available():
                sys.modules['native_source.watermark']=importlib.import_module('diffusers.pipelines.stable_diffusion_xl.watermark')
            cls=importlib.import_module('native_source.RegionalDiffusion_xl').RegionalDiffusionXLPipeline
        pipe=cls.from_pretrained(str(a.sdxl_model),torch_dtype=torch.float16,use_safetensors=True,local_files_only=True).to('cuda')
        pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
        pipe.enable_xformers_memory_efficient_attention()
    if hasattr(pipe,'torch_fix_seed'):
        original=pipe.torch_fix_seed
        def compatible_seed(self,seed=42):
            fn=torch.use_deterministic_algorithms
            try: original(seed=seed)
            finally: torch.use_deterministic_algorithms=fn
        pipe.torch_fix_seed=types.MethodType(compatible_seed,pipe)
    return pipe

def generate_one(pipe,t,settings):
    import torch
    common=dict(width=settings['width'],height=settings['height'],num_inference_steps=settings['num_inference_steps'],guidance_scale=settings['guidance_scale'],num_images_per_prompt=1)
    if t['method']=='ragd':
        kw={k:t['plan'][k] for k in ('SR_hw_split_ratio','SR_prompt','HB_prompt_list')+FIELDS}
        return pipe(**common,**kw,prompt=t['prompt_en'],seed=t['seed'],HB_replace=settings['HB_replace'],SR_delta=settings['SR_delta'],max_sequence_length=512)
    common.update(negative_prompt='',generator=torch.Generator(device='cuda').manual_seed(t['seed']))
    if t['generation_mode']=='regional':
        return pipe(**common,prompt=t['plan']['regional_prompt'],split_ratio=t['plan']['split_ratio'],base_prompt=t['prompt_en'],base_ratio=settings['base_ratio'],batch_size=1,seed=t['seed'])
    # Standard SDXL class in a separate process: no residual RPG attention hooks.
    return pipe(**common,prompt=t['prompt_en'])

def worker(a):
    method=a.method; tasks=tasks_for(method); conf=configuration(method,a,tasks)
    out=a.output_root/method;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze(out/'configuration.json',conf);freeze(out/'tasks.frozen.json',tasks)
    done={t['task_id'] for t in tasks if completed(out,t,conf)}
    selected=select_smoke(tasks,method) if a.smoke else tasks
    pending=[t for t in selected if t['generation_mode']==a.worker_mode and t['task_id'] not in done]
    print('VERIFIED %s: %d/450 completed; %d selected pending'%(method,len(done),len(pending)),flush=True)
    if not pending: return
    for t in pending:
        image,sidecar,ar=paths(out,t); old=list(ar.glob('attempt_*'))
        require(not old or (a.retry_failed and len(old)<2),'Unfinished task retained: '+t['task_id']+'; inspect and use --retry-failed once')
        require(not ((image.exists() or sidecar.exists()) and not old),'Untracked image exists')
    require(shutil.disk_usage(out).free>3*2**30,'Less than 3 GiB free')
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']: os.environ[key]='1'
    import torch
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    sf=out/'sessions'/(stamp+'.json')
    session={'started_at':now(),'mode':a.worker_mode,'status':'loading','completed_this_session':0,
             'python':sys.version,'executable':sys.executable,
             'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','diffusers','transformers','accelerate','safetensors']}}
    write(sf,session)
    try:
        start=time.perf_counter();pipe=load_pipeline(method,a.worker_mode,a,conf['generation'])
        session.update(load_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(0),status='generating')
        write(sf,session)
        freeze(out/('scheduler_'+a.worker_mode+'.json'),{'class':type(pipe.scheduler).__name__,'config':dict(pipe.scheduler.config)})
        for t in pending:
            if (a.output_root/'STOP').exists():
                session['status']='stopped_between_images';print('STOP file found',flush=True);return
            require(shutil.disk_usage(out).free>3*2**30,'Less than 3 GiB free')
            image,sidecar,ar=paths(out,t);ar.mkdir(parents=True,exist_ok=True)
            attempt=ar/('attempt_%02d'%(len(list(ar.glob('attempt_*')))+1));attempt.mkdir()
            for old in [image,sidecar]:
                if old.exists(): old.rename(attempt/('previous_'+old.name))
            meta={'status':'started','started_at':now(),'task':t,'task_sha256':digest(t),'configuration_sha256':digest(conf),
                  'generation':conf['generation'],'session':stamp}
            write(attempt/'record.json',meta)
            try:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                result=generate_one(pipe,t,conf['generation']);torch.cuda.synchronize()
                elapsed=time.perf_counter()-start;im=result.images[0]
                require(im.size==(1536,1024) and im.mode=='RGB','Unexpected image size/mode')
                tmp=attempt/'image.png';im.save(tmp);image.parent.mkdir(parents=True,exist_ok=True);tmp.replace(image)
                meta.update(status='completed',finished_at=now(),generation_seconds=elapsed,image_sha256=sha(image),
                            image_bytes=image.stat().st_size,cuda_peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
                write(attempt/'record.json',meta);write(sidecar,meta)
                done.add(t['task_id']);session['completed_this_session']+=1;write(sf,session)
                print('DONE %s %d/450 | %.2fs'%(t['task_id'],len(done),elapsed),flush=True)
                del result,im;gc.collect()
            except BaseException as exc:
                meta.update(status='failed',finished_at=now(),error_type=type(exc).__name__,error=str(exc))
                write(attempt/'record.json',meta);(attempt/'traceback.txt').write_text(traceback.format_exc())
                raise
        session['status']='completed_selected_tasks'
    except BaseException as exc:
        session.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        session['finished_at']=now();write(sf,session)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['check','generate','report'])
    ap.add_argument('--method',choices=METHODS+['all'],default='all')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--execute',action='store_true')
    ap.add_argument('--retry-failed',action='store_true')
    ap.add_argument('--worker-mode',choices=['plain','regional','baseline_fallback'],help=argparse.SUPPRESS)
    ap.add_argument('--output-root',type=Path,default=Path('/root/autodl-tmp/zh150_baselines_generation_v1'))
    ap.add_argument('--sdxl-model',type=Path,default=Path('/root/autodl-tmp/stable-diffusion-xl-base-1.0'))
    ap.add_argument('--flux-model',type=Path,default=Path('/root/autodl-tmp/ragd_workspace/models/FLUX.1-dev'))
    ap.add_argument('--ragd-repo',type=Path,default=Path('/root/autodl-tmp/ragd_workspace/RAG-Diffusion'))
    a=ap.parse_args()
    for n in ['output_root','sdxl_model','flux_model','ragd_repo']: setattr(a,n,getattr(a,n).resolve())
    check_bundle()
    if a.worker_mode:
        require(a.command=='generate' and a.execute and a.method!='all','Invalid worker invocation');worker(a);return
    methods=METHODS if a.method=='all' else [a.method]
    for method in methods:
        tasks=tasks_for(method);conf=configuration(method,a,tasks);out=a.output_root/method
        if (out/'configuration.json').exists(): require(read(out/'configuration.json')==conf,'Existing frozen configuration differs')
        count=sum(completed(out,t,conf) for t in tasks)
        print(json.dumps({'method':method,'prompts':150,'tasks':450,'completed':count,'smoke_tasks':len(select_smoke(tasks,method)),
                          'generation':conf['generation'],'API_calls':0},ensure_ascii=False,indent=2),flush=True)
    if a.command!='generate' or not a.execute:
        print('READ ONLY: no model loading, API calls or GPU generation.');return
    a.output_root.mkdir(parents=True,exist_ok=True)
    lock=(a.output_root/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (a.output_root/'STOP').exists(),'STOP file exists; remove it explicitly before resuming')
    for method in methods:
        modes=['plain'] if method=='sdxl' else (['regional','baseline_fallback'] if method=='rpg' else ['regional'])
        for mode in modes:
            if (a.output_root/'STOP').exists(): return
            cmd=[PYTHONS[method],'-u',str(Path(__file__).resolve()),'generate','--execute','--method',method,'--worker-mode',mode]
            for n in ['output_root','sdxl_model','flux_model','ragd_repo']:cmd+=['--'+n.replace('_','-'),str(getattr(a,n))]
            if a.smoke:cmd+=['--smoke']
            if a.retry_failed:cmd+=['--retry-failed']
            subprocess.run(cmd,check=True)
    print('Selected generation finished. Run report for verified full counts.',flush=True)
if __name__=='__main__':main()
