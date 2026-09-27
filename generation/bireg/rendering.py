"""Thin adapter around the six unchanged historical Kolors source files."""
import contextlib
import importlib
import os
import sys
import time
from pathlib import Path
from .common import BACKEND, digest, encoded, environment, filehash, load, lock, now, save, source_hashes
from .layout import check_layout
from .planning import frozen_plans
from .metadata import encode_nonfinite

def tasks_for(cfg,rows,run):
    frozen=frozen_plans(cfg,rows,run)
    if not (run/'plans.frozen.json').exists() or load(run/'plans.frozen.json')!=frozen:raise ValueError('Run freeze before generation; frozen plans differ or are absent')
    tasks=[]
    for rec in frozen['plans']:
        row=rec['prompt'];g=cfg['generation'][row['language']]
        for seed in g['seeds']:
            t=dict(prompt=row,plan=rec['plan'],generation=g,seed=seed,planning=rec,
                   frozen_plans_sha256=filehash(run/'plans.frozen.json'))
            t['task_id']=row['prompt_id']+f'_seed_{seed}';t['task_sha256']=digest(encoded(t));tasks.append(t)
    return tasks

def completed_task(run,task):
    d=run/'generation'/task['task_id']
    for p in sorted(d.glob('attempt_*/result.json')):
        r=load(p)
        if r['task_sha256']!=task['task_sha256']:raise ValueError('Generation task mismatch')
        if r['status']=='completed':
            image=run/r['image_path']
            if filehash(image)!=r['image_sha256']:raise ValueError('Image hash mismatch')
            return r
    return None

@contextlib.contextmanager
def compatibility(pipe):
    """One cell grid; restore the historical seed helper's global name assignment."""
    import torch
    matrix=importlib.import_module('matrix');mod=importlib.import_module(type(pipe).__module__)
    old_matrix=mod.matrixdealer;old_deterministic=torch.use_deterministic_algorithms
    def dispatch(state,ratio,base):
        if state is pipe and ratio=='1.0':
            state.split_ratio=[matrix.Row(0.,1.,[matrix.Region(0.,1.,float(base),0)])]
            state.baseratio=[[float(base)]]
        else:return old_matrix(state,ratio,base)
    mod.matrixdealer=dispatch
    try:yield
    finally:
        mod.matrixdealer=old_matrix
        torch.use_deterministic_algorithms=old_deterministic

def backend_imports():
    source_hashes();sys.path.insert(0,str(BACKEND)) if str(BACKEND) not in sys.path else None
    for name in ['matrix','cross_attention1','configuration_chatglm','modeling_chatglm','tokenization_chatglm','RegionalKolorsDiffusion_xl']:
        if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=BACKEND/(name+'.py'):raise ValueError('Conflicting imported backend module: '+name)
    try:
        return (importlib.import_module('RegionalKolorsDiffusion_xl').RegionalDiffusionXLPipeline,
                importlib.import_module('modeling_chatglm').ChatGLMModel,
                importlib.import_module('tokenization_chatglm').ChatGLMTokenizer)
    except (ImportError,AttributeError) as e:
        raise RuntimeError('Historical renderer dependencies are incompatible. Use the existing RPG environment; see docs/ENVIRONMENT.md. '+str(e)) from e

def checkpoint_info(model):
    model=Path(model).resolve()
    if not model.is_dir():raise ValueError('Missing local Kolors checkpoint')
    names=['model_index.json','text_encoder/config.json','scheduler/scheduler_config.json','unet/config.json','vae/config.json']
    for name in names:
        if not (model/name).is_file():raise ValueError('Missing checkpoint config: '+name)
    weights=sorted(p for p in model.rglob('*') if p.is_file() and p.suffix in ('.bin','.safetensors','.model'))
    if not weights:raise ValueError('No local checkpoint weights/tokenizer found')
    return dict(path=str(model),config_sha256={n:filehash(model/n) for n in names},
                weight_inventory=[dict(path=str(p.relative_to(model)),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in weights],
                weight_content_hashes_verified=False)

class Renderer:
    def __init__(self,model,cpu_offload=True,xformers=True):
        if not xformers:raise ValueError('The retained regional attention requires xformers; --no-xformers is unsupported')
        self.model=Path(model).resolve();self.checkpoint=checkpoint_info(model)
        self.cpu_offload=cpu_offload;self.xformers=xformers;self.pipe=None;self.load_seconds=None
    def load(self):
        if self.pipe is not None:return
        start=time.perf_counter()
        import torch
        from diffusers import AutoencoderKL,UNet2DConditionModel,EulerDiscreteScheduler
        if not torch.cuda.is_available():raise RuntimeError('This historical backend requires CUDA')
        pipeline,encoder,tokenizer=backend_imports();kw=dict(local_files_only=True)
        self.pipe=pipeline(vae=AutoencoderKL.from_pretrained(str(self.model/'vae'),torch_dtype=torch.float16,**kw),
            text_encoder=encoder.from_pretrained(str(self.model/'text_encoder'),torch_dtype=torch.float16,**kw),
            tokenizer=tokenizer.from_pretrained(str(self.model/'text_encoder'),**kw),
            unet=UNet2DConditionModel.from_pretrained(str(self.model/'unet'),torch_dtype=torch.float16,**kw),
            scheduler=EulerDiscreteScheduler.from_pretrained(str(self.model/'scheduler'),**kw),force_zeros_for_empty_prompt=False)
        if self.cpu_offload:self.pipe.enable_model_cpu_offload()
        else:self.pipe.to('cuda')
        if self.xformers:self.pipe.enable_xformers_memory_efficient_attention()
        torch.cuda.synchronize();self.load_seconds=time.perf_counter()-start
    def generate(self,task,path):
        self.load();import torch
        from diffusers import EulerDiscreteScheduler,DPMSolverMultistepScheduler
        g=task['generation'];plan=task['plan'];row=task['prompt']
        check_layout('Final split ratio: '+plan['split_ratio']+'\nRegional Prompt: '+plan['regional_prompt'],row['prompt'],g['width'],g['height'],max_regions=None)
        cls={'EulerDiscreteScheduler':EulerDiscreteScheduler,'DPMSolverMultistepScheduler':DPMSolverMultistepScheduler}[g['scheduler']]
        self.pipe.scheduler=cls.from_pretrained(str(self.model/'scheduler'),local_files_only=True,use_karras_sigmas=g['use_karras_sigmas'])
        raw_parts=(row['prompt']+' BREAK '+plan['regional_prompt']).split('BREAK')
        token_counts=[len(self.pipe.tokenizer(s,truncation=False)['input_ids']) for s in raw_parts]
        # Native encoder truncation is retained and disclosed, not used for semantic selection.
        token_record=dict(actual_encoder_limit=256,part_token_counts=token_counts,truncated_parts=[i for i,n in enumerate(token_counts) if n>256])
        generator=torch.Generator(device='cuda').manual_seed(task['seed'])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        with compatibility(self.pipe):
            image=self.pipe(prompt=plan['regional_prompt'],split_ratio=plan['split_ratio'],base_prompt=row['prompt'],base_ratio=g['lambda_global'],
                height=g['height'],width=g['width'],batch_size=1,seed=task['seed'],generator=generator,
                num_inference_steps=g['steps'],guidance_scale=g['guidance_scale'],negative_prompt=g['negative_prompt']).images[0]
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        if image.size!=(g['width'],g['height']):raise ValueError('Unexpected output image size')
        if image.mode!='RGB':image=image.convert('RGB')
        temp=path.with_name('image.partial.png');image.save(temp);os.replace(temp,path)
        return dict(generation_seconds=elapsed,checkpoint=self.checkpoint,tokenization=token_record,
            scheduler_class=type(self.pipe.scheduler).__name__,scheduler_config=dict(self.pipe.scheduler.config),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cuda_version=torch.version.cuda,gpu_name=torch.cuda.get_device_name(),
            image_size=list(image.size),cpu_offload=self.cpu_offload,xformers=self.xformers,
            deterministic_bitwise_reproduction_guaranteed=False)

def generate(cfg,rows,run,model,execute=False,limit=None,retry_failed=False,cpu_offload=True,xformers=True):
    tasks=tasks_for(cfg,rows,run);done=sum(completed_task(run,t) is not None for t in tasks)
    print(f'Generation: {done}/{len(tasks)} verified completed',flush=True)
    if not execute:return
    if model is None:raise ValueError('--model-path is required for execution')
    with lock(run):
        renderer=Renderer(model,cpu_offload,xformers);count=0
        binding=dict(checkpoint=renderer.checkpoint,cpu_offload=cpu_offload,xformers=xformers,environment=environment())
        save(run/'render_binding.json',binding)
        for task in tasks:
            if completed_task(run,task):continue
            if limit is not None and count>=limit:break
            if (run/'STOP').exists():print('STOP file found.');break
            base=run/'generation'/task['task_id'];attempts=sorted(base.glob('attempt_*'))
            if attempts and not retry_failed:raise ValueError('Prior incomplete/failed generation retained; inspect then use --retry-failed')
            d=base/f'attempt_{len(attempts)+1:02d}';save(d/'task.json',task)
            save(d/'started.json',dict(at=now(),task_sha256=task['task_sha256']))
            # Load measured separately. Runtime logs are local; do not publish unchecked.
            try:
                renderer.load()
                if renderer.load_seconds is not None:
                    print(f'Pipeline loaded: {renderer.load_seconds:.2f}s',flush=True)
                    save(d/'load.json',dict(load_seconds=renderer.load_seconds,environment=environment()));renderer.load_seconds=None
                meta=renderer.generate(task,d/'image.png')
                if 'scheduler_config' in meta:
                    meta['scheduler_config'], changes = encode_nonfinite(meta['scheduler_config'])
                    meta['metadata_serialization'] = dict(version='scheduler-metadata-v1', nonfinite_fields=changes)
                record=dict(meta,status='completed',task_sha256=task['task_sha256'],at=now(),
                            image_path=str((d/'image.png').relative_to(run)),image_sha256=filehash(d/'image.png'))
                save(d/'result.json',record)
            except Exception as e:
                save(d/'result.json',dict(status='failed',at=now(),task_sha256=task['task_sha256'],error_type=type(e).__name__,error=str(e)))
                raise
            count+=1
            print(f"DONE {task['task_id']} | {record['generation_seconds']:.2f}s",flush=True)
