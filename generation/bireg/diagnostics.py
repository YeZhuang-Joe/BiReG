"""CPU-side dependency/checkpoint probe; no API or full model loading."""
import importlib
import subprocess
from pathlib import Path
from .common import environment,save
from .rendering import backend_imports,checkpoint_info

def doctor(model,output=None):
    result=environment();result['GPU_generation_calls']=0;result['API_calls']=0
    pipeline,encoder,tokenizer=backend_imports()
    result['historical_modules_imported']=True
    import diffusers
    source=Path(diffusers.__file__).resolve()
    result['diffusers_module_path']=str(source)
    try:
        r=subprocess.run(['git','-C',str(source.parent),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10)
        result['diffusers_git_revision']=r.stdout.strip() if r.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):result['diffusers_git_revision']=None
    if model:
        result['checkpoint']=checkpoint_info(model)
        tok=tokenizer.from_pretrained(str(Path(model)/'text_encoder'),local_files_only=True)
        result['tokenizer_loaded']=True
        result['tokenizer_probe_lengths']={s:len(tok(s,truncation=False)['input_ids']) for s in ['A red cup.','一个红色杯子。']}
    if output:save(output,result)
    return result
