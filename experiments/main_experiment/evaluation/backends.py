"""Historical evaluators, with new output binding and orchestration only."""
import importlib.util
import importlib.metadata
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from common import ROOT, read, sha, need, freeze, digest, english_value, valid_zh


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def english_provenance(config):
    engine = load_file('historical_en', ROOT / 'vendor/historical_en.py')
    args = SimpleNamespace(repo=Path(config['english_repo']).resolve())
    # Reuse exact historical source/weight/environment inspection.
    # ROOT here is used only to locate the frozen prompt manifest for provenance.
    engine.ROOT = ROOT / 'data'
    prov = engine.provenance(args)
    return engine, args, prov


def chinese_provenance(config):
    repo = Path(config['chinese_repo']).resolve()
    old = read(ROOT / 'references/historical_zh_judge.json')
    for name, expected in old['source_sha256'].items():
        need(sha(repo / 'eval/src' / name) == expected, 'Historical Chinese source differs: ' + name)
    evidence = read(ROOT / 'references/historical_zh_evaluation.json')
    need(sha(repo / 'data/test_prompts_zh.csv') == evidence['official_csv_sha256'], 'Historical annotation CSV differs')
    model = Path(config['chinese_model_dir']).resolve()
    for name, ref in [('config.json', 'historical_model_config.json'), ('generation_config.json', 'historical_generation_config.json')]:
        need(read(model / name) == read(ROOT / 'references' / ref), 'Judge model configuration differs: ' + name)
    packages = {}
    for name in ['requests', 'pandas', 'Pillow', 'tqdm', 'vllm', 'transformers', 'torch']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'historical_judge': old, 'repo': str(repo), 'api_url': config['api_url'], 'local_model_dir': str(model), 'model_config_sha256': sha(model / 'config.json'), 'generation_config_sha256': sha(model / 'generation_config.json'), 'client_python': sys.version, 'client_packages': packages, 'weight_identity': 'operator-declared revision; not a complete weight-content attestation', 'effective_server_sampling': 'not independently established'}


def check_image_unchanged(r):
    need(sha(r['image_path']) == r['image_sha256'], 'Input image changed during evaluation')
    need(sha(Path(r['image_path']).with_suffix('.json')) == r['sidecar_sha256'], 'Input sidecar changed during evaluation')


def english_score(config, records, folder, command, retry_failed):
    engine, args, prov = english_provenance(config)
    freeze(folder / 'evaluator.json', prov)
    scored = []
    for cat in engine.CATS:
        selected = [r for r in records if r['category'] == cat]
        if not selected:
            continue
        d = folder / cat
        samples = d / 'samples'
        samples.mkdir(parents=True, exist_ok=True)
        mapping = []
        for i, r in enumerate(selected):
            caption = r['caption']
            filename = caption + f'_{i:06d}.png'
            need(not any(c in caption for c in ['_', '/', '\\', '\n', '\r', '\x00']) and len(filename.encode()) <= 255, 'Unsupported evaluator filename: ' + r['prompt_id'])
            target = samples / filename
            if target.is_symlink():
                need(target.resolve() == Path(r['image_path']), 'Wrong evaluator image link')
            else:
                need(not target.exists(), 'Evaluator export collision')
                target.symlink_to(r['image_path'])
            mapping.append(dict(r, question_id=i, filename=filename))
        need({p.name for p in samples.iterdir()} == {r['filename'] for r in mapping}, 'Unexpected evaluator image')
        freeze(d / 'image_map.json', mapping)
        scores = {}
        for branch in engine.branches(cat):
            for r in mapping:
                check_image_unchanged(r)
            if command == 'run':
                attempts = d / ('branch_' + branch)
                if attempts.exists() and any(attempts.iterdir()) and not (d / ('completed_' + branch + '.json')).exists():
                    need(retry_failed, 'Unfinished branch; inspect its log, then use --retry-failed')
                engine.run_branch(args, d, branch, cat, mapping, prov)
            values = engine.branch_result(d, branch, mapping, prov)
            need(values is not None, 'Missing completed branch: ' + cat + '/' + branch)
            scores[branch] = values
        for r in mapping:
            check_image_unchanged(r)
            values = {k: v[r['question_id']] for k, v in scores.items()}
            scored.append({'prompt_id':r['prompt_id'], 'method':r['method'], 'seed':r['seed'], 'category':cat, 'score':english_value(cat, r['complex_kind'], values), 'branches':values, 'image_sha256':r['image_sha256']})
    return scored


def chinese_score(config, records, folder, command, retry_failed):
    prov = chinese_provenance(config)
    freeze(folder / 'evaluator.json', prov)
    src = Path(config['chinese_repo']).resolve() / 'eval/src'
    sys.path.insert(0, str(src))
    # All imports in this file are from the hash-checked historical pair.
    import eval_common
    import vllm_request
    need(Path(eval_common.__file__).resolve() == src / 'eval_common.py', 'Wrong eval_common import')
    need(Path(vllm_request.__file__).resolve() == src / 'vllm_request.py', 'Wrong request client import')
    if command == 'run':
        from urllib.request import urlopen
        with urlopen(config['api_url'].rstrip('/') + '/v1/models', timeout=10) as response:
            import json
            service = json.load(response)
        need(any(x.get('id') == 'QwenVL' for x in service.get('data', [])), 'QwenVL service unavailable')
    scored = []
    for r in records:
        check_image_unchanged(r)
        task = {'index':r['source_index'], 'prompt':r['caption'], 'testpoint':r['annotations']['考点'], 'test_desc':r['annotations']['考点对应描述'], 'img_path':r['image_path'], 'lang':'zh', 'max_retries':2, 'api_url':config['api_url']}
        d = folder / 'records' / r['prompt_id'] / ('seed_' + str(r['seed']))
        d.mkdir(parents=True, exist_ok=True)
        freeze(d / 'binding.json', {'image':r, 'evaluation_task':task})
        receipt = d / 'completed.json'
        if receipt.exists():
            proof = read(receipt)
            response_path = d / proof['response_file']
            need(sha(response_path) == proof['response_sha256'], 'Cached judge response changed')
            result = read(response_path)
        else:
            need(command == 'run', 'Missing judge result: ' + str(d))
            attempts = sorted(d.glob('attempt_*'))
            need(not attempts or retry_failed, 'Earlier incomplete/failed judge attempt; inspect and use --retry-failed')
            work = d / ('attempt_%03d' % (len(attempts) + 1))
            work.mkdir()
            cwd = Path.cwd()
            try:
                # Historical evaluate_batch appends results.json; isolate it per image/attempt.
                os.chdir(work)
                result = eval_common.call_evaluation_vllm(task)
                freeze(work / 'response.json', result)
                valid_zh(result, task)
                check_image_unchanged(r)
            except Exception as exc:
                freeze(work / 'failure.json', {'type':type(exc).__name__, 'error':str(exc)})
                raise
            finally:
                os.chdir(cwd)
            response_path = work / 'response.json'
            freeze(receipt, {'response_file':str(response_path.relative_to(d)), 'response_sha256':sha(response_path)})
        obj = valid_zh(result, task)
        scored.append({'prompt_id':r['prompt_id'], 'method':r['method'], 'seed':r['seed'], 'testpoint':obj['testpoint'], 'score':obj['score'], 'image_sha256':r['image_sha256']})
        print('SCORED', r['method'], r['prompt_id'], r['seed'], flush=True)
    return scored
