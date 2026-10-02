"""Reuse the existing BiReG renderer; validate new benchmark plan provenance."""
from bireg_experiment.adapters.rpg_kolors import RPGKolorsAdapter
from regional_compat import single_region_compatibility
from plan_check import check_plan
from bridge import sha

class BenchmarkBiReGAdapter(RPGKolorsAdapter):
    method_id='bireg'

    def _validate_task(self, task):
        p=task.get('plan_provenance') or {}
        if p.get('status')!='frozen' or p.get('planner_family')!='bireg':
            raise ValueError('A frozen BiReG plan is required')
        if p.get('language')!=task['language'] or p.get('source_prompt_sha256')!=sha(task['prompt'].encode()):
            raise ValueError('Plan language/caption mismatch')
        if sha(p['response'].encode())!=p['response_sha256']:raise ValueError('Plan changed')
        parsed=check_plan(p['response'])
        if task['plan']!={k:parsed[k] for k in ['split_ratio','regional_prompt','region_count']}:
            raise ValueError('Runtime plan differs from frozen response')
        return super()._validate_task(task)

    def generate(self, task, output_path):
        self._validate_task(task);self.load()
        with single_region_compatibility(self.pipe):
            meta=super().generate(task,output_path)
        meta.update(adapter='BiReGAdapter',shared_renderer_adapter='RPGKolorsAdapter',
                    plan_response_sha256=task['plan_provenance']['response_sha256'],
                    template_sha256=task['plan_provenance']['template_sha256'],
                    planner_id=task['plan_provenance']['planner_id'],
                    planning_source_type=task['plan_provenance']['source_type'])
        return meta
