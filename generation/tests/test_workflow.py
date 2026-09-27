"""Offline integration checks. Fixtures are not API responses or model outputs."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bireg.common import ROOT, load, source_hashes
from bireg.layout import check_layout
from bireg.workflow import run

TEXT='Final split ratio: 0.5,0.5\nRegional Prompt: a red cup BREAK a blue bowl'
SECRET='OFFLINE_TEST_CREDENTIAL_ONLY'
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV1sAAAAASUVORK5CYII=')

class Response:
    status=200
    def __init__(self,content):self.content=content
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return json.dumps({'model':'offline-fixture','choices':[{'finish_reason':'stop','message':{'content':self.content}}]}).encode()

class Opener:
    def __init__(self,texts):self.texts=iter(texts);self.calls=0
    def open(self,req,timeout):
        self.calls+=1
        if req.get_header('Authorization')!='Bearer '+SECRET:raise AssertionError('Missing authentication')
        return Response(next(self.texts))

class FakeRenderer:
    calls=0
    def __init__(self,*args):self.checkpoint={'fixture':True};self.load_seconds=None
    def load(self):pass
    def generate(self,task,path):
        type(self).calls+=1
        if os.environ.get('BIREG_WORKFLOW_PRIVATE_API_KEY')==SECRET:raise AssertionError('Credential leaked to renderer')
        path.write_bytes(PNG)
        return dict(generation_seconds=0.01,scheduler_config={'fixture_nonfinite':float('-inf')},fixture_only=True)

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        config=load(ROOT/'configs/api.example.json');config.update(api_key=SECRET,interval_seconds=0)
        self.key=self.base/'private.json';self.key.write_text(json.dumps(config))
        self.args=argparse.Namespace(prompt='A red cup to the left of a blue bowl.',prompt_file=None,language='auto',detect_only=False,seed=None,output=self.base/'result',api_config=self.key,model_path=self.base/'weights',check=False,retry_failed=False)
        FakeRenderer.calls=0
    def tearDown(self):self.tmp.cleanup()
    def execute(self,texts=(TEXT,)):
        op=Opener(texts)
        with patch('bireg.rendering.checkpoint_info',return_value={'fixture':True}),patch('bireg.rendering.Renderer',FakeRenderer),patch('bireg.planning.build_opener',return_value=op),patch('sys.stdout',new_callable=io.StringIO):
            run(self.args)
        return op
    def test_backend_and_templates_unchanged(self):self.assertEqual(len(source_hashes()),8)
    def test_real_workflow_mocked_io_resume_and_metadata(self):
        op=self.execute();self.assertEqual(op.calls,1);self.assertEqual(FakeRenderer.calls,1)
        result=load(self.args.output/'output.json')
        self.assertEqual(result['record']['scheduler_config']['fixture_nonfinite'],{'__nonfinite_float__':'-Infinity'})
        self.assertEqual((self.args.output/'image.png').read_bytes(),PNG)
        self.assertEqual(self.execute(()).calls,0);self.assertEqual(FakeRenderer.calls,1)
        for path in self.args.output.rglob('*'):
            if path.is_file():self.assertNotIn(SECRET.encode(),path.read_bytes())
    def test_invalid_plan_retry_then_accept(self):
        self.assertEqual(self.execute(('invalid',TEXT)).calls,2)
        self.assertEqual(FakeRenderer.calls,1)
        self.assertEqual(load(self.args.output/'run/plans.frozen.json')['plans'][0]['attempt'],2)
    def test_exhausted_budget_does_not_render(self):
        with self.assertRaises(ValueError):self.execute(('invalid','invalid'))
        self.assertEqual(FakeRenderer.calls,0)
    def test_modified_prompt_blocks_resume(self):
        self.execute();self.args.prompt='A green cup to the left of a blue bowl.'
        with self.assertRaises(ValueError):self.execute(())
        self.assertEqual(FakeRenderer.calls,1)
    def test_changed_seed_blocks_resume(self):
        self.execute();self.args.seed=42
        with self.assertRaises(ValueError):self.execute(())
    def test_changed_canonical_image_is_detected(self):
        self.execute();result=load(self.args.output/'output.json')
        Path(result['canonical_image']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):self.execute(())
    def test_check_is_read_only(self):
        self.args.check=True;self.assertEqual(self.execute(()).calls,0)
        self.assertFalse(self.args.output.exists());self.assertEqual(FakeRenderer.calls,0)
    def test_ambiguous_stops_before_credential_read(self):
        self.args.prompt='红色 red blue green';self.args.api_config=self.base/'missing.json'
        with self.assertRaisesRegex(ValueError,'Mixed'):self.execute(())
        self.assertFalse(self.args.output.exists())
    def test_detect_only_needs_no_credentials_or_model(self):
        self.args.detect_only=True;self.args.api_config=self.base/'missing.json'
        with patch('bireg.rendering.checkpoint_info',side_effect=AssertionError('Model accessed')),patch('sys.stdout',new_callable=io.StringIO):run(self.args)
        self.assertFalse(self.args.output.exists())
    def test_chinese_profile_and_manual_override(self):
        self.args.prompt='红色 red blue green';self.args.language='zh';self.execute()
        config=load(self.args.output/'config.json')['generation']['zh']
        self.assertEqual((config['width'],config['height'],config['lambda_global']),(1536,1024,.2))
    def test_native_layout_geometry_and_rejection(self):
        for w,h in ((1024,1024),(1536,1024)):
            p=check_layout(TEXT.replace('0.5,0.5','2,3'),'caption',w,h)
            self.assertEqual(p['normalized_boxes_xyxy'],[[0.,0.,.4,1.],[.4,0.,1.,1.]])
            check_layout('Final split ratio: 1.0\nRegional Prompt: a cup','caption',w,h)
            for ratio in ('1.0','0,1','0.000001,1'):
                with self.assertRaises(ValueError):check_layout(TEXT.replace('0.5,0.5',ratio),'caption',w,h)

if __name__=='__main__':unittest.main()
