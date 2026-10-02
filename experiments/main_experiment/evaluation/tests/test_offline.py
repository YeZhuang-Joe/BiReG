"""No GPU/model/service requests. Synthetic cases are not experimental results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import english_value, summarize, valid_zh, freeze
from backends import load_file
from common import ROOT

class ScoringTests(unittest.TestCase):
    def test_complex_rules(self):
        values = {'a':0.2,'s':0.8,'c':0.5}
        self.assertAlmostEqual(english_value('complex','spatial',values),0.5)
        self.assertAlmostEqual(english_value('complex','action',values),0.35)
        self.assertAlmostEqual(english_value('complex','both',values),0.5)
        self.assertEqual(english_value('spatial',None,values),0.8)
        with self.assertRaises(ValueError):
            english_value('color',None,{'a':float('nan')})

    def test_checkpoint_weighting_and_seed_sd(self):
        rows=[]
        for seed in [1234,2468,42]:
            rows.extend([
                {'prompt_id':'a','method':'bireg','seed':seed,'testpoint':['属性-颜色','属性-颜色','属性-颜色'],'score':[1,1,1]},
                {'prompt_id':'b','method':'bireg','seed':seed,'testpoint':['属性-颜色'],'score':[0]},
            ])
        result=summarize(rows,'zh',True)
        for r in result['results']:
            self.assertEqual(r['mean'],0.75)
            self.assertEqual(r['sample_sd'],0)
        with self.assertRaises(ValueError):
            summarize(rows+[rows[0]],'zh',True)

    def test_english_sd(self):
        rows=[{'prompt_id':'a','method':'sdxl','seed':s,'category':'color','score':v} for s,v in zip([2026,3407,5678],[0,1,2])]
        result=summarize(rows,'en',False)['results'][0]
        self.assertEqual(result['mean'],1)
        self.assertEqual(result['sample_sd'],1)
        self.assertIsNone(summarize(rows[:1],'en',False)['results'][0]['sample_sd'])

    def test_judge_alignment(self):
        task={'prompt':'原文','img_path':'image.png','testpoint':['风格']}
        result={'prompt':'原文','img_path':'image.png','result_json':{'testpoint':['风格'],'score':[1]}}
        valid_zh(result,task)
        for invalid in [[True],[0.5],[],[2]]:
            result['result_json']['score']=invalid
            with self.assertRaises(ValueError):valid_zh(result,task)
        result['result_json']['score']=[1]
        result['result_json']['testpoint']=['属性']
        with self.assertRaises(ValueError):valid_zh(result,task)

    def test_immutable_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'proof.json'
            freeze(p,{'image':'one'});freeze(p,{'image':'one'})
            with self.assertRaises(ValueError):freeze(p,{'image':'two'})

    def test_branch_ids(self):
        engine=load_file('test_historical_en',ROOT/'vendor/historical_en.py')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'scores.json'
            p.write_text(json.dumps([{'question_id':2,'answer':0.4},{'question_id':1,'answer':0.8}]))
            self.assertEqual(engine.read_scores(p,[{'question_id':1},{'question_id':2}]),{2:0.4,1:0.8})
            with self.assertRaises(ValueError):engine.read_scores(p,[{'question_id':1}])
            p.write_text(json.dumps([{'question_id':1,'answer':0.4},{'question_id':1,'answer':0.8}]))
            with self.assertRaises(ValueError):engine.read_scores(p,[{'question_id':1}])

if __name__=='__main__':unittest.main()
