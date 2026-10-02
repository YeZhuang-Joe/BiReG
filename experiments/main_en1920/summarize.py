"""Reproduce EN1920 category means and sample SDs across three seed means."""
import json,math,statistics
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parent
METHODS=['sdxl','rpg','kolors','rpg_kolors','ragd','bireg']
CATEGORIES=['color','shape','texture','spatial','non_spatial','complex']
SEEDS=[2026,3407,5678]
def load(path):
 return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def main():
 prompts=load(ROOT/'data/prompts.frozen.jsonl');ids={r['prompt_id'] for r in prompts}
 assert len(prompts)==len(ids)==1920,'Unexpected prompt count or duplicate ID'
 scores=load(ROOT/'data/scores.jsonl');seen=set();groups=defaultdict(list);categories={}
 for r in scores:
  p,m,s,c,v=(r[k] for k in ['prompt_id','method','seed','category','score'])
  assert p in ids and m in METHODS and s in SEEDS and c in CATEGORIES
  assert type(v) in (int,float) and math.isfinite(v),'Invalid score'
  key=(p,m,s);assert key not in seen,'Duplicate score';seen.add(key)
  categories.setdefault(p,c);assert categories[p]==c
  groups[m,c,s].append(v)
 assert seen=={(p,m,s) for p in ids for m in METHODS for s in SEEDS},'Incomplete coverage'
 assert Counter(categories.values())==Counter(dict.fromkeys(CATEGORIES,320))
 results=[]
 for m in METHODS:
  for c in CATEGORIES:
   assert all(len(groups[m,c,s])==320 for s in SEEDS)
   seed_means={str(s):statistics.mean(groups[m,c,s]) for s in SEEDS}
   results.append(dict(method=m,category=c,n_prompts=320,n_images=960,seed_means=seed_means,mean=statistics.mean(seed_means.values()),sd=statistics.stdev(seed_means.values())))
 out=ROOT/'results';out.mkdir(exist_ok=True)
 (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
 lines=['| Method | '+' | '.join(CATEGORIES)+' |','|---|'+'---:|'*6]
 for m in METHODS:
  r=[x for x in results if x['method']==m]
  lines.append('| '+m+' | '+' | '.join(f"{x['mean']:.4f} ± {x['sd']:.4f}" for x in r)+' |')
 (out/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('PASS: 34,560 records; complete prompt/method/seed coverage. Results: results/summary.json and results/table.md')
 print('\n'.join(lines))
if __name__=='__main__':main()
