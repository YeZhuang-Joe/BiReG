"""CPU checks against the retained matrix code; no image quality decisions."""
import ast
import contextlib
import io
import math
import re
from functools import lru_cache
from types import SimpleNamespace
from .common import BACKEND, source_hashes

@lru_cache(maxsize=1)
def native_functions():
    source_hashes()
    tree=ast.parse((BACKEND/'matrix.py').read_bytes())
    tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    ns={};exec(compile(tree,'historical_matrix.py','exec'),ns)
    tree=ast.parse((BACKEND/'RegionalKolorsDiffusion_xl.py').read_bytes())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='regional_info')
    ns['TOKENS']=254
    exec(compile(ast.Module(body=[node],type_ignores=[]),'historical_regional_info.py','exec'),ns)
    return ns

def check_layout(text,base_prompt,width,height,max_regions=7):
    lines=text.splitlines()
    if len(lines)!=2 or not lines[0].startswith('Final split ratio: ') or not lines[1].startswith('Regional Prompt: '):
        raise ValueError('Require exactly two labeled lines, no fences or commentary')
    ratio=lines[0][19:];regional=lines[1][17:]
    if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?(?:[,;][0-9]+(?:\.[0-9]+)?)*',ratio):raise ValueError('Invalid numeric ratio grammar')
    numbers=[[float(x) for x in row.split(',')] for row in ratio.split(';')]
    if any(not math.isfinite(x) or x<=0 for row in numbers for x in row):raise ValueError('Weights must be finite and positive')
    segments=regional.split(' BREAK ')
    if any(not x.strip() or 'BREAK' in x for x in segments):raise ValueError('Empty/malformed regional delimiter')
    if any(t in (regional+' '+base_prompt).upper() for t in ['ADDBASE','ADDCOL','ADDROW','ADDCOMM']):raise ValueError('Reserved backend control token')
    if 'BREAK' in base_prompt:raise ValueError('Base caption contains reserved BREAK token')
    # Explicit row heights with widths, plus historical implicit full-width rows.
    count=len(numbers[0]) if len(numbers)==1 else sum(max(1,len(r)-1) for r in numbers)
    if count!=len(segments):raise ValueError(f'Layout has {count} regions; descriptions have {len(segments)}')
    if max_regions is not None and count>max_regions:raise ValueError('Region count exceeds configured execution budget')
    if count==1 and ratio!='1.0':raise ValueError('Historical single-region compatibility requires literal 1.0')
    ns=native_functions();state=SimpleNamespace(prompt=base_prompt+' BREAK '+regional,usebase=True)
    ns['regional_info'](state,state.prompt)
    if state.pt!=[[i,i+1] for i in range(count+1)]:raise ValueError('Historical context spans exceed one encoded block per description')
    with contextlib.redirect_stdout(io.StringIO()):
        ns['keyconverter'](state,ratio,True)
        if count==1:
            state.split_ratio=[ns['Row'](0.,1.,[ns['Region'](0.,1.,0.5,0)])]
        else:ns['matrixdealer'](state,ratio,0.5)
    boxes=[]
    for row in state.split_ratio:
        for c in row.cols:
            if c.breaks:raise ValueError('Unexpected skipped context segment')
            boxes.append([c.start,row.start,c.end,row.end])
    if len(boxes)!=count:raise ValueError('Native renderer consumes a different number of regions')
    resolutions=[]
    # Mirror historical crop arithmetic, including its non-square correction convention.
    for divisor in (8,16,32,64):
        h,w=height//divisor,width//divisor;sumout=0;total_h=0
        for row in state.split_ratio:
            sumin=0;widths=[];heights=[]
            for cell in row.cols:
                addin=addout=0
                sumin+=int(h*cell.end)-int(h*cell.start)
                if cell.end>=.999:
                    addin=sumin-h;sumout+=int(w*row.end)-int(w*row.start)
                    if row.end>=.999:addout=sumout-w
                y0,y1=int(h*row.start)+addout,int(h*row.end)
                x0,x1=int(w*cell.start)+addin,int(w*cell.end)
                if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError(f'Invalid crop at {w}x{h}')
                widths.append(x1-x0);heights.append(y1-y0)
            if sum(widths)!=w or len(set(heights))!=1:raise ValueError('Native row concatenation mismatch')
            total_h+=heights[0]
        if total_h!=h:raise ValueError('Native image concatenation mismatch')
        resolutions.append([w,h])
    return dict(split_ratio=ratio,regional_prompt=regional,region_count=count,normalized_boxes_xyxy=boxes,
                cpu_checked_feature_sizes=resolutions,parser='native_execution_candidate_v1',
                numerical_weights_normalized_by_native_backend=True,semantic_validation=False)
