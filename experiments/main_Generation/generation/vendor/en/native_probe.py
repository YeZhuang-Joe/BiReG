"""CPU structural probe of uploaded functions; no model inference or API calls."""
import ast,json,zipfile,io,contextlib,hashlib,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent/'native_source'

def load_functions():
 tree=ast.parse((ROOT/'matrix.py').read_text())
 # Preserve all original definitions/constants, omit unused third-party imports.
 tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 ns={};exec(compile(tree,'matrix.py','exec'),ns)
 tree=ast.parse((ROOT/'RegionalDiffusion_xl.py').read_text())
 info=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='regional_info')
 exec(compile(ast.Module(body=[info],type_ignores=[]),'regional_info','exec'),ns|{}) if False else None
 ns['TOKENS']=75
 exec(compile(ast.Module(body=[info],type_ignores=[]),'regional_info','exec'),ns)
 return ns

def probe(ns,text,caption):
 lines=text.splitlines()
 if not lines or not lines[0].startswith('Final split ratio: ') or len(lines)<2 or not lines[1].startswith('Regional Prompt: '):raise ValueError('Missing labels')
 ratio=lines[0][19:];prompt='\n'.join(lines[1:])[17:]
 state=SimpleNamespace(prompt=caption+' BREAK '+prompt,usebase=True)
 original=state.prompt;segments=original.split('BREAK')
 ns['regional_info'](state,original)
 with contextlib.redirect_stdout(io.StringIO()):
  ns['keyconverter'](state,ratio,True)
  ns['matrixdealer'](state,ratio,'0.5')
 cells=[];i=1;used=[0];geometry=[]
 for row in state.split_ratio:
  for cell in row.cols:
   # Exact context index progression used by matsepcalc.
   span=state.pt[i];used.append(i)
   cells.append(dict(box=[cell.start,row.start,cell.end,row.end],segment_index=i,
     skipped_breaks=cell.breaks,context_span=span,empty_segment=not segments[i].strip()))
   i+=1+cell.breaks
 for size in [32,64,128]:
  sumout=0;row_heights=[];row_widths=[]
  for row in state.split_ratio:
   sumin=0;heights=[];widths=[]
   for cell in row.cols:
    addin=addout=0;sumin+=int(size*cell.end)-int(size*cell.start)
    if cell.end>=.999:
     addin=sumin-size;sumout+=int(size*row.end)-int(size*row.start)
     if row.end>=.999:addout=sumout-size
    heights.append(int(size*row.end)-(int(size*row.start)+addout))
    widths.append(int(size*cell.end)-(int(size*cell.start)+addin))
   if len(set(heights))!=1 or min(widths)<=0 or heights[0]<=0:raise ValueError('Invalid crop dimensions')
   row_heights.append(heights[0]);row_widths.append(sum(widths))
  if any(w!=size for w in row_widths) or sum(row_heights)!=size:raise ValueError('Concat/reshape dimension mismatch')
  geometry.append(size)
 return dict(ratio=ratio,descriptions=len(segments)-1,grid_regions=len(cells),cells=cells,
  unused_regional_segment_indices=[i for i in range(1,len(segments)) if i not in used],
  tested_square_latent_sizes=geometry,extra_output_lines=len(lines)-2,
  context_span_exceeds_encoded_blocks=any(c['context_span'][1]>len(segments) for c in cells))
