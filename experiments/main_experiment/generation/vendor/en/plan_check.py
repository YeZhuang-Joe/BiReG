"""Native raw-plan protocol v2. Does not rewrite or rank output content."""
from functools import lru_cache
from native_probe import load_functions,probe
from math import isfinite
_NS=load_functions()
@lru_cache(maxsize=4096)
def check_plan(text,max_regions=7):
    lines=text.splitlines()
    if len(lines)!=2 or not lines[0].startswith('Final split ratio: ') or not lines[1].startswith('Regional Prompt: '):
        raise ValueError('Expected two labeled lines; no output editing')
    ratio=lines[0][19:];prompt=lines[1][17:]
    numbers=[float(v) for row in ratio.split(';') for v in row.split(',')]
    if not numbers or any(not isfinite(v) or v<=0 for v in numbers):raise ValueError('Nonpositive/nonfinite native weight')
    if any(k in prompt for k in ('ADDBASE','ADDROW','ADDCOL','ADDCOMM')):raise ValueError('Reserved native control token in description')
    if ratio=='1.0':
        if not prompt.split('BREAK')[0].strip():raise ValueError('Empty first description')
        native=dict(cells=[dict(box=[0,0,1,1],segment_index=1,skipped_breaks=0)],
            unused_regional_segment_indices=list(range(2,len(prompt.split('BREAK'))+1)),descriptions=len(prompt.split('BREAK')),grid_regions=1,
            single_region_compatibility='existing_single_region_compatibility')
    else:native=probe(_NS,text,'Native audit base caption')
    if native.get('context_span_exceeds_encoded_blocks'):raise ValueError('Native context index outside encoded blocks')
    return dict(split_ratio=ratio,regional_prompt=prompt,region_count=native['grid_regions'],
        native_execution=native,protocol='native_raw_v2',declared_budget=7,
        budget_enforced=False)
