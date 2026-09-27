#!/usr/bin/env python3
"""Probe existing environment/tokenizer. Does not call API or generate images."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bireg.diagnostics import doctor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model-path',type=Path,default=Path('/root/autodl-tmp/weights/Kolors'))
p.add_argument('--output',type=Path)
a=p.parse_args()
print(json.dumps(doctor(a.model_path,a.output),ensure_ascii=False,indent=2))
