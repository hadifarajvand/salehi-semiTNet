#!/usr/bin/env python3
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]; out=ROOT.parent/(ROOT.name+'.zip')
exclude=('data/raw/','data/processed/','assets/checkpoints/','vendor/SemiT-SAM/','outputs/full/','outputs/inference/')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for p in ROOT.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(ROOT).as_posix()
        if '__pycache__/' in rel or rel.endswith('.pyc'): continue
        if any(rel.startswith(x) for x in exclude): continue
        z.write(p,Path(ROOT.name)/rel)
print(out)
