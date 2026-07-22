#!/usr/bin/env python3
from __future__ import annotations
import argparse, shutil, tarfile, tempfile, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMMIT='bf66074bee9da9b37fd68454bcbac9140c4f59e2'
URL=f'https://github.com/isjinghao/SemiT-SAM/archive/{COMMIT}.tar.gz'
DEST=ROOT/'vendor/SemiT-SAM'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    if DEST.exists():
        if not args.force:
            print(f'[ok] model dependency already exists: {DEST}')
            from patch_upstream import patch; patch(DEST); return
        shutil.rmtree(DEST)
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        arc=Path(td)/'model_dependency.tar.gz'
        print('[download]',URL)
        req=urllib.request.Request(URL,headers={'User-Agent':'SemiTNet-Project/1.0'})
        with urllib.request.urlopen(req,timeout=300) as r, open(arc,'wb') as f: shutil.copyfileobj(r,f)
        with tarfile.open(arc,'r:gz') as t: t.extractall(td)
        src=Path(td)/f'SemiT-SAM-{COMMIT}'
        shutil.move(str(src),str(DEST))
    (DEST/'UPSTREAM_COMMIT.txt').write_text(COMMIT+'\n')
    from patch_upstream import patch
    patch(DEST)
    print('[ok] model dependency prepared:',DEST)
if __name__=='__main__': main()
