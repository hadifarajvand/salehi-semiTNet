#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA_REPO='Bryceee/TISI15k-Dataset'
MODEL_REPO='Bryceee/SemiTNet'
MODEL_FILE='SemiTNet_Tooth_Instance_Segmentation_32Classes.pth'
MODEL_SHA256='8364853c7632a491fd66108e23a536fa68e9a9c9b416b21c69143a4d02a26c0a'

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def dataset(force=False):
    from huggingface_hub import snapshot_download
    out=ROOT/'data/raw/TSI15k'; out.mkdir(parents=True,exist_ok=True)
    snapshot_download(repo_id=DATA_REPO,repo_type='dataset',local_dir=out,local_dir_use_symlinks=False,force_download=force)
    print('[ok] dataset snapshot:',out)

def checkpoint(force=False):
    from huggingface_hub import hf_hub_download
    outdir=ROOT/'assets/checkpoints'; outdir.mkdir(parents=True,exist_ok=True)
    cached=Path(hf_hub_download(repo_id=MODEL_REPO,filename=MODEL_FILE,force_download=force))
    out=outdir/MODEL_FILE; shutil.copy2(cached,out)
    got=sha256(out)
    if got!=MODEL_SHA256: raise SystemExit(f'Checkpoint SHA256 mismatch: {got}')
    (outdir/'CHECKSUMS.json').write_text(json.dumps({MODEL_FILE:got},indent=2))
    print('[ok] checkpoint verified:',out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',action='store_true'); ap.add_argument('--checkpoint',action='store_true'); ap.add_argument('--all',action='store_true'); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
    if not (a.dataset or a.checkpoint or a.all): a.all=True
    if a.dataset or a.all: dataset(a.force)
    if a.checkpoint or a.all: checkpoint(a.force)
if __name__=='__main__': main()
