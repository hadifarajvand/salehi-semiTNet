#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

def decode_seg(seg,h,w):
    if isinstance(seg,dict): return mask_utils.decode(seg).astype(bool)
    if isinstance(seg,list):
        rles=mask_utils.frPyObjects(seg,h,w); return np.any(mask_utils.decode(rles).astype(bool),axis=2)
    return np.zeros((h,w),bool)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ground-truth',required=True); ap.add_argument('--predictions',required=True); ap.add_argument('--images',required=True); ap.add_argument('--output',required=True); ap.add_argument('--count',type=int,default=6); ap.add_argument('--score',type=float,default=0.25); a=ap.parse_args()
    gt=json.loads(Path(a.ground_truth).read_text()); preds=json.loads(Path(a.predictions).read_text()); by=defaultdict(list)
    for p in preds:
        if p.get('score',1)>=a.score: by[p['image_id']].append(p)
    selected=[x for x in gt['images'] if x['id'] in by][:a.count]
    if not selected: raise SystemExit('No scored predictions matched ground-truth image IDs.')
    fig,axs=plt.subplots(len(selected),2,figsize=(13,3.2*len(selected))); axs=np.atleast_2d(axs)
    for row,im in enumerate(selected):
        path=Path(a.images)/Path(im['file_name']).name; arr=np.asarray(Image.open(path).convert('RGB')); overlay=arr.copy().astype(float)
        combined=np.zeros(arr.shape[:2],bool)
        for pred in by[im['id']]: combined |= decode_seg(pred.get('segmentation'),arr.shape[0],arr.shape[1])
        overlay[combined]=0.55*overlay[combined]+0.45*np.array([255,255,255])
        axs[row,0].imshow(arr); axs[row,0].set_title(f"Input — {path.name}")
        axs[row,1].imshow(overlay.astype(np.uint8)); axs[row,1].set_title(f"Predicted tooth instances — {len(by[im['id']])}")
        axs[row,0].axis('off'); axs[row,1].axis('off')
    fig.tight_layout(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=180); plt.close(fig); print(out)
if __name__=='__main__': main()
