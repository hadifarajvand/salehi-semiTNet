#!/usr/bin/env python3
"""Compute the metrics using the same COCOeval recipe shipped by the authors.

The public upstream utility calls COCOeval at IoU=0.5, reports segmentation
stats[0] as IoU, derives Dice from segmentation precision/recall, and derives
identification F1 from bbox precision/recall. This script keeps that recipe,
but makes paths portable and writes machine-readable JSON.
"""
from __future__ import annotations
import argparse, contextlib, io, json, math
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def safe_f1(p: float, r: float) -> float:
    return 0.0 if not math.isfinite(p+r) or p+r <= 0 else 2*p*r/(p+r)


def eval_coco(coco_gt: COCO, coco_dt: COCO, kind: str, image_ids=None, threshold=0.5):
    ev=COCOeval(coco_gt,coco_dt,kind)
    ev.params.iouThrs=np.array([threshold])
    if image_ids is not None: ev.params.imgIds=list(image_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats


def one_group(coco_gt, coco_dt, ids, threshold):
    if not ids: return None
    b=eval_coco(coco_gt,coco_dt,'bbox',ids,threshold)
    s=eval_coco(coco_gt,coco_dt,'segm',ids,threshold)
    bbox_precision=float(b[1]); bbox_recall=float(b[8])
    mask_precision=float(s[1]); mask_recall=float(s[8])
    return {
      'iou':float(s[0])*100,
      'dice':safe_f1(mask_precision,mask_recall)*100,
      'precision':bbox_precision*100,
      'recall':bbox_recall*100,
      'f1':safe_f1(bbox_precision,bbox_recall)*100,
      'images':len(ids),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ground-truth',required=True)
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--iou-threshold',type=float,default=0.5)
    a=ap.parse_args()
    coco_gt=COCO(a.ground_truth); coco_dt=coco_gt.loadRes(a.predictions)
    ids=coco_gt.getImgIds()
    fully=[i for i in ids if len(coco_gt.getAnnIds(imgIds=[i]))==32]
    partial=[i for i in ids if len(coco_gt.getAnnIds(imgIds=[i]))<32]
    result={
      'mode':'released_checkpoint_inference',
      'metric_recipe':'author_public_COCOeval_recipe_at_iou_0.5',
      **one_group(coco_gt,coco_dt,ids,a.iou_threshold),
      'fully_dentate':one_group(coco_gt,coco_dt,fully,a.iou_threshold),
      'partially_edentulous':one_group(coco_gt,coco_dt,partial,a.iou_threshold),
      'ground_truth':str(Path(a.ground_truth).resolve()),
      'predictions':str(Path(a.predictions).resolve()),
    }
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
