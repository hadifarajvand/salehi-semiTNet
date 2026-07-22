#!/usr/bin/env python3
from pathlib import Path
import argparse, json, math
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['smoke','inference'],required=True); a=ap.parse_args(); out=ROOT/f'outputs/{a.mode}'
    req=[out/'metrics.json']
    if a.mode=='inference': req += [out/'coco_instances_results.json',out/'PAPER_COMPARISON.md',out/'figures/qualitative_predictions.png']
    if a.mode=='smoke': req += [out/'history.csv',out/'run_manifest.json',out/'figures/training_curves.png',out/'figures/predictions.png']
    missing=[str(p.relative_to(ROOT)) for p in req if not p.exists()]
    if missing: raise SystemExit('Missing: '+', '.join(missing))
    m=json.loads((out/'metrics.json').read_text())
    numeric=[v for v in m.values() if isinstance(v,(int,float))]
    if not numeric or any(not math.isfinite(float(v)) for v in numeric): raise SystemExit('Metrics are missing or non-finite')
    if a.mode=='smoke' and float(m.get('dice',0)) <= 0: raise SystemExit('Smoke Dice must be positive')
    print(f'[PASS] {a.mode} acceptance check')
if __name__=='__main__': main()
