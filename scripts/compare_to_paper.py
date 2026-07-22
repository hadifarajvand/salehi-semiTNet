#!/usr/bin/env python3
from pathlib import Path
import argparse, json
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',required=True); a=ap.parse_args()
    got=json.loads(Path(a.results).read_text()); target=json.loads((ROOT/'paper/reference/paper_targets.json').read_text())['overall']
    aliases={'iou':['iou','segm_iou'],'dice':['dice','segm_dice'],'precision':['precision','id_precision'],'recall':['recall','id_recall'],'f1':['f1','id_f1']}
    lines=['# Comparison with the published SemiTNet results','','| Metric | Paper | Run | Difference |','|---|---:|---:|---:|']
    for k,names in aliases.items():
        v=next((got.get(n) for n in names if got.get(n) is not None),None)
        lines.append(f'| {k} | {target[k]:.2f} | {"N/A" if v is None else f"{float(v):.2f}"} | {"N/A" if v is None else f"{float(v)-target[k]:+.2f}"} |')
    out=Path(a.results).parent/'PAPER_COMPARISON.md'; out.write_text('\n'.join(lines)+'\n'); print(out)
if __name__=='__main__': main()
