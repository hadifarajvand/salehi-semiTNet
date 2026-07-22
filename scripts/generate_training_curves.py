#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser(); ap.add_argument('--metrics-log',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
rows=[]
for line in Path(a.metrics_log).read_text().splitlines():
 try: rows.append(json.loads(line))
 except Exception: pass
loss_keys=sorted({k for r in rows for k in r if k.startswith('total_loss') or k=='total_loss'})
fig,ax=plt.subplots(figsize=(9,4.5))
if rows and loss_keys:
 x=[r.get('iteration',i) for i,r in enumerate(rows)]
 for k in loss_keys: ax.plot(x,[r.get(k,float('nan')) for r in rows],label=k)
 ax.legend()
else: ax.text(.5,.5,'No training-loss values found',ha='center',va='center')
ax.set_xlabel('Iteration'); ax.set_ylabel('Loss'); ax.set_title('SemiTNet training history'); fig.tight_layout(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=180); print(out)
