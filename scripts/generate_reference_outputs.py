#!/usr/bin/env python3
from pathlib import Path
import csv, json
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'outputs/reference'; out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(open(ROOT/'paper/reference/table2_overall.csv')))
metrics=['iou','dice','precision','recall','f1']
for metric in metrics:
    fig,ax=plt.subplots(figsize=(9,4.8)); ax.bar([r['model'] for r in rows],[float(r[metric]) for r in rows]); ax.set_ylabel(metric.upper()); ax.set_ylim(85,100); ax.set_title(f'Published {metric.upper()} comparison'); ax.tick_params(axis='x',rotation=25); fig.tight_layout(); fig.savefig(out/f'paper_{metric}_comparison.png',dpi=180); plt.close(fig)
(Path(out/'paper_targets.json')).write_text((ROOT/'paper/reference/paper_targets.json').read_text())
print('[ok] reference charts:',out)
