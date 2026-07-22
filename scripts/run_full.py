#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--num-gpus',type=int,default=int(os.getenv('NUM_GPUS','8'))); ap.add_argument('--burnin-iter',type=int,default=int(os.getenv('BURNIN_ITER','3000'))); ap.add_argument('--batch-size',type=int,default=int(os.getenv('BATCH_SIZE','16'))); ap.add_argument('--init-weights',default=os.getenv('INIT_WEIGHTS','')); a=ap.parse_args()
    vendor=ROOT/'vendor/SemiT-SAM'; dataset=Path(a.dataset).resolve()
    if not (vendor/'REPRODUCTION_PATCH_MANIFEST.json').exists(): raise SystemExit('Run `make bootstrap` first.')
    env=os.environ.copy(); env.update({'TSI15K_ROOT':str(dataset),'NUM_GPUS':str(a.num_gpus),'BURNIN_ITER':str(a.burnin_iter),'BATCH_SIZE':str(a.batch_size),'INIT_WEIGHTS':str(Path(a.init_weights).resolve()) if a.init_weights else ''})
    manifest={'dataset':str(dataset),'num_gpus':a.num_gpus,'burnin_iter':a.burnin_iter,'burnin_source':'inferred from upstream debug comment, not specified by article','batch_size_total':a.batch_size,'init_weights':a.init_weights or None}
    out=ROOT/'outputs/full'; out.mkdir(parents=True,exist_ok=True); (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run(['bash','repro_train_teacher.sh'],cwd=vendor,env=env,check=True)
    subprocess.run(['bash','repro_train_student.sh'],cwd=vendor,env=env,check=True)
    print('[ok] full training finished; evaluate the generated student checkpoint with run_official_inference.py')
if __name__=='__main__': main()
