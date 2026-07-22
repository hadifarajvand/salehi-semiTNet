#!/usr/bin/env python3
from pathlib import Path
import json, platform, shutil, sys
ROOT=Path(__file__).resolve().parents[1]
free_gb=round(shutil.disk_usage(ROOT).free/1024**3,2)
report={'python':sys.version.split()[0],'platform':platform.platform(),'upstream_patched':(ROOT/'vendor/SemiT-SAM/REPRODUCTION_PATCH_MANIFEST.json').exists(),'dataset_prepared':(ROOT/'data/processed/TSI15k/dataset_manifest.json').exists(),'released_checkpoint':(ROOT/'assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth').exists(),'disk_free_gb':free_gb if free_gb < 1_000_000 else None}
try:
 import torch
 report.update({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count(),'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
except Exception as e: report['torch_error']=str(e)
report['paper_reference']={'hardware':'8x V100 32GB','training_time_hours':6,'iterations':26250,'batch_total':16,'inference_fps':0.658}
report['recommendation']='released-checkpoint inference first' if report.get('cuda_available') else 'smoke only here; use Linux NVIDIA GPU for inference/full training'
out=ROOT/'outputs/PREFLIGHT.json'; out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
