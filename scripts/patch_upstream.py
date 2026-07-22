#!/usr/bin/env python3
from __future__ import annotations
import json, re, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def replace_once(text, old, new, label):
    if old in text: return text.replace(old,new,1), True
    if new in text: return text, False
    raise RuntimeError(f'Cannot locate patch target: {label}')

def patch(dest: Path):
    manifest={'upstream':str(dest),'changes':[],'paper_faithful':{},'inferred':{}}
    # Remove the public branch's forced private debug overrides.
    p=dest/'train_net.py'; s=p.read_text()
    marker='# set by handdle for debug; add by hj'
    start=s.find(marker)
    if start >= 0:
        first=s.find('###########################################',start)
        second=s.find('###########################################',first+1)
        if first < 0 or second < 0: raise RuntimeError('Malformed private debug override block')
        end=s.find('\n',second)
        end=len(s) if end < 0 else end+1
        line_start=s.rfind('\n',0,start)+1
        s2=s[:line_start]+'    # Reproduction wrapper: all settings remain controlled by YAML/CLI.\n'+s[end:]
    else:
        s2=s
    if '/root/paddlejob/' in s2: raise RuntimeError('Private path remains after patching train_net.py')
    # Runtime TSI15k registration.
    hook='from tsi15k_runtime import register_tsi15k\n'
    if hook not in s2:
        s2=s2.replace('from config.add_cfg import add_ssl_config\n','from config.add_cfg import add_ssl_config\n'+hook)
        s2=s2.replace('def setup(args):\n','def setup(args):\n    register_tsi15k()\n')
    s2=s2.replace('(\"coco_2017_unlabel_train\",)', '(\"tsi15k_unlabeled\",)')
    p.write_text(s2); manifest['changes'].append('removed forced private train_net overrides and added runtime registration')

    runtime="""from __future__ import annotations\nimport json, os\nfrom pathlib import Path\nfrom PIL import Image\nfrom detectron2.data import DatasetCatalog, MetadataCatalog\nfrom detectron2.data.datasets import register_coco_instances\n\n_REGISTERED=False\ndef _unlabeled_loader(root):\n    image_dir=Path(root)/\"unlabeled/images\"\n    rows=[]\n    for i,p in enumerate(sorted(image_dir.glob(\"*\"))):\n        if p.suffix.lower() not in {\".jpg\",\".jpeg\",\".png\",\".bmp\"}: continue\n        with Image.open(p) as im: w,h=im.size\n        rows.append({\"file_name\":str(p),\"image_id\":p.stem,\"height\":h,\"width\":w})\n    if not rows: raise RuntimeError(f\"No unlabeled images found in {image_dir}\")\n    return rows\n\ndef register_tsi15k():\n    global _REGISTERED\n    if _REGISTERED: return\n    root=Path(os.environ.get(\"TSI15K_ROOT\",\"data/processed/TSI15k\")).resolve()\n    train_json=root/\"labeled/train/annotations.json\"\n    test_json=root/\"labeled/test/annotations.json\"\n    train_images=root/\"labeled/train/images\"\n    test_images=root/\"labeled/test/images\"\n    register_coco_instances(\"tsi15k_train\",{},str(train_json),str(train_images))\n    register_coco_instances(\"tsi15k_test\",{},str(test_json),str(test_images))\n    DatasetCatalog.register(\"tsi15k_unlabeled\",lambda r=root:_unlabeled_loader(r))\n    MetadataCatalog.get(\"tsi15k_unlabeled\").set(evaluator_type=\"\")\n    _REGISTERED=True\n"""
    (dest/'tsi15k_runtime.py').write_text(runtime)
    manifest['changes'].append('added environment-driven TSI15k dataset registration')

    base=dest/'configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml'
    cfg=base.read_text()
    cfg=cfg.replace('NUM_CLASSES: 52 # 32','NUM_CLASSES: 32')
    cfg=cfg.replace('IMS_PER_BATCH: 32','IMS_PER_BATCH: 16')
    base.write_text(cfg)
    manifest['changes'].append('restored paper 32 classes and total batch size 16')

    base2=dest/'configs/coco/instance-segmentation/Base-COCO-InstanceSegmentation.yaml'
    cfg2=base2.read_text().replace('TRAIN: ("coco_2017_train",)','TRAIN: ("tsi15k_train",)').replace('TEST: ("coco_2017_val",)','TEST: ("tsi15k_test",)')
    base2.write_text(cfg2)
    manifest['changes'].append('switched config datasets to tsi15k_train/test')

    scripts={
      'repro_train_teacher.sh':"""#!/usr/bin/env bash\nset -euo pipefail\n: ${NUM_GPUS:=8}\n: ${INIT_WEIGHTS:=}\n: ${BATCH_SIZE:=16}\npython3 train_net.py --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --num-gpus "$NUM_GPUS" SSL.TRAIN_SSL False MODEL.WEIGHTS "$INIT_WEIGHTS" SOLVER.IMS_PER_BATCH "$BATCH_SIZE" OUTPUT_DIR output/reproduction_teacher\n""",
      'repro_train_student.sh':"""#!/usr/bin/env bash\nset -euo pipefail\n: ${NUM_GPUS:=8}\n: ${BURNIN_ITER:=3000}\n: ${BATCH_SIZE:=16}\npython3 train_net.py --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --num-gpus "$NUM_GPUS" SSL.TRAIN_SSL True SSL.TEACHER_CKPT output/reproduction_teacher/model_best.pth SSL.BURNIN_ITER "$BURNIN_ITER" SSL.EVAL_WHO STUDENT SOLVER.IMS_PER_BATCH "$BATCH_SIZE" OUTPUT_DIR output/reproduction_student\n""",
      'repro_eval_released.sh':"""#!/usr/bin/env bash\nset -euo pipefail\n: ${NUM_GPUS:=1}\n: ${CHECKPOINT:=../../assets/checkpoints/SemiTNet_Tooth_Instance_Segmentation_32Classes.pth}\npython3 train_net.py --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml --eval-only --num-gpus "$NUM_GPUS" SSL.TRAIN_SSL True SSL.EVAL_WHO STUDENT MODEL.WEIGHTS "$CHECKPOINT" OUTPUT_DIR output/released_checkpoint_eval\n"""
    }
    for name,body in scripts.items():
        q=dest/name; q.write_text(body); q.chmod(0o755)
    manifest['changes'].append('added portable teacher/student/evaluation launchers')
    manifest['paper_faithful']={'classes':32,'batch_total':16,'max_iter':26250,'lr':1e-4,'steps':[24000,25000],'input':1024,'queries':100,'decoder_layers':9}
    manifest['inferred']={'burnin_iter':3000,'reason':'commented debug value in upstream; article does not state exact transition'}
    (dest/'REPRODUCTION_PATCH_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('[ok] upstream patches applied')

if __name__=='__main__':
    patch(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'vendor/SemiT-SAM')
