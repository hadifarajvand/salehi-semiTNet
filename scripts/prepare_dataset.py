#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, shutil, tarfile
from pathlib import Path
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def extract_archives(root):
    for p in list(root.rglob('*')):
        if p.is_file() and (tarfile.is_tarfile(p) if p.suffix in {'.tar','.gz','.tgz'} else False):
            out=p.parent/(p.stem+'_extracted'); out.mkdir(exist_ok=True)
            with tarfile.open(p) as t: t.extractall(out)

def coco_candidates(root):
    rows=[]
    for p in root.rglob('*.json'):
        try:
            d=json.loads(p.read_text())
            if isinstance(d,dict) and isinstance(d.get('images'),list) and isinstance(d.get('annotations'),list): rows.append((p,d))
        except Exception: pass
    return rows

def classify_json(p,d):
    s=str(p).lower()
    if any(x in s for x in ['test','val','validation']): return 'test'
    if any(x in s for x in ['train','label']): return 'train'
    return 'unknown'

def build_image_index(root):
    idx={}
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in EXT: idx.setdefault(p.name,p)
    return idx

def link_or_copy(src,target):
    try: os.link(src,target)
    except (OSError,FileExistsError):
        if not target.exists(): shutil.copy2(src,target)

def copy_split(index,dest,name,p,d):
    outimg=dest/f'labeled/{name}/images'; outimg.mkdir(parents=True,exist_ok=True)
    new=[]
    for im in d['images']:
        src=index.get(Path(im['file_name']).name)
        if not src: continue
        target=outimg/src.name
        if not target.exists(): link_or_copy(src,target)
        x=dict(im); x['file_name']=target.name; new.append(x)
    keep={x['id'] for x in new}; anns=[x for x in d['annotations'] if x.get('image_id') in keep]
    out=dict(d); out['images']=new; out['annotations']=anns
    outjson=dest/f'labeled/{name}/annotations.json'; outjson.parent.mkdir(parents=True,exist_ok=True); outjson.write_text(json.dumps(out))
    return len(new),len(anns)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--dest',required=True); ap.add_argument('--seed',type=int,default=2024); a=ap.parse_args()
    src=Path(a.source); dest=Path(a.dest)
    if dest.exists(): shutil.rmtree(dest)
    dest.mkdir(parents=True)
    extract_archives(src)
    index=build_image_index(src)
    cs=coco_candidates(src); train=None; test=None
    for p,d in cs:
        typ=classify_json(p,d)
        if typ=='train' and (train is None or len(d['images'])>len(train[1]['images'])): train=(p,d)
        if typ=='test' and (test is None or len(d['images'])>len(test[1]['images'])): test=(p,d)
    split_source='official'
    if train and test:
        tr=copy_split(index,dest,'train',*train); te=copy_split(index,dest,'test',*test)
    elif cs:
        # Deterministic fallback preserving COCO annotations.
        p,d=max(cs,key=lambda x:len(x[1]['images'])); rng=random.Random(a.seed); images=list(d['images']); rng.shuffle(images)
        cut=round(len(images)*7/8); groups={'train':images[:cut],'test':images[cut:]}; counts={}
        for name,ims in groups.items():
            ids={x['id'] for x in ims}; dd=dict(d); dd['images']=ims; dd['annotations']=[x for x in d['annotations'] if x.get('image_id') in ids]
            counts[name]=copy_split(index,dest,name,p,dd)
        tr,te=counts['train'],counts['test']; split_source='deterministic_7_to_1_fallback'
    else: raise SystemExit('No COCO annotation JSON found in dataset snapshot.')
    # Unlabeled images: files not referenced by labeled JSON, favor folders named unlabeled.
    used={p.name for p in (dest/'labeled').rglob('*') if p.is_file() and p.suffix.lower() in EXT}
    candidates=[p for p in src.rglob('*') if p.is_file() and p.suffix.lower() in EXT and p.name not in used]
    candidates.sort(key=lambda p:(0 if 'unlabel' in str(p).lower() else 1,str(p)))
    udir=dest/'unlabeled/images'; udir.mkdir(parents=True)
    images=[]
    for i,p in enumerate(candidates):
        t=udir/p.name
        if not t.exists(): link_or_copy(p,t)
        images.append({'id':p.stem,'file_name':t.name})
    (dest/'unlabeled/images.json').write_text(json.dumps({'images':images},indent=2))
    manifest={'source':str(src.resolve()),'split_source':split_source,'seed_if_fallback':a.seed,'train_images':tr[0],'train_annotations':tr[1],'test_images':te[0],'test_annotations':te[1],'unlabeled_images':len(images)}
    (dest/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
