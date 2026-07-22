#!/usr/bin/env python3
from pathlib import Path
import argparse, json
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); a=ap.parse_args(); root=Path(a.root)
    files=[p for p in root.rglob('*') if p.is_file()]
    imgs=[p for p in files if p.suffix.lower() in EXT]
    js=[p for p in files if p.suffix.lower()=='.json']
    report={'root':str(root.resolve()),'files':len(files),'images':len(imgs),'json_files':[str(p.relative_to(root)) for p in js],'top_level':[p.name for p in root.iterdir()] if root.exists() else []}
    out=root/'DATASET_INVENTORY.json'; out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not imgs: raise SystemExit('No images found; verify the Hugging Face dataset download.')
if __name__=='__main__': main()
