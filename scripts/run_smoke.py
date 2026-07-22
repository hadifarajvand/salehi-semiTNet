#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, random, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml
ROOT=Path(__file__).resolve().parents[1]

class SyntheticTeeth(Dataset):
    def __init__(self,n,h,w,seed,labeled=True): self.n=n; self.h=h; self.w=w; self.seed=seed; self.labeled=labeled
    def __len__(self): return self.n
    def __getitem__(self,i):
        rng=random.Random(self.seed+i); mask=Image.new('L',(self.w,self.h),0); d=ImageDraw.Draw(mask)
        yy=np.linspace(-1,1,self.h)[:,None]; xx=np.linspace(-1,1,self.w)[None,:]
        arr=0.18+0.08*np.exp(-((yy+0.1)**2+(xx*0.7)**2)*2)+rng.random()*0.02
        n=rng.randint(8,14)
        for k in range(n):
            x=int((k+1)*self.w/(n+1)+rng.uniform(-2,2)); y=int(self.h*0.54+8*math.sin((x/self.w)*math.pi)+rng.uniform(-2,2)); rw=rng.randint(3,5); rh=rng.randint(7,11)
            d.ellipse((x-rw,y-rh,x+rw,y+rh),fill=255)
        m=np.asarray(mask,np.float32)/255.0
        arr=np.clip(arr+0.62*m+np.random.default_rng(self.seed+i).normal(0,0.035,(self.h,self.w)),0,1)
        x=torch.tensor(arr[None],dtype=torch.float32); y=torch.tensor(m[None],dtype=torch.float32)
        return (x,y) if self.labeled else x

class TinySemiTransformer(nn.Module):
    def __init__(self,d=32):
        super().__init__(); self.patch=nn.Conv2d(1,d,4,4); layer=nn.TransformerEncoderLayer(d,4,d*2,batch_first=True,dropout=0.0); self.tr=nn.TransformerEncoder(layer,2); self.dec=nn.Sequential(nn.ConvTranspose2d(d,16,4,4),nn.GELU(),nn.Conv2d(16,1,1))
    def forward(self,x):
        z=self.patch(x); b,c,h,w=z.shape; z=self.tr(z.flatten(2).transpose(1,2)).transpose(1,2).reshape(b,c,h,w); return self.dec(z)

def metrics(logits,y):
    p=(torch.sigmoid(logits)>0.40); t=y>0.5; tp=(p&t).sum().item(); fp=(p&~t).sum().item(); fn=(~p&t).sum().item(); eps=1e-8
    precision=tp/(tp+fp+eps); recall=tp/(tp+fn+eps); dice=2*tp/(2*tp+fp+fn+eps); iou=tp/(tp+fp+fn+eps); f1=2*precision*recall/(precision+recall+eps)
    return {'iou':iou*100,'dice':dice*100,'precision':precision*100,'recall':recall*100,'f1':f1*100}

def evaluate(model,loader,device):
    model.eval(); vals=[]
    with torch.no_grad():
        for x,y in loader: x,y=x.to(device),y.to(device); vals.append(metrics(model(x),y))
    return {k:sum(v[k] for v in vals)/len(vals) for k in vals[0]}

def main():
    cfg=yaml.safe_load((ROOT/'configs/smoke.yaml').read_text()); seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(min(4,torch.get_num_threads()))
    device=torch.device('cpu'); out=ROOT/'outputs/smoke'; figdir=out/'figures'; figdir.mkdir(parents=True,exist_ok=True)
    labeled=DataLoader(SyntheticTeeth(cfg['train_labeled'],cfg['image_height'],cfg['image_width'],seed),batch_size=cfg['batch_size'],shuffle=True)
    unlabeled=DataLoader(SyntheticTeeth(cfg['train_unlabeled'],cfg['image_height'],cfg['image_width'],seed+1000,False),batch_size=cfg['batch_size'],shuffle=True)
    val=DataLoader(SyntheticTeeth(cfg['validation'],cfg['image_height'],cfg['image_width'],seed+2000),batch_size=cfg['batch_size'])
    teacher=TinySemiTransformer().to(device); opt=torch.optim.AdamW(teacher.parameters(),lr=cfg['learning_rate']); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0],device=device)); history=[]; start=time.time()
    for ep in range(cfg['teacher_epochs']):
        teacher.train(); total=0
        for x,y in labeled: opt.zero_grad(); loss=lossfn(teacher(x),y); loss.backward(); opt.step(); total+=loss.item()
        vm=evaluate(teacher,val,device); history.append({'phase':'teacher','epoch':ep+1,'loss':total/len(labeled),**vm})
    student=TinySemiTransformer().to(device); student.load_state_dict(teacher.state_dict()); opt2=torch.optim.AdamW(student.parameters(),lr=cfg['learning_rate']*0.5)
    for ep in range(cfg['student_epochs']):
        student.train(); total=0; ui=iter(unlabeled)
        for x,y in labeled:
            try: u=next(ui)
            except StopIteration: ui=iter(unlabeled); u=next(ui)
            with torch.no_grad(): pseudo=(torch.sigmoid(teacher(u))>cfg['pseudo_threshold']).float()
            opt2.zero_grad(); sup=lossfn(student(x),y); uns=lossfn(student(u),pseudo); loss=sup+0.5*uns; loss.backward(); opt2.step(); total+=loss.item()
            with torch.no_grad():
                for tp,sp in zip(teacher.parameters(),student.parameters()): tp.mul_(cfg['ema_decay']).add_(sp,alpha=1-cfg['ema_decay'])
        vm=evaluate(student,val,device); history.append({'phase':'student','epoch':ep+1,'loss':total/len(labeled),**vm})
    final=evaluate(student,val,device); final.update({'mode':'synthetic_smoke','runtime_seconds':time.time()-start,'device':'cpu','labeled_train':cfg['train_labeled'],'unlabeled_train':cfg['train_unlabeled'],'validation':cfg['validation']})
    (out/'metrics.json').write_text(json.dumps(final,indent=2))
    with open(out/'history.csv','w',newline='') as f: wr=csv.DictWriter(f,fieldnames=history[0].keys()); wr.writeheader(); wr.writerows(history)
    (out/'run_manifest.json').write_text(json.dumps({'config':cfg,'model':'TinySemiTransformer','meaning':'software smoke only; not a TSI15k or paper result'},indent=2))
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(range(1,len(history)+1),[r['loss'] for r in history],marker='o',label='Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Smoke teacher/student training'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'training_curves.png',dpi=180); plt.close(fig)
    x,y=next(iter(val)); student.eval()
    with torch.no_grad(): p=torch.sigmoid(student(x))
    fig,axs=plt.subplots(3,3,figsize=(10,7))
    for i in range(3):
        axs[i,0].imshow(x[i,0],cmap='gray'); axs[i,0].set_title('Input'); axs[i,1].imshow(y[i,0],cmap='gray'); axs[i,1].set_title('Ground truth'); axs[i,2].imshow(p[i,0],cmap='gray'); axs[i,2].set_title('Prediction'); [a.axis('off') for a in axs[i]]
    fig.tight_layout(); fig.savefig(figdir/'predictions.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3)); ax.axis('off'); labels=['Labeled images','Teacher pretraining','Unlabeled pseudo-labels','Student training','EMA teacher']; xs=np.linspace(.08,.92,len(labels))
    for i,(xv,label) in enumerate(zip(xs,labels)):
        ax.text(xv,.5,label,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='none'))
        if i<len(labels)-1: ax.annotate('',xy=(xs[i+1]-.08,.5),xytext=(xv+.08,.5),arrowprops=dict(arrowstyle='->'))
    fig.tight_layout(); fig.savefig(figdir/'teacher_student_workflow.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3)); ax.axis('off'); ax.text(.5,.75,'Conv patch embedding → Transformer encoder → convolutional decoder',ha='center',bbox=dict(boxstyle='round',facecolor='none')); ax.text(.5,.35,'CPU-safe smoke architecture (not the full SemiTNet network)',ha='center'); fig.tight_layout(); fig.savefig(figdir/'architecture.png',dpi=180); plt.close(fig)
    print(json.dumps(final,indent=2))
if __name__=='__main__': main()
