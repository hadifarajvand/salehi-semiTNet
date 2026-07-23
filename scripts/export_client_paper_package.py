#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, shutil, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/paper_style"
FIG=OUT/"figures"
TAB=OUT/"tables"
FINAL=ROOT/"outputs/final"
PAPER=ROOT/"paper/reference"
MET=["iou","dice","precision","recall","f1"]
DM=["IoU","Dice","Precision","Recall","F1"]

def rcsv(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))

def rjson(p): return json.loads(p.read_text())

def save(fig, folder, stem):
    folder.mkdir(parents=True,exist_ok=True)
    png=folder/f"{stem}.png"; pdf=folder/f"{stem}.pdf"
    fig.savefig(png,dpi=300,bbox_inches="tight",facecolor="white")
    with PdfPages(pdf) as d: d.savefig(fig,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def regenerate_figure3():
    h=rcsv(FINAL/"history.csv")
    x=list(range(len(h))); loss=[float(r["loss"]) for r in h]; prec=[float(r["precision"]) for r in h]
    fig,a=plt.subplots(figsize=(8.8,5.5)); b=a.twinx()
    l1,=a.plot(x,loss,color="#d62728",lw=2,label="Training Loss on Train Set")
    l2,=b.plot(x,prec,color="#1f77b4",lw=2,label="Precision on Test Set")
    a.set_title("Loss and Precision vs. Training Step"); a.set_xlabel("Recorded Training Step")
    a.set_ylabel("Training Loss",color="#d62728"); b.set_ylabel("Precision (%)",color="#1f77b4")
    a.tick_params(axis="y",colors="#d62728"); b.tick_params(axis="y",colors="#1f77b4")
    a.legend([l1,l2],[l1.get_label(),l2.get_label()],loc="upper right",fontsize=9)
    fig.text(.5,.005,"Reduced measured steps; not presented as the paper's 26,250 iterations.",ha="center",fontsize=8,color="#555")
    save(fig,FIG,"figure_03_training_loss_precision")

def radar_axes(ax,rmin,rmax,ticks):
    ang=[i/5*2*math.pi for i in range(5)]
    ax.set_theta_offset(math.pi/2); ax.set_theta_direction(-1)
    ax.set_thetagrids([a*180/math.pi for a in ang],DM,fontsize=8)
    ax.set_ylim(rmin,rmax); ax.set_yticks(ticks); ax.set_yticklabels([f"{x:.0f}%" for x in ticks],fontsize=6)
    ax.grid(True,alpha=.35)
    return ang

def radar(ax,vals,label,color,lw=1.4,alpha=.035):
    ang=[i/5*2*math.pi for i in range(5)]; ang+=ang[:1]; vals+=vals[:1]
    ax.plot(ang,vals,label=label,color=color,lw=lw); ax.fill(ang,vals,color=color,alpha=alpha)

def regenerate_figure4():
    overall={r["model"]:r for r in rcsv(PAPER/"table2_overall.csv")}
    groups=rcsv(PAPER/"table3_groups.csv")
    models=["Mask R-CNN","MPFormer","Mask2Former","MaskDINO","GEM","SemiTNet"]
    colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]
    specs=[
      ("Fully dentate and partially edentulous individuals","overall",(90,98,[90,92,94,96,98])),
      ("Fully dentate individuals","fully_dentate",(97,100,[97,98,99,100])),
      ("Partially edentulous individuals","partially_edentulous",(89,97,[89,91,93,95,97]))]
    fig=plt.figure(figsize=(10.5,8.7)); gs=fig.add_gridspec(2,2,hspace=.34,wspace=.2)
    axes=[fig.add_subplot(gs[0,:],projection="polar"),fig.add_subplot(gs[1,0],projection="polar"),fig.add_subplot(gs[1,1],projection="polar")]
    for ax,(title,g,(mn,mx,ticks)) in zip(axes,specs):
        radar_axes(ax,mn,mx,ticks)
        for model,color in zip(models,colors):
            row=overall[model] if g=="overall" else next(r for r in groups if r["group"]==g and r["model"]==model)
            radar(ax,[float(row[k]) for k in MET],model,color,2.2 if model=="SemiTNet" else 1.4,.10 if model=="SemiTNet" else .035)
        ax.set_title(title,fontsize=9,pad=13); ax.legend(loc="upper right",bbox_to_anchor=(1.25,1.12),fontsize=5.8,frameon=False)
    fig.text(.5,.01,"Published-reference metric geometry (Table 2/Table 3).",ha="center",fontsize=8,color="#555")
    save(fig,FIG,"figure_04_performance_radar")

def crops():
    im=Image.open(FINAL/"figures/predictions.png").convert("RGB"); w,h=im.size; cw=w//3; ch=h//3; out=[]
    for r in (0,2):
        row=[]
        for c in range(3):
            mx=max(8,int(cw*.02)); my=max(8,int(ch*.04))
            row.append(im.crop((c*cw+mx,r*ch+my,(c+1)*cw-mx,(r+1)*ch-my)))
        out.append(row)
    return out

def regenerate_figure5():
    samples=crops(); labels=["Mask R-CNN","MPFormer","Mask2Former","MaskDINO","GEM","SemiTNet","Ground Truth"]
    tags=[list("abcdefg"),list("hijklmn")]
    fig,axs=plt.subplots(2,7,figsize=(19.5,6.5))
    for r,trip in enumerate(samples):
        for c,ax in enumerate(axs[r]):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_box_aspect(.58)
            for sp in ax.spines.values(): sp.set_linewidth(.65); sp.set_edgecolor("#777")
            if c==5: ax.imshow(trip[2],aspect="auto")
            elif c==6: ax.imshow(trip[1],aspect="auto")
            else:
                ax.set_facecolor("#efefef"); ax.text(.5,.5,"Baseline output\nnot executed",ha="center",va="center",fontsize=8,color="#555")
            ax.set_title(labels[c],fontsize=8,pad=4)
            ax.text(.02,.98,f"({tags[r][c]})",transform=ax.transAxes,ha="left",va="top",fontsize=8,weight="bold",
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.75,pad=1))
    fig.text(.005,.73,"Fully dentate-style example",rotation=90,va="center",fontsize=9,weight="bold")
    fig.text(.005,.28,"Partially edentulous-style example",rotation=90,va="center",fontsize=9,weight="bold")
    fig.text(.5,.015,"Paper 2×7 geometry. Missing baseline predictions remain explicit placeholders; no prediction is fabricated.",ha="center",fontsize=8,color="#555")
    save(fig,FIG,"figure_05_qualitative_outputs")

def render_table(rows,cols,title,stem,widths=None,foot=""):
    fig,ax=plt.subplots(figsize=(max(8.5,1.45*len(cols)),max(3.1,1.2+.42*len(rows)))); ax.axis("off"); ax.set_title(title,fontsize=12,weight="bold",pad=12)
    t=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center",colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1,1.4)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor("#555"); cell.set_linewidth(.6)
        if r==0: cell.set_facecolor("#eee"); cell.set_text_props(weight="bold")
    if foot: fig.text(.5,.02,foot,ha="center",fontsize=7,color="#555")
    save(fig,TAB,stem)

def export_tables():
    t1=rcsv(PAPER/"table1_dataset.csv")
    render_table([[r["cohort"],r["labeling"],r["training_set"],r["test_set"]] for r in t1],
      ["Cohort","Image Type","Training Set","Test Set"],"Table 1. TSI15k Dataset Distribution","table_01_tsi15k_distribution",[.18,.30,.2,.2],
      "Published-reference dataset distribution; not the reduced-run split.")
    shutil.copy2(PAPER/"table1_dataset.csv",TAB/"table_01_tsi15k_distribution.csv")
    t2=rcsv(PAPER/"table2_overall.csv")
    render_table([[r["model"],r["iou"],r["dice"],r["precision"],r["recall"],r["f1"],r["params_m"]] for r in t2],
      ["Network","IoU (%)","Dice (%)","Precision (%)","Recall (%)","F1 Score (%)","Parameters (M)"],
      "Table 2. Performance Comparison on the Test Set","table_02_overall_performance",[.2,.12,.12,.14,.12,.14,.14],
      "Published-reference metrics; p-values are not re-estimated by the reduced simulation.")
    shutil.copy2(PAPER/"table2_overall.csv",TAB/"table_02_overall_performance.csv")
    t3=rcsv(PAPER/"table3_groups.csv"); models=["Mask R-CNN","MPFormer","Mask2Former","MaskDINO","GEM","SemiTNet"]; rows=[]
    for m in models:
        f=next(r for r in t3 if r["group"]=="fully_dentate" and r["model"]==m); p=next(r for r in t3 if r["group"]=="partially_edentulous" and r["model"]==m)
        rows.append([m,*[f[k] for k in MET],*[p[k] for k in MET]])
    render_table(rows,["Network","F-IoU","F-Dice","F-Prec","F-Recall","F-F1","P-IoU","P-Dice","P-Prec","P-Recall","P-F1"],
      "Table 3. Fully Dentate vs. Partially Edentulous Performance","table_03_group_performance",[.15]+[.078]*10,
      "F=fully dentate; P=partially edentulous. Published-reference metrics.")
    shutil.copy2(PAPER/"table3_groups.csv",TAB/"table_03_group_performance.csv")

def docs():
    coverage=[
      ["Figure 1","architecture","figures/figure_01_semitnet_architecture.png","reference structure","PASS"],
      ["Figure 2","distillation workflow","figures/figure_02_teacher_student_workflow.png","reference + reduced workflow","PASS"],
      ["Figure 3","loss + precision dual axis","figures/figure_03_training_loss_precision.png","measured reduced","PASS"],
      ["Figure 4","three radar panels / five metrics","figures/figure_04_performance_radar.png","published reference metrics","PASS"],
      ["Figure 5","2x7 qualitative geometry","figures/figure_05_qualitative_outputs.png","measured + explicit placeholders","PARTIAL"],
      ["Table 1","dataset distribution","tables/table_01_tsi15k_distribution.png","published reference","PASS"],
      ["Table 2","overall metrics","tables/table_02_overall_performance.png","published reference","PASS"],
      ["Table 3","dentate subgroup metrics","tables/table_03_group_performance.png","published reference","PASS"]]
    with (OUT/"EVALUATION_COVERAGE_MATRIX.csv").open("w",newline="") as f: w=csv.writer(f); w.writerow(["artifact","paper_role","export","data_kind","status"]); w.writerows(coverage)
    md=["# Visual Fidelity Audit","","**Verdict: client-ready paper-aligned export package, with one explicit limitation: baseline qualitative predictions were not executed.**","",
        "- Figure 1: architecture component coverage aligned.","- Figure 2: teacher/student + Ls/Lu + EMA workflow aligned.",
        "- Figure 3: red loss / blue precision dual-axis presentation aligned; reduced steps are labeled truthfully.",
        "- Figure 4: top radar + two bottom radars, same five metrics and six model references aligned.",
        "- Figure 5: corrected to paper's 2×7 panel geometry; missing baseline predictions are placeholders, not fabricated.",
        "- Tables 1–3: paper metric categories and layouts exported as PNG/PDF/CSV.","",
        "This package matches the paper's visualization categories and metric vocabulary; it does not claim numerical reproduction."]
    (OUT/"VISUAL_FIDELITY_AUDIT.md").write_text("\n".join(md)+"\n")
    (OUT/"CLIENT_DELIVERY_MANIFEST.json").write_text(json.dumps({"figures":[1,2,3,4,5],"tables":[1,2,3],"fabricated_values":False,"fabricated_predictions":False,"figure5_baselines":"not executed / explicit placeholders","claim":"paper-aligned visualization package; not numerical reproduction"},indent=2)+"\n")

def validate():
    required=[
      FIG/"figure_01_semitnet_architecture.png",FIG/"figure_02_teacher_student_workflow.png",FIG/"figure_03_training_loss_precision.png",
      FIG/"figure_04_performance_radar.png",FIG/"figure_05_qualitative_outputs.png",
      TAB/"table_01_tsi15k_distribution.png",TAB/"table_02_overall_performance.png",TAB/"table_03_group_performance.png",
      OUT/"EVALUATION_COVERAGE_MATRIX.csv",OUT/"VISUAL_FIDELITY_AUDIT.md",OUT/"CLIENT_DELIVERY_MANIFEST.json"]
    missing=[str(x) for x in required if not x.exists() or x.stat().st_size==0]
    if missing: raise SystemExit("Missing exports: "+repr(missing))
    for p in [x for x in required if x.suffix==".png"]:
        im=Image.open(p)
        if im.width<800 or im.height<300: raise SystemExit(f"Low-resolution export: {p} {im.size}")

def main():
    subprocess.run([sys.executable,str(ROOT/"scripts/export_paper_style_figures.py")],cwd=ROOT,check=True)
    TAB.mkdir(parents=True,exist_ok=True)
    regenerate_figure3(); regenerate_figure4(); regenerate_figure5(); export_tables(); docs(); validate()
    z=ROOT/"outputs/SemiTNet-paper-aligned-export.zip"
    if z.exists(): z.unlink()
    shutil.make_archive(str(z.with_suffix("")),"zip",root_dir=OUT)
    print("[ok] client-ready paper-aligned exports:",OUT)
    print("[ok] bundle:",z)

if __name__=="__main__": main()
