#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/paper_style'
FIG_DIR = OUT / 'figures'
FINAL = ROOT / 'outputs/final'
PAPER = ROOT / 'paper/reference'


def read_json(path: Path):
    return json.loads(path.read_text())


def ensure_clean_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_png_pdf(fig, png_path: Path, pdf_path: Path):
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', facecolor='white')


def box(ax, xy, wh, text, fc='#f5f7fb', ec='#2f3b52', fs=11, weight='normal'):
    x, y = xy
    w, h = wh
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.02',
                                  linewidth=1.4, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight=weight, color='#182233')


def arrow(ax, start, end, color='#2f3b52', style='-|>', lw=1.6):
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=0, shrinkB=0))


def figure1():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Figure 1. Implemented reduced-scale SemiTNet architecture', fontsize=16, weight='bold', pad=16)

    groups = [
        (0.03, 0.18, 0.18, 0.64, 'Image Encoder'),
        (0.25, 0.18, 0.22, 0.64, 'Feature Representation'),
        (0.50, 0.18, 0.22, 0.64, 'Query / Transformer\nProcessing'),
        (0.75, 0.18, 0.22, 0.64, 'Mask / Class Decoder'),
    ]
    for x, y, w, h, title in groups:
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, linewidth=2.0, edgecolor='#50607a'))
        ax.text(x + w / 2, y + h + 0.03, title, ha='center', va='bottom', fontsize=12, weight='bold')

    box(ax, (0.05, 0.54), (0.14, 0.12), 'Panoramic\nRadiograph', fc='#fff4d6')
    box(ax, (0.05, 0.30), (0.14, 0.12), 'Patch / Image\nEncoder', fc='#d8ecff')

    box(ax, (0.28, 0.50), (0.16, 0.14), 'QuickSemiTransformer', fc='#dff6ea', weight='bold')
    box(ax, (0.28, 0.30), (0.16, 0.12), 'Spatial Positional\nEncoding', fc='#f0e7ff')

    box(ax, (0.53, 0.52), (0.16, 0.12), 'Transformer\nEncoder', fc='#f0e7ff')
    box(ax, (0.53, 0.30), (0.16, 0.12), 'Feature\nRepresentation', fc='#dff6ea')

    box(ax, (0.78, 0.52), (0.16, 0.12), 'Segmentation\nDecoder', fc='#d8ecff')
    box(ax, (0.78, 0.30), (0.16, 0.12), '32 Tooth Classes\n+ Background', fc='#fff4d6', weight='bold')
    ax.text(0.86, 0.12, 'Tooth Segmentation\nand Identification', ha='center', va='center', fontsize=13, weight='bold')

    arrow(ax, (0.19, 0.60), (0.28, 0.60))
    arrow(ax, (0.18, 0.36), (0.28, 0.36))
    arrow(ax, (0.44, 0.57), (0.53, 0.57))
    arrow(ax, (0.44, 0.36), (0.53, 0.36))
    arrow(ax, (0.69, 0.58), (0.78, 0.58))
    arrow(ax, (0.69, 0.36), (0.78, 0.36))
    arrow(ax, (0.86, 0.30), (0.86, 0.18))

    ax.text(0.5, 0.05, 'Implemented modules are mapped onto the same four-stage organization as the published framework.',
            ha='center', va='center', fontsize=10, color='#44546a')
    return fig


def figure2():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Figure 2. Teacher–student semi-supervised workflow', fontsize=16, weight='bold', pad=16)

    box(ax, (0.05, 0.72), (0.15, 0.1), '60 labeled\nimages', fc='#fff4d6')
    box(ax, (0.25, 0.70), (0.18, 0.12), 'Teacher\nsupervised\ntraining', fc='#dff6ea', weight='bold')
    arrow(ax, (0.20, 0.77), (0.25, 0.77))

    box(ax, (0.05, 0.47), (0.15, 0.1), '20 label-hidden\nimages', fc='#fff4d6')
    box(ax, (0.25, 0.45), (0.18, 0.12), 'Teacher\ninference', fc='#d8ecff')
    box(ax, (0.46, 0.45), (0.15, 0.12), 'Confidence\nfiltering', fc='#f0e7ff')
    box(ax, (0.64, 0.45), (0.14, 0.12), 'Pseudo-labels', fc='#dff6ea')
    arrow(ax, (0.20, 0.52), (0.25, 0.52))
    arrow(ax, (0.43, 0.51), (0.46, 0.51))
    arrow(ax, (0.61, 0.51), (0.64, 0.51))

    box(ax, (0.10, 0.17), (0.14, 0.1), 'Labeled\nimages', fc='#fff4d6')
    box(ax, (0.28, 0.10), (0.18, 0.22), 'Student', fc='#d8ecff', weight='bold')
    box(ax, (0.50, 0.10), (0.22, 0.22), 'Supervised +\nunsupervised loss', fc='#f0e7ff')
    box(ax, (0.76, 0.10), (0.12, 0.22), 'Student\nupdate', fc='#dff6ea')
    box(ax, (0.90, 0.10), (0.08, 0.22), 'Teacher\nupdate', fc='#fff4d6')
    box(ax, (0.90, 0.36), (0.08, 0.1), 'EMA', fc='#fff4d6')
    box(ax, (0.83, 0.72), (0.14, 0.1), '16 held-out\nimages', fc='#fff4d6')
    box(ax, (0.83, 0.52), (0.14, 0.12), 'Final\nevaluation', fc='#d8ecff', weight='bold')

    arrow(ax, (0.24, 0.22), (0.28, 0.22))
    arrow(ax, (0.46, 0.21), (0.50, 0.21))
    arrow(ax, (0.72, 0.21), (0.76, 0.21))
    arrow(ax, (0.88, 0.22), (0.90, 0.22))
    arrow(ax, (0.26, 0.78), (0.26, 0.22), style='->')
    arrow(ax, (0.85, 0.72), (0.85, 0.64))
    arrow(ax, (0.82, 0.22), (0.90, 0.22))
    arrow(ax, (0.94, 0.22), (0.94, 0.36), style='->')
    arrow(ax, (0.94, 0.36), (0.72, 0.36), style='->')
    ax.text(0.60, 0.36, 'Teacher weights ← EMA ← Student weights', ha='center', va='center', fontsize=11, color='#44546a')
    ax.text(0.50, 0.04, 'Workflow follows the measured reduced-scale simulation; no subgroup metrics were re-estimated.',
            ha='center', va='center', fontsize=10, color='#44546a')
    return fig


def load_history():
    rows = []
    with (FINAL / 'history.csv').open() as f:
        for row in csv.DictReader(f):
            row['epoch'] = int(row['epoch'])
            for k in ['loss', 'iou', 'dice', 'precision', 'recall', 'f1']:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def figure3():
    rows = load_history()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    phases = [r['phase'] for r in rows]
    x = list(range(1, len(rows) + 1))
    losses = [r['loss'] for r in rows]
    precisions = [r['precision'] for r in rows]
    split = phases.index('student') + 1 if 'student' in phases else len(rows)

    for ax in axes:
        ax.axvspan(0.5, split + 0.5, color='#eaf2ff', alpha=0.9, zorder=0)
        ax.axvspan(split + 0.5, len(rows) + 0.5, color='#eef9f1', alpha=0.9, zorder=0)
        ax.axvline(split + 0.5, color='#5a6472', lw=1.2, ls='--')
        ax.grid(True, axis='y', alpha=0.25)

    axes[0].plot(x, losses, marker='o', lw=2.2, color='#1f4b99')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training loss', fontsize=13, weight='bold')
    axes[1].plot(x, precisions, marker='o', lw=2.2, color='#1b7f5a')
    axes[1].set_ylabel('Precision (%)')
    axes[1].set_xlabel('Training stage step')
    axes[1].set_title('Precision variation on test set', fontsize=13, weight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{r['phase'][0].upper()}{r['epoch']}" for r in rows])
    axes[0].text(0.02, 0.90, 'Teacher stage', transform=axes[0].transAxes, fontsize=10, color='#315ea9')
    axes[0].text(0.63, 0.90, 'Student stage', transform=axes[0].transAxes, fontsize=10, color='#2f8f64')
    fig.suptitle('Figure 3. Training loss curve and precision variation measured during the reduced-scale SemiTNet simulation.',
                 fontsize=15, weight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def radar_axes(ax, categories):
    angles = [n / len(categories) * 2 * math.pi for n in range(len(categories))]
    angles += angles[:1]
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8)
    ax.grid(True, alpha=0.3)
    return angles


def plot_radar(ax, values, label, color, ls='-', lw=2.1, alpha=0.18, z=2):
    cats = ['IoU', 'Dice', 'Precision', 'Recall', 'F1']
    angles = [n / len(cats) * 2 * math.pi for n in range(len(cats))]
    vals = values + values[:1]
    ax.plot(angles + angles[:1], vals, color=color, lw=lw, ls=ls, label=label, zorder=z)
    ax.fill(angles + angles[:1], vals, color=color, alpha=alpha, zorder=z - 1)


def figure4():
    overall = {}
    with (PAPER / 'table2_overall.csv').open() as f:
        for row in csv.DictReader(f):
            overall[row['model']] = row
    groups = []
    with (PAPER / 'table3_groups.csv').open() as f:
        for row in csv.DictReader(f):
            groups.append(row)
    measured = read_json(FINAL / 'metrics.json')
    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 6))
    cats = ['IoU', 'Dice', 'Precision', 'Recall', 'F1']
    refs = ['Mask R-CNN', 'MPFormer', 'Mask2Former', 'MaskDINO', 'GEM', 'SemiTNet']
    colors = ['#8a8f98', '#6d79b3', '#66a61e', '#e6ab02', '#e7298a', '#1f4b99']
    panel_specs = [
        ('(a) Overall test set', 'overall', True),
        ('(b) Fully dentate', 'fully_dentate', False),
        ('(c) Partially edentulous', 'partially_edentulous', False),
    ]
    for ax, (title, group, include_measured) in zip(axs, panel_specs):
        radar_axes(ax, cats)
        if group == 'overall':
            for label, color in zip(refs, colors):
                row = overall[label]
                vals = [float(row[c.lower()]) for c in cats]
                ls = '-' if label == 'SemiTNet' else '--'
                plot_radar(ax, vals, f'Published reference — {label}', color, ls=ls, lw=2.5 if label == 'SemiTNet' else 1.7, alpha=0.12 if label == 'SemiTNet' else 0.06)
            vals = [float(measured[k.lower()]) for k in cats]
            plot_radar(ax, vals, 'Measured reduced simulation', '#d62728', ls='-', lw=3.0, alpha=0.18, z=4)
        else:
            filtered = [r for r in groups if r['group'] == group]
            for label, color in zip(refs, colors):
                row = next(r for r in filtered if r['model'] == label)
                vals = [float(row[c.lower()]) for c in cats]
                ls = '-' if label == 'SemiTNet' else '--'
                plot_radar(ax, vals, f'Published reference — {label}', color, ls=ls, lw=2.5 if label == 'SemiTNet' else 1.7, alpha=0.12 if label == 'SemiTNet' else 0.06)
            ax.text(0.5, -0.22, 'Reduced simulation subgroup metrics not separately measured.\nReference traces shown for visual comparison.',
                    transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#5a6472')
        ax.set_title(title, fontsize=13, weight='bold', pad=18)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Figure 4. Radar-chart comparison following the published SemiTNet figure design.', fontsize=15, weight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig


def trim_whitespace(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    rgb = img.convert('RGB')
    bg_img = Image.new(rgb.mode, rgb.size, bg)
    diff = ImageChops.difference(rgb, bg_img) if False else None
    return img


def crop_panel(img: Image.Image, box):
    return img.crop(box)


def figure5():
    src = Image.open(FINAL / 'figures/predictions.png').convert('RGB')
    w, h = src.size
    cell_w = w // 3
    cell_h = h // 3
    pads = 18
    panels = []
    for r in range(3):
        row = []
        for c in range(3):
            left = c * cell_w + 10
            upper = r * cell_h + 10
            right = (c + 1) * cell_w - 10
            lower = (r + 1) * cell_h - 10
            row.append(src.crop((left, upper, right, lower)))
        panels.append(row)

    fig, axs = plt.subplots(3, 3, figsize=(14, 12))
    labels = [['(a) Panoramic radiograph', '(b) Ground truth', '(c) SemiTNet simulation prediction'],
              ['(d) Panoramic radiograph', '(e) Ground truth', '(f) SemiTNet simulation prediction'],
              ['(g) Panoramic radiograph', '(h) Ground truth', '(i) SemiTNet simulation prediction']]
    for r in range(3):
        for c in range(3):
            ax = axs[r, c]
            ax.imshow(panels[r][c])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(labels[r][c], fontsize=10, pad=6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor('#444')
    fig.suptitle('Figure 5. Representative qualitative tooth segmentation and identification outputs from the measured reduced-scale SemiTNet simulation with corresponding ground truth.',
                 fontsize=15, weight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def write_comparison(measured):
    paper = read_json(PAPER / 'paper_targets.json')
    txt = []
    txt.append('# Figure comparison')
    txt.append('')
    txt.append('| Paper Figure | Paper content | Our closest output | Match status |')
    txt.append('|---|---|---|---|')
    rows = [
        ('Figure 1', 'Architecture diagram', 'figure_01_semitnet_architecture.png', 'architecture concept reproduced; visual/style match strong; execution architecture reduced, not identical'),
        ('Figure 2', 'Teacher/student distillation', 'figure_02_teacher_student_workflow.png', 'teacher/student workflow reproduced; conceptual match strong'),
        ('Figure 3', 'Training loss + precision curve', 'figure_03_training_loss_precision.png', 'loss + precision generated directly from measured history; data provenance fully measured; match strong in figure type'),
        ('Figure 4', 'Radar comparison', 'figure_04_performance_radar.png', 'paper radar-chart style reproduced; overall measured simulation included; subgroup simulation unavailable; published subgroup data shown only as reference; match strong visually, partial experimentally'),
        ('Figure 5', 'Qualitative outputs', 'figure_05_qualitative_outputs.png', 'qualitative prediction/ground-truth layout available; only SemiTNet simulation executed; multi-network paper comparison not reproduced; match moderate visually'),
    ]
    for row in rows:
        txt.append(f'| {row[0]} | {row[1]} | `{row[2]}` | {row[3]} |')
    txt.append('')
    txt.append('## Metric comparison')
    txt.append('')
    txt.append('| Metric | Paper SemiTNet | Reduced simulation | Difference |')
    txt.append('|---|---:|---:|---:|')
    for metric in ['iou', 'dice', 'precision', 'recall', 'f1']:
        pv = paper['overall'][metric]
        sv = measured[metric]
        txt.append(f'| {metric.upper()} | {pv:.2f} | {sv:.8f} | {sv - pv:+.8f} |')
    txt.append('')
    txt.append('The reduced simulation is not numerically equivalent to the full-scale paper experiment. The exported figures reproduce the paper’s presentation structure and experimental visualization categories using the available measured outputs.')
    (OUT / 'FIGURE_COMPARISON.md').write_text('\n'.join(txt) + '\n')


def main():
    ensure_clean_dir(FIG_DIR)
    metrics = read_json(FINAL / 'metrics.json')
    history = list(csv.DictReader((FINAL / 'history.csv').open()))
    manifest = read_json(FINAL / 'run_manifest.json')

    figs = [figure1(), figure2(), figure3(), figure4(), figure5()]
    names = [
        'figure_01_semitnet_architecture',
        'figure_02_teacher_student_workflow',
        'figure_03_training_loss_precision',
        'figure_04_performance_radar',
        'figure_05_qualitative_outputs',
    ]
    for fig, name in zip(figs, names):
        save_png_pdf(fig, FIG_DIR / f'{name}.png', FIG_DIR / f'{name}.pdf')
        plt.close(fig)

    write_comparison(metrics)
    provenance = {
        'dataset_manifest': str(ROOT / 'data/processed/quick_teeth/split_manifest.json'),
        'source_metrics': 'outputs/paper_style/source_metrics.json',
        'source_history': 'outputs/paper_style/source_history.csv',
        'source_run_manifest': 'outputs/paper_style/source_run_manifest.json',
        'generated_figures': [
            {
                'figure': 1,
                'path': 'outputs/paper_style/figures/figure_01_semitnet_architecture.png',
                'paper_counterpart': 'Figure 1',
                'source_files': ['outputs/final/run_manifest.json'],
                'data_kind': 'structural',
                'fabricated_values': False,
            },
            {
                'figure': 2,
                'path': 'outputs/paper_style/figures/figure_02_teacher_student_workflow.png',
                'paper_counterpart': 'Figure 2',
                'source_files': ['outputs/final/run_manifest.json', 'outputs/final/metrics.json'],
                'data_kind': 'structural',
                'fabricated_values': False,
            },
            {
                'figure': 3,
                'path': 'outputs/paper_style/figures/figure_03_training_loss_precision.png',
                'paper_counterpart': 'Figure 3',
                'source_files': ['outputs/final/history.csv'],
                'data_kind': 'measured',
                'fabricated_values': False,
            },
            {
                'figure': 4,
                'path': 'outputs/paper_style/figures/figure_04_performance_radar.png',
                'paper_counterpart': 'Figure 4',
                'source_files': ['outputs/final/metrics.json', 'paper/reference/table2_overall.csv', 'paper/reference/table3_groups.csv', 'paper/reference/paper_targets.json'],
                'data_kind': 'published_reference_plus_measured',
                'fabricated_values': False,
            },
            {
                'figure': 5,
                'path': 'outputs/paper_style/figures/figure_05_qualitative_outputs.png',
                'paper_counterpart': 'Figure 5',
                'source_files': ['outputs/final/figures/predictions.png'],
                'data_kind': 'measured',
                'fabricated_values': False,
            },
        ],
    }
    (OUT / 'figure_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    shutil.copy2(FINAL / 'metrics.json', OUT / 'source_metrics.json')
    shutil.copy2(FINAL / 'history.csv', OUT / 'source_history.csv')
    shutil.copy2(FINAL / 'run_manifest.json', OUT / 'source_run_manifest.json')

    comparison = [
        '# Paper-style figure export',
        '',
        f"Generated from measured outputs in `outputs/final/` with {len(history)} history rows.",
        '',
        '## Output set',
        '- `outputs/paper_style/figures/figure_01_semitnet_architecture.png`',
        '- `outputs/paper_style/figures/figure_02_teacher_student_workflow.png`',
        '- `outputs/paper_style/figures/figure_03_training_loss_precision.png`',
        '- `outputs/paper_style/figures/figure_04_performance_radar.png`',
        '- `outputs/paper_style/figures/figure_05_qualitative_outputs.png`',
        '',
        '## Provenance',
        '- `outputs/final/metrics.json`',
        '- `outputs/final/history.csv`',
        '- `outputs/final/run_manifest.json`',
        '- `outputs/final/figures/predictions.png`',
        '- `paper/reference/table2_overall.csv`',
        '- `paper/reference/table3_groups.csv`',
        '- `paper/reference/paper_targets.json`',
        '',
        '## Constraint',
        'No training or inference rerun. No measured metric rewrite.',
    ]
    (OUT / 'FIGURE_COMPARISON.md').write_text('\n'.join(comparison) + '\n')
    print(f'[ok] wrote figures to {FIG_DIR}')


if __name__ == '__main__':
    main()
