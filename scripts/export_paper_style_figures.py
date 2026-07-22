#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

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
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.86])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.98, 'Published SemiTNet reference architecture', ha='center', va='top', fontsize=13, weight='bold')

    xs = [0.03, 0.19, 0.35, 0.50, 0.66, 0.82]
    widths = [0.13, 0.13, 0.11, 0.11, 0.11, 0.13]
    labels = [
        'Panoramic\nradiograph',
        'Image\nEncoder',
        'Simple / Basic\nFeature Pyramid',
        'Query\nInitialization Unit',
        'Mask\nDecoder',
        'Predictions\n(class / box / mask)',
    ]
    colors = ['#fff4d6', '#d8ecff', '#e8f6ea', '#f0e7ff', '#d8ecff', '#fff4d6']
    for x, w, label, color in zip(xs, widths, labels, colors):
        box(ax, (x, 0.42), (w, 0.18), label, fc=color, fs=10 if 'Predictions' not in label else 9, weight='bold')
    arrow(ax, (0.16, 0.51), (0.19, 0.51))
    arrow(ax, (0.32, 0.51), (0.35, 0.51))
    arrow(ax, (0.46, 0.51), (0.50, 0.51))
    arrow(ax, (0.61, 0.51), (0.66, 0.51))
    arrow(ax, (0.77, 0.51), (0.82, 0.51))

    ax.text(0.42, 0.22, 'ViT-Tiny / MobileSAM reference\n+ multi-scale features C2–C5', ha='center', va='center', fontsize=10)
    ax.text(0.56, 0.70, 'confidence ranking\nTop-k query selection', ha='center', va='center', fontsize=9)
    ax.text(0.69, 0.22, '9 decoder layers\ncross-attention / refinement', ha='center', va='center', fontsize=9)
    ax.text(0.17, 0.18, 'C2=1/4', fontsize=9)
    ax.text(0.17, 0.12, 'C3=1/8', fontsize=9)
    ax.text(0.17, 0.06, 'C4=1/16', fontsize=9)
    ax.text(0.17, 0.00, 'C5=1/32', fontsize=9)
    ax.text(0.5, 0.92, 'Published SemiTNet reference architecture; reduced executable simulation uses a simplified Transformer implementation.',
            ha='center', va='center', fontsize=9, color='#55667a')
    return fig


def figure1b():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box(ax, (0.05, 0.60), (0.18, 0.16), 'Panoramic\nRadiograph', fc='#fff4d6')
    box(ax, (0.28, 0.58), (0.22, 0.20), 'QuickSemiTransformer\nreduced implementation', fc='#d8ecff', weight='bold')
    box(ax, (0.56, 0.58), (0.16, 0.20), 'Transformer\nencoder', fc='#f0e7ff')
    box(ax, (0.77, 0.58), (0.18, 0.20), 'Segmentation /\nclassification heads', fc='#e8f6ea')
    arrow(ax, (0.23, 0.68), (0.28, 0.68))
    arrow(ax, (0.50, 0.68), (0.56, 0.68))
    arrow(ax, (0.72, 0.68), (0.77, 0.68))
    ax.text(0.5, 0.25, 'Reduced executable simulation architecture documented for provenance.', ha='center', fontsize=9)
    return fig


def figure2():
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.98, 'Published SemiTNet distillation workflow', ha='center', va='top', fontsize=13, weight='bold')

    box(ax, (0.05, 0.72), (0.18, 0.10), 'Original unlabeled\npanoramic', fc='#fff4d6')
    box(ax, (0.28, 0.70), (0.12, 0.12), 'Teacher', fc='#dff6ea', weight='bold')
    box(ax, (0.46, 0.70), (0.14, 0.12), 'Pseudo-label', fc='#d8ecff')
    box(ax, (0.67, 0.70), (0.14, 0.12), 'Lu comparison', fc='#f0e7ff')
    arrow(ax, (0.23, 0.77), (0.28, 0.77))
    arrow(ax, (0.40, 0.77), (0.46, 0.77))
    arrow(ax, (0.60, 0.77), (0.67, 0.77))

    box(ax, (0.05, 0.45), (0.18, 0.10), 'Unlabeled\npanoramic', fc='#fff4d6')
    box(ax, (0.28, 0.43), (0.14, 0.12), 'Strong\naugmentation', fc='#f0e7ff')
    box(ax, (0.48, 0.43), (0.12, 0.12), 'Student', fc='#d8ecff', weight='bold')
    box(ax, (0.66, 0.43), (0.16, 0.12), 'Student\nprediction', fc='#dff6ea')
    box(ax, (0.86, 0.43), (0.10, 0.12), 'Lu', fc='#fff4d6')
    arrow(ax, (0.23, 0.50), (0.28, 0.50))
    arrow(ax, (0.42, 0.49), (0.48, 0.49))
    arrow(ax, (0.60, 0.49), (0.66, 0.49))
    arrow(ax, (0.82, 0.49), (0.86, 0.49))

    box(ax, (0.05, 0.18), (0.18, 0.10), 'Labeled panoramic\n+ ground truth', fc='#fff4d6')
    box(ax, (0.28, 0.16), (0.14, 0.12), 'Student', fc='#d8ecff', weight='bold')
    box(ax, (0.48, 0.16), (0.14, 0.12), 'Supervised\nloss Ls', fc='#f0e7ff')
    box(ax, (0.66, 0.16), (0.14, 0.12), 'Ls + λ Lu', fc='#dff6ea')
    box(ax, (0.86, 0.16), (0.10, 0.12), 'EMA\nTeacher', fc='#fff4d6')
    arrow(ax, (0.23, 0.23), (0.28, 0.23))
    arrow(ax, (0.42, 0.22), (0.48, 0.22))
    arrow(ax, (0.62, 0.22), (0.66, 0.22))
    arrow(ax, (0.80, 0.22), (0.86, 0.22))
    arrow(ax, (0.91, 0.22), (0.91, 0.55), style='->')
    arrow(ax, (0.91, 0.55), (0.86, 0.55), style='->')
    ax.text(0.50, 0.06, 'Reduced execution uses the same teacher–student workflow at reduced scale.', ha='center', fontsize=9, color='#55667a')
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
    fig, ax1 = plt.subplots(figsize=(11.5, 6.2))
    x = list(range(1, len(rows) + 1))
    losses = [r['loss'] for r in rows]
    precisions = [r['precision'] for r in rows]
    ax2 = ax1.twinx()
    ax1.plot(x, losses, color='#d62728', marker='o', lw=2.2, label='Training Loss')
    ax2.plot(x, precisions, color='#1f77b4', marker='o', lw=2.2, label='Precision (%)')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss', color='#d62728')
    ax2.set_ylabel('Precision (%)', color='#1f77b4')
    ax1.tick_params(axis='y', colors='#d62728')
    ax2.tick_params(axis='y', colors='#1f77b4')
    ax1.grid(True, axis='y', alpha=0.25)
    ax1.set_title('Loss and Precision vs. Training Step', fontsize=14, weight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in x])
    ax1.legend(handles=[ax1.lines[0], ax2.lines[0]], loc='upper right', frameon=True)
    phases = [r['phase'][0].upper() for r in rows]
    for xpos, lab in [(1, 'Teacher'), (len(rows), 'Student')]:
        ax1.text(xpos, ax1.get_ylim()[0], lab, ha='center', va='top', fontsize=9)
    return fig


def radar_axes(ax, categories, rmin, rmax, yticks):
    angles = [n / len(categories) * 2 * math.pi for n in range(len(categories))]
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / math.pi for a in angles], categories, fontsize=10)
    ax.set_ylim(rmin, rmax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{t:.0f}' for t in yticks], fontsize=8)
    ax.grid(True, alpha=0.3)
    return angles


def plot_radar(ax, categories, values, label, color, ls='-', lw=2.1, alpha=0.18, z=2):
    angles = [n / len(categories) * 2 * math.pi for n in range(len(categories))]
    angles = angles + angles[:1]
    vals = values + values[:1]
    ax.plot(angles, vals, color=color, lw=lw, ls=ls, label=label, zorder=z)
    ax.fill(angles, vals, color=color, alpha=alpha, zorder=z - 1)


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
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.20)
    axs = [fig.add_subplot(gs[0, :], projection='polar'), fig.add_subplot(gs[1, 0], projection='polar'), fig.add_subplot(gs[1, 1], projection='polar')]
    cats = ['IoU', 'Dice', 'Precision', 'Recall', 'F1']
    refs = ['Mask R-CNN', 'MPFormer', 'Mask2Former', 'MaskDINO', 'GEM', 'SemiTNet']
    colors = ['#8a8f98', '#6d79b3', '#66a61e', '#e6ab02', '#e7298a', '#1f4b99']
    panel_specs = [
        ('(a) Overall test set', 'overall', True),
        ('(b) Fully dentate', 'fully_dentate', False),
        ('(c) Partially edentulous', 'partially_edentulous', False),
    ]
    ranges = [(90, 98, [90, 92, 94, 96, 98]), (97, 100, [97, 98, 99, 100]), (89, 97, [89, 91, 93, 95, 97])]
    for ax, (title, group, include_measured), (rmin, rmax, yticks) in zip(axs, panel_specs, ranges):
        radar_axes(ax, cats, rmin, rmax, yticks)
        if group == 'overall':
            for label, color in zip(refs, colors):
                row = overall[label]
                vals = [float(row[c.lower()]) for c in cats]
                ls = '-' if label == 'SemiTNet' else '--'
                plot_radar(ax, cats, vals, f'{label}', color, ls=ls, lw=2.7 if label == 'SemiTNet' else 1.5, alpha=0.14 if label == 'SemiTNet' else 0.04)
        else:
            filtered = [r for r in groups if r['group'] == group]
            for label, color in zip(refs, colors):
                row = next(r for r in filtered if r['model'] == label)
                vals = [float(row[c.lower()]) for c in cats]
                ls = '-' if label == 'SemiTNet' else '--'
                plot_radar(ax, cats, vals, f'{label}', color, ls=ls, lw=2.7 if label == 'SemiTNet' else 1.5, alpha=0.14 if label == 'SemiTNet' else 0.04)
            ax.text(0.5, -0.22, 'Reduced simulation subgroup metrics not separately measured.\nReference traces shown for visual comparison.',
                    transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#5a6472')
        ax.set_title(title, fontsize=13, weight='bold', pad=18)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.42, 0.01))
    return fig


def figure4b():
    measured = read_json(FINAL / 'metrics.json')
    cats = ['IoU', 'Dice', 'Precision', 'Recall', 'F1']
    vals = [float(measured[k.lower()]) for k in cats]
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6.5, 5.8))
    radar_axes(ax, cats, 0, 100, [0, 25, 50, 75, 100])
    plot_radar(ax, cats, vals, 'Measured reduced simulation', '#d62728', lw=2.8, alpha=0.18)
    ax.set_title('Measured reduced simulation', fontsize=12, pad=14)
    return fig


def figure5():
    src = Image.open(FINAL / 'figures/predictions.png').convert('RGB')
    w, h = src.size
    cols, rows = 3, 3
    cell_w = w // cols
    cell_h = h // rows
    chosen_rows = [0, 2]
    fig, axs = plt.subplots(2, 8, figsize=(22, 8))
    labels = ['Input', 'Mask R-CNN', 'MPFormer', 'Mask2Former', 'MaskDINO', 'GEM', 'SemiTNet', 'Ground Truth']
    panel_labels = [['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'], ['(i)', '(j)', '(k)', '(l)', '(m)', '(n)', '(o)', '(p)']]
    for ridx, row in enumerate(chosen_rows):
        for c in range(cols):
            left = c * cell_w + 12
            upper = row * cell_h + 12
            right = (c + 1) * cell_w - 12
            lower = (row + 1) * cell_h - 12
            crop = src.crop((left, upper, right, lower))
            ax = axs[ridx, 0 if c == 0 else 6 if c == 2 else 7]
        # build row manually below
    # rebuild as 8 columns with placeholders, using extracted real panels for input, SemiTNet, ground truth
    for ridx, row in enumerate(chosen_rows):
        trip = []
        for c in range(cols):
            left = c * cell_w + 12
            upper = row * cell_h + 12
            right = (c + 1) * cell_w - 12
            lower = (row + 1) * cell_h - 12
            trip.append(src.crop((left, upper, right, lower)))
        for c in range(8):
            ax = axs[ridx, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor('#555')
            if c in (0, 6, 7):
                img = trip[0] if c == 0 else trip[2] if c == 6 else trip[1]
                ax.imshow(img)
            else:
                ax.set_facecolor('#ececec')
                ax.text(0.5, 0.5, 'Not executed\nin reduced\nsimulation', ha='center', va='center', fontsize=11, color='#555')
            ax.set_title(labels[c], fontsize=9, pad=5)
            ax.text(0.02, 0.98, panel_labels[ridx][c], ha='left', va='top', transform=ax.transAxes, fontsize=9, weight='bold')
    fig.text(0.06, 0.94, 'Fully dentate', fontsize=12, weight='bold')
    fig.text(0.06, 0.47, 'Partially edentulous', fontsize=12, weight='bold')
    fig.text(0.5, 0.02, 'Paper-aligned qualitative layout; unavailable baselines shown explicitly as not executed.', ha='center', fontsize=9, color='#55667a')
    return fig


def figure5b():
    src = Image.open(FINAL / 'figures/predictions.png').convert('RGB')
    w, h = src.size
    cell_w = w // 3
    cell_h = h // 3
    chosen_rows = [0, 2]
    fig, axs = plt.subplots(2, 3, figsize=(11, 7))
    labels = ['Input', 'SemiTNet', 'Ground Truth']
    for ridx, row in enumerate(chosen_rows):
        trip = []
        for c in range(3):
            left = c * cell_w + 12
            upper = row * cell_h + 12
            right = (c + 1) * cell_w - 12
            lower = (row + 1) * cell_h - 12
            trip.append(src.crop((left, upper, right, lower)))
        for c in range(3):
            ax = axs[ridx, c]
            ax.imshow(trip[0] if c == 0 else trip[2] if c == 1 else trip[1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(labels[c], fontsize=10)
            ax.text(0.02, 0.98, '(a)' if ridx == 0 and c == 0 else '(b)' if ridx == 0 and c == 1 else '(c)' if ridx == 0 and c == 2 else '(d)' if ridx == 1 and c == 0 else '(e)' if ridx == 1 and c == 1 else '(f)',
                    transform=ax.transAxes, va='top', ha='left', fontsize=9, weight='bold')
    fig.text(0.05, 0.94, 'Fully dentate', fontsize=12, weight='bold')
    fig.text(0.05, 0.47, 'Partially edentulous', fontsize=12, weight='bold')
    fig.text(0.5, 0.02, 'Measured qualitative result only.', ha='center', fontsize=9, color='#55667a')
    return fig


def write_comparison(measured):
    paper = read_json(PAPER / 'paper_targets.json')
    txt = []
    txt.append('# Figure comparison')
    txt.append('')
    txt.append('| Figure | Paper structure | Generated structure | Fidelity |')
    txt.append('|---|---|---|---|')
    rows = [
        ('Figure 1', 'published architecture', 'figure_01_semitnet_architecture.png + figure_01b_implemented_reduced_architecture.png', 'HIGH structural reference match after fix; implementation truth documented in Figure 1b'),
        ('Figure 2', 'distillation workflow', 'figure_02_teacher_student_workflow.png', 'HIGH workflow-layout match after fix'),
        ('Figure 3', 'dual-axis loss + precision', 'figure_03_training_loss_precision.png', 'HIGH visual-format match; measured series shorter than paper'),
        ('Figure 4', 'top + bottom radar geometry', 'figure_04_performance_radar.png + figure_04b_measured_reduced_simulation.png', 'HIGH paper-reference visual match; measured reduced results separated into Figure 4b'),
        ('Figure 5', '2-row qualitative comparison', 'figure_05_qualitative_outputs.png + figure_05b_measured_qualitative_only.png', 'PARTIAL; baselines unavailable and not fabricated; paper-aligned layout plus measured-only Figure 5b supplied'),
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

    figs = [figure1(), figure1b(), figure2(), figure3(), figure4(), figure4b(), figure5(), figure5b()]
    names = [
        'figure_01_semitnet_architecture',
        'figure_01b_implemented_reduced_architecture',
        'figure_02_teacher_student_workflow',
        'figure_03_training_loss_precision',
        'figure_04_performance_radar',
        'figure_04b_measured_reduced_simulation',
        'figure_05_qualitative_outputs',
        'figure_05b_measured_qualitative_only',
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
                'data_kind': 'published_reference_structure',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
            {
                'figure': 1.1,
                'path': 'outputs/paper_style/figures/figure_01b_implemented_reduced_architecture.png',
                'paper_counterpart': 'Figure 1b',
                'source_files': ['outputs/final/run_manifest.json'],
                'data_kind': 'implemented_structure',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
            {
                'figure': 2,
                'path': 'outputs/paper_style/figures/figure_02_teacher_student_workflow.png',
                'paper_counterpart': 'Figure 2',
                'source_files': ['outputs/final/run_manifest.json', 'outputs/final/metrics.json'],
                'data_kind': 'published_reference_structure_mapped_to_measured_workflow',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
            {
                'figure': 3,
                'path': 'outputs/paper_style/figures/figure_03_training_loss_precision.png',
                'paper_counterpart': 'Figure 3',
                'source_files': ['outputs/final/history.csv'],
                'data_kind': 'measured',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
            {
                'figure': 4,
                'path': 'outputs/paper_style/figures/figure_04_performance_radar.png',
                'paper_counterpart': 'Figure 4',
                'source_files': ['outputs/final/metrics.json', 'paper/reference/table2_overall.csv', 'paper/reference/table3_groups.csv', 'paper/reference/paper_targets.json'],
                'data_kind': 'published_reference',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
            {
                'figure': '4b',
                'path': 'outputs/paper_style/figures/figure_04b_measured_reduced_simulation.png',
                'paper_counterpart': 'Figure 4b',
                'source_files': ['outputs/final/metrics.json'],
                'data_kind': 'measured',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
            {
                'figure': 5,
                'path': 'outputs/paper_style/figures/figure_05_qualitative_outputs.png',
                'paper_counterpart': 'Figure 5',
                'source_files': ['outputs/final/figures/predictions.png'],
                'data_kind': 'measured_plus_explicit_unavailable_placeholders',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
            {
                'figure': 5.1,
                'path': 'outputs/paper_style/figures/figure_05b_measured_qualitative_only.png',
                'paper_counterpart': 'Figure 5b',
                'source_files': ['outputs/final/figures/predictions.png'],
                'data_kind': 'measured',
                'fabricated_values': False,
                'fabricated_predictions': False,
            },
        ],
    }
    (OUT / 'figure_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    shutil.copy2(FINAL / 'metrics.json', OUT / 'source_metrics.json')
    shutil.copy2(FINAL / 'history.csv', OUT / 'source_history.csv')
    shutil.copy2(FINAL / 'run_manifest.json', OUT / 'source_run_manifest.json')

    comparison = [
        '# Export readme',
        '',
        f"Generated from measured outputs in `outputs/final/` with {len(history)} history rows.",
        '',
        '## Output set',
        '- `outputs/paper_style/figures/figure_01_semitnet_architecture.png`',
        '- `outputs/paper_style/figures/figure_02_teacher_student_workflow.png`',
        '- `outputs/paper_style/figures/figure_03_training_loss_precision.png`',
        '- `outputs/paper_style/figures/figure_04_performance_radar.png`',
        '- `outputs/paper_style/figures/figure_04b_measured_reduced_simulation.png`',
        '- `outputs/paper_style/figures/figure_05_qualitative_outputs.png`',
        '- `outputs/paper_style/figures/figure_05b_measured_qualitative_only.png`',
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
    (OUT / 'EXPORT_README.md').write_text('\n'.join(comparison) + '\n')
    print(f'[ok] wrote figures to {FIG_DIR}')


if __name__ == '__main__':
    main()
