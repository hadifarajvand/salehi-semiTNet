#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import io
import json
import time
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/processed/quick_teeth"
MANIFEST_PATH = DATA_ROOT / "split_manifest.json"
OUT = ROOT / "outputs/final"
FIG = OUT / "figures"
H, W = 128, 256
BATCH = 4
TEACHER_EPOCHS = 8
STUDENT_PASSES = 2
PSEUDO_THRESHOLD = 0.35
UNSUP_WEIGHT = 0.05
MIN_CONFIDENT_PSEUDO_PIXELS = 16


def annotation_mask(ann_path: Path) -> np.ndarray:
    ann = json.loads(ann_path.read_text())
    height = int(ann["size"]["height"])
    width = int(ann["size"]["width"])
    target = np.zeros((height, width), dtype=np.uint8)
    for obj in ann.get("objects", []):
        try:
            cls = int(str(obj.get("classTitle", "0")).strip())
        except ValueError:
            continue
        if cls < 1 or cls > 32:
            continue
        geometry = obj.get("geometryType") or obj.get("shape") or ""
        if "bitmap" in obj or geometry == "bitmap":
            bm = obj.get("bitmap", {})
            if not bm.get("data"):
                continue
            raw = base64.b64decode(bm["data"])
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                pass
            local = np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=bool)
            origin = bm.get("origin", [0, 0])
            x, y = int(origin[0]), int(origin[1])
            y2, x2 = min(height, y + local.shape[0]), min(width, x + local.shape[1])
            if y2 <= y or x2 <= x:
                continue
            crop = local[: y2 - y, : x2 - x]
            region = target[y:y2, x:x2]
            region[crop] = cls
        elif geometry == "polygon" or "points" in obj:
            exterior = obj.get("points", {}).get("exterior", [])
            if len(exterior) >= 3:
                canvas = Image.new("L", (width, height), 0)
                ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in exterior], fill=1)
                target[np.asarray(canvas, dtype=bool)] = cls
    return target


class PreparedTeeth(Dataset):
    def __init__(self, split: str, with_labels: bool):
        self.img_dir = DATA_ROOT / split / "img"
        self.ann_dir = DATA_ROOT / split / "ann"
        self.with_labels = with_labels
        self.images = sorted(p for p in self.img_dir.iterdir() if p.is_file())
        if not self.images:
            raise RuntimeError(f"Prepared split is empty: {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("L").resize((W, H), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).copy()[None] / 255.0)
        if not self.with_labels:
            return x
        ann_path = self.ann_dir / f"{img_path.name}.json"
        if not ann_path.exists():
            raise RuntimeError(f"Missing prepared annotation: {ann_path}")
        mask = annotation_mask(ann_path)
        mask = Image.fromarray(mask).resize((W, H), Image.Resampling.NEAREST)
        y = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return x, y


class QuickSemiTransformer(nn.Module):
    def __init__(self, dim=24, classes=33):
        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=8, stride=8)
        token_count = (H // 8) * (W // 8)
        self.pos = nn.Parameter(torch.zeros(1, token_count, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, dropout=0.0, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 24, kernel_size=8, stride=8),
            nn.GELU(),
            nn.Conv2d(24, classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.patch(x)
        b, c, h, w = z.shape
        tokens = z.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self.pos[:, : tokens.shape[1]])
        z = tokens.transpose(1, 2).reshape(b, c, h, w)
        return self.decoder(z)


def batch_counts(pred, target):
    pfg, tfg = pred > 0, target > 0
    seg_tp = (pfg & tfg).sum().item()
    seg_fp = (pfg & ~tfg).sum().item()
    seg_fn = (~pfg & tfg).sum().item()
    id_tp = ((pred == target) & tfg).sum().item()
    id_fp = ((pred != target) & pfg).sum().item()
    id_fn = ((pred != target) & tfg).sum().item()
    return seg_tp, seg_fp, seg_fn, id_tp, id_fp, id_fn


def evaluate(model, loader, device):
    totals = np.zeros(6, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            totals += np.asarray(batch_counts(pred, y), dtype=np.float64)
    stp, sfp, sfn, itp, ifp, ifn = totals
    eps = 1e-8
    iou = stp / (stp + sfp + sfn + eps)
    dice = 2 * stp / (2 * stp + sfp + sfn + eps)
    precision = itp / (itp + ifp + eps)
    recall = itp / (itp + ifn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"iou": iou * 100, "dice": dice * 100, "precision": precision * 100, "recall": recall * 100, "f1": f1 * 100}


def supervised_loss(logits, target, weights):
    ce = F.cross_entropy(logits, target, weight=weights)
    probs = torch.softmax(logits, dim=1)
    fg_prob = 1.0 - probs[:, 0]
    fg_true = (target > 0).float()
    intersection = (fg_prob * fg_true).sum(dim=(1, 2))
    denom = fg_prob.sum(dim=(1, 2)) + fg_true.sum(dim=(1, 2))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
    return ce + 0.5 * dice_loss


def train_epoch(model, loader, opt, weights, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = supervised_loss(model(x), y, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))


def model_health(model, loader, weights, device):
    model.eval()
    losses = []
    fg_pred = 0
    total_px = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(supervised_loss(logits, y, weights).item())
            pred = logits.argmax(1)
            fg_pred += (pred > 0).sum().item()
            total_px += pred.numel()
    return {
        "supervised_loss": float(np.mean(losses)),
        "foreground_fraction": fg_pred / max(1, total_px),
    }


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit("Prepared dataset manifest is missing. Run: python project.py download")
    manifest = json.loads(MANIFEST_PATH.read_text())
    split = manifest.get("split", {})
    if split != {"train_labeled": 60, "pseudo_unlabeled": 20, "test": 16}:
        raise SystemExit(f"Unexpected prepared split: {split}. Rerun: python project.py download")
    if manifest.get("verified_total_images") != 598 or manifest.get("verified_classes") != [str(i) for i in range(1, 33)]:
        raise SystemExit("Prepared dataset manifest failed identity verification. Rerun: python project.py download")

    seed = int(manifest["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(6, torch.get_num_threads()))

    labeled = DataLoader(PreparedTeeth("train_labeled", True), batch_size=BATCH, shuffle=True, num_workers=0)
    labeled_eval = DataLoader(PreparedTeeth("train_labeled", True), batch_size=BATCH, shuffle=False, num_workers=0)
    pseudo = DataLoader(PreparedTeeth("pseudo_unlabeled", False), batch_size=BATCH, shuffle=True, num_workers=0)
    test = DataLoader(PreparedTeeth("test", True), batch_size=BATCH, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.ones(33, device=device)
    weights[0] = 0.02
    history = []
    start = time.time()

    teacher = QuickSemiTransformer().to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=6e-4, weight_decay=1e-4)
    for epoch in range(TEACHER_EPOCHS):
        loss = train_epoch(teacher, labeled, opt, weights, device)
        vals = evaluate(teacher, test, device)
        history.append({"phase": "teacher", "epoch": epoch + 1, "loss": loss, **vals})

    teacher_health = model_health(teacher, labeled_eval, weights, device)
    student = QuickSemiTransformer().to(device)
    student.load_state_dict(teacher.state_dict())
    opt_s = torch.optim.AdamW(student.parameters(), lr=1.5e-4, weight_decay=1e-4)
    total = 0.0
    student_steps = 0
    pseudo_pixels_used = 0

    for pass_idx in range(STUDENT_PASSES):
        student.train()
        piter = iter(pseudo)
        for x, y in labeled:
            try:
                u = next(piter)
            except StopIteration:
                piter = iter(pseudo)
                u = next(piter)
            x, y, u = x.to(device), y.to(device), u.to(device)
            with torch.no_grad():
                teacher_probs = torch.softmax(teacher(u), 1)
                conf, pseudo_y = teacher_probs.max(1)
                confident = (conf >= PSEUDO_THRESHOLD) & (pseudo_y > 0)
                pseudo_target = pseudo_y.clone()
                pseudo_target[~confident] = -100
                confident_count = int(confident.sum().item())

            opt_s.zero_grad()
            logits_x = student(x)
            sup = supervised_loss(logits_x, y, weights)
            logits_u = student(u)
            if confident_count >= MIN_CONFIDENT_PSEUDO_PIXELS:
                uns = F.cross_entropy(logits_u, pseudo_target, weight=weights, ignore_index=-100)
                pseudo_pixels_used += confident_count
            else:
                uns = logits_u.sum() * 0.0
            loss = sup + UNSUP_WEIGHT * uns
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt_s.step()
            total += loss.item()
            student_steps += 1
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.mul_(0.995).add_(sp, alpha=0.005)

        vals = evaluate(student, test, device)
        history.append({"phase": "student", "epoch": pass_idx + 1, "loss": total / max(1, student_steps), **vals})

    student_health = model_health(student, labeled_eval, weights, device)
    collapsed = (
        student_health["foreground_fraction"] < 0.001
        or student_health["supervised_loss"] > teacher_health["supervised_loss"] * 1.20
    )
    selected = teacher if collapsed else student
    selected_name = "teacher_fallback" if collapsed else "student"
    final = evaluate(selected, test, device)
    if all(final[k] <= 0.0 for k in ("iou", "dice", "precision", "recall", "f1")):
        raise SystemExit("Final model produced all-zero metrics; refusing to create a PASS deliverable.")

    final.update({
        "source_kind": "measured_full_pipeline_reduced_subset",
        "dataset": manifest["dataset"],
        "dataset_license": manifest["license"],
        "verified_dataset_images": manifest["verified_total_images"],
        "classes": 32,
        "used_labeled_train": split["train_labeled"],
        "used_label_hidden_pseudo": split["pseudo_unlabeled"],
        "used_test": split["test"],
        "image_size": [H, W],
        "runtime_seconds": time.time() - start,
        "device": str(device),
        "selected_checkpoint": selected_name,
        "student_collapse_guard_triggered": collapsed,
        "teacher_health": teacher_health,
        "student_health": student_health,
        "confident_pseudo_pixels_used": pseudo_pixels_used,
        "note": "Measured end-to-end teacher/student simulation using the verified deterministic prepared subset. Low-confidence pseudo labels are ignored rather than treated as background."
    })

    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(final, indent=2))
    with (OUT / "history.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=history[0].keys())
        wr.writeheader(); wr.writerows(history)
    (OUT / "run_manifest.json").write_text(json.dumps({
        "dataset_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "dataset": manifest["dataset"],
        "verified_total_images": manifest["verified_total_images"],
        "seed": seed,
        "model": "QuickSemiTransformer",
        "workflow": ["teacher supervised warm-up", "pseudo-label generation", "student training", "EMA teacher update", "collapse guard", "held-out evaluation"],
        "split": split,
        "selected_checkpoint": selected_name,
        "pseudo_threshold": PSEUDO_THRESHOLD,
        "unsupervised_weight": UNSUP_WEIGHT,
    }, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(history) + 1), [r["loss"] for r in history], marker="o")
    ax.set_xlabel("Stage epoch"); ax.set_ylabel("Loss"); ax.set_title("Teacher/student simulation training")
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["IoU", "Dice", "Precision", "Recall", "F1"]
    values = [final[k.lower()] for k in names]
    bars = ax.bar(names, values); ax.set_ylim(0, 100); ax.set_ylabel("Percent"); ax.set_title(f"Measured simulation metrics ({selected_name})")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "metrics.png", dpi=180); plt.close(fig)

    x, y = next(iter(test)); selected.eval()
    with torch.no_grad(): pred = selected(x.to(device)).argmax(1).cpu()
    n = min(3, len(x)); fig, axs = plt.subplots(n, 3, figsize=(11, 3 * n))
    if n == 1: axs = np.array([axs])
    for i in range(n):
        axs[i, 0].imshow(x[i, 0], cmap="gray"); axs[i, 0].set_title("Panoramic X-ray")
        axs[i, 1].imshow(y[i], vmin=0, vmax=32, cmap="nipy_spectral"); axs[i, 1].set_title("Ground truth (32 classes)")
        axs[i, 2].imshow(pred[i], vmin=0, vmax=32, cmap="nipy_spectral"); axs[i, 2].set_title(f"Prediction ({selected_name})")
        for axis in axs[i]: axis.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "predictions.png", dpi=180); plt.close(fig)

    report = f"""# SemiTNet Pipeline Simulation Results

Dataset: {manifest['dataset']} ({manifest['license']})

The dataset download was verified before simulation: {manifest['verified_total_images']} panoramic images and tooth classes 1–32. A deterministic prepared subset was used to execute the complete teacher/student simulation workflow end-to-end.

- Labeled training images: {split['train_labeled']}
- Label-hidden pseudo-label images: {split['pseudo_unlabeled']}
- Held-out test images: {split['test']}
- Selected checkpoint: {selected_name}
- Collapse guard triggered: {collapsed}
- Confident pseudo-label pixels used: {pseudo_pixels_used}
- IoU: {final['iou']:.2f}%
- Dice: {final['dice']:.2f}%
- Precision: {final['precision']:.2f}%
- Recall: {final['recall']:.2f}%
- F1: {final['f1']:.2f}%
- Runtime: {final['runtime_seconds']:.1f} seconds
"""
    (OUT / "RESULTS.md").write_text(report)
    print(json.dumps(final, indent=2))
    print("[ok] measured outputs:", OUT)


if __name__ == "__main__":
    main()
