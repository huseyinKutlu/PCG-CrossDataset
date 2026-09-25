#!/usr/bin/env python3
"""
07_train_baseline.py
PCG Q1 Manuscript — Adim A: Baseline modeller + egitim/degerlendirme iskeleti.

AMAC:
  - Boru hattinin uctan uca dondugunu GERCEK sinyalle dogrulamak.
  - Iki baseline: 1D-CNN (raw) ve 2D-CNN (mel).
  - Segment-duzeyi egitim -> HASTA-duzeyi degerlendirme (resmi protokol).
  - Uclu metrik: W.acc (resmi) + macro-F1 + UAR.
  - Tek fold (varsayilan fold0), erken durdurma, epoch tavani.

Sonraki adimda (Adim B) ana hibrit model AYNI iskelete takilacak.

Kullanim:
  conda activate deep_metalearning
  python 07_train_baseline.py --model cnn1d --fold 0 --epochs 20
  python 07_train_baseline.py --model cnn2d --fold 0 --epochs 20

Cikti:
  reports/baseline_<model>_fold<k>.txt   (egitim logu + final metrikler)
  checkpoints/baseline_<model>_fold<k>.pth
"""
from __future__ import annotations
import sys
import argparse
import time
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 05_dataset ve 06_metrics rakamla basladigi icin import_module ile yukluyoruz
ds_mod = import_module("05_dataset")
me_mod = import_module("06_metrics")

PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_DIR  = PROJECT_ROOT / "splits"
REPORT_DIR = PROJECT_ROOT / "reports"
CKPT_DIR   = PROJECT_ROOT / "checkpoints"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3


# =============================================================================
# Baseline modeller
# =============================================================================
class CNN1D(nn.Module):
    """Ham PCG sinyali (1, 10000) icin basit ama saglam 1D-CNN baseline."""
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        def block(ci, co, k=7, s=1, p=3):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, s, p), nn.BatchNorm1d(co),
                nn.ReLU(inplace=True), nn.MaxPool1d(4))
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64),
            block(64, 128), block(128, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):       # x: (B,1,10000)
        return self.head(self.features(x))


class CNN2D(nn.Module):
    """Log-mel (1,64,201) icin basit 2D-CNN baseline (ResNet-vari degil, sade)."""
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):       # x: (B,1,64,201)
        return self.head(self.features(x))


MODELS = {"cnn1d": (CNN1D, "raw"), "cnn2d": (CNN2D, "mel")}


# =============================================================================
# Egitim / Degerlendirme
# =============================================================================
def run_epoch(model, loader, input_key, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch[input_key].to(DEVICE, non_blocking=True)
            y = batch["label"].to(DEVICE, non_blocking=True)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y); n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_patient_level(model, loader, input_key, split_csv,
                           agg_method="mean"):
    """Segment olasiliklari -> hasta agregasyonu -> resmi metrikler."""
    model.eval()
    all_probs, all_pids = [], []
    for batch in loader:
        x = batch[input_key].to(DEVICE, non_blocking=True)
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_pids.extend(batch["patient_id"])
    seg_probs = np.concatenate(all_probs, axis=0)

    pat_probs, pids = ds_mod.aggregate_to_patient(seg_probs, all_pids, agg_method)
    pat_pred = pat_probs.argmax(axis=1)

    truth = ds_mod.patient_true_labels(split_csv)
    y_true = np.array([truth[p] for p in pids])

    m = me_mod.all_metrics(y_true, pat_pred)
    cm = me_mod.confusion_matrix(y_true, pat_pred)
    return m, cm, (y_true, pat_pred, pat_probs, pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="cnn2d")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5, help="erken durdurma")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"

    log_lines = []
    def log(m=""):
        print(m); log_lines.append(str(m))

    log("="*70)
    log(f"Baseline egitimi: {args.model} | fold {args.fold} | device={DEVICE}")
    log("="*70)

    ModelCls, input_key = MODELS[args.model]

    # Datasetler
    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root,
                                     fold=args.fold, split="train")
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root,
                                     fold=args.fold, split="val")
    log(f"Train: {len(tr)} segment / {tr.records['patient_id'].nunique()} hasta")
    log(f"Val  : {len(va)} segment / {va.records['patient_id'].nunique()} hasta")

    train_loader = ds_mod.make_train_loader(
        tr, batch_size=args.batch_size, num_workers=args.num_workers, balanced=True)
    val_loader = ds_mod.make_eval_loader(
        va, batch_size=128, num_workers=args.num_workers)

    # Model + loss (sinif-agirlikli) + opt
    model = ModelCls().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model parametre: {n_params:,}")
    class_w = tr.class_weights().to(DEVICE)
    log(f"Sinif agirliklari (P/U/A): {class_w.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_wacc, best_epoch, bad = -1.0, -1, 0
    ckpt = CKPT_DIR / f"baseline_{args.model}_fold{args.fold}.pth"

    log("\nEpoch | train_loss | val W.Acc | macro-F1 |  UAR  | (P/U/A recall)")
    log("-"*70)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, input_key, criterion, optimizer)
        scheduler.step()
        m, cm, _ = evaluate_patient_level(model, val_loader, input_key, cv_csv)
        dt = time.time() - t0
        log(f"{ep:5d} | {tr_loss:10.4f} | {m['weighted_accuracy']:9.4f} | "
            f"{m['macro_f1']:8.4f} | {m['uar']:.4f} | "
            f"{m['recall_present']:.2f}/{m['recall_unknown']:.2f}/"
            f"{m['recall_absent']:.2f}  ({dt:.0f}s)")

        if m["weighted_accuracy"] > best_wacc:
            best_wacc = m["weighted_accuracy"]; best_epoch = ep; bad = 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "metrics": m, "args": vars(args)}, ckpt)
        else:
            bad += 1
            if bad >= args.patience:
                log(f"\nErken durdurma: {args.patience} epoch iyilesme yok.")
                break

    # En iyi modelle final val raporu
    log("\n" + "="*70)
    log(f"En iyi epoch: {best_epoch} | val W.Acc={best_wacc:.4f}")
    ck = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    m, cm, _ = evaluate_patient_level(model, val_loader, input_key, cv_csv)
    log("\nFinal (val, en iyi model):")
    log("   " + me_mod.format_metrics(m))
    log("\nConfusion matrix [gercek x tahmin] (P/U/A):")
    log("        pred_P  pred_U  pred_A")
    for i, cn in enumerate(me_mod.CLASSES):
        log(f"   {cn:>7s} {cm[i,0]:6d} {cm[i,1]:7d} {cm[i,2]:7d}")

    # cogunluk-sinifi tuzagi uyarisi
    if m["recall_present"] < 0.1 and m["recall_unknown"] < 0.1:
        log("\n[UYARI] Model neredeyse hep Absent tahmin ediyor olabilir — "
            "sampler/agirlik/lr kontrol edilmeli.")
    else:
        log("\n[OK] Model azinlik siniflarini da ogreniyor (cogunluk tuzagi yok).")

    rep = REPORT_DIR / f"baseline_{args.model}_fold{args.fold}.txt"
    rep.write_text("\n".join(log_lines), encoding="utf-8")
    log(f"\nRapor: {rep}")
    log(f"Checkpoint: {ckpt}")
    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
