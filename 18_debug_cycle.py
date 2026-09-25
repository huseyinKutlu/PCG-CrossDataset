#!/usr/bin/env python3
"""
18_debug_cycle.py
HybridPCG4'un cokusunu teshis: gercek CirCor verisinde otokorelasyon
periyot tahmini mantikli mi (60-120 bpm) yoksa hep ust sinira mi yapisiyor?

Eger periyotlar saglikli degilse, faz kodlamasi gurultu ekliyor demektir.
"""
from __future__ import annotations
from pathlib import Path
from importlib import import_module
import numpy as np
import torch

ds_mod = import_module("05_dataset")
hy_mod = import_module("09_model_hybrid")

PROJECT_ROOT = Path(__file__).resolve().parent
cv_csv = PROJECT_ROOT / "splits" / "circor2022_5fold_cv.csv"

print("="*70)
print("CYCLE-AWARE faz teshisi (gercek CirCor verisi)")
print("="*70)

tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=PROJECT_ROOT, fold=0, split="train")
loader = ds_mod.make_eval_loader(tr, batch_size=128, num_workers=4)

all_bpm = []
n_batch = 0
for b in loader:
    raw = b["raw"].squeeze(1)   # (B, 10000)
    period = hy_mod.estimate_cycle_period(raw, fs=2000)   # (B,)
    bpm = (60 * 2000 / period).numpy()
    all_bpm.append(bpm)
    n_batch += 1
    if n_batch >= 10: break   # ~1280 segment yeter

all_bpm = np.concatenate(all_bpm)
print(f"\nIncelenen segment: {len(all_bpm)}")
print(f"Tahmini kalp hizi (bpm) dagilimi:")
print(f"  min={all_bpm.min():.1f} | max={all_bpm.max():.1f}")
print(f"  mean={all_bpm.mean():.1f} | median={np.median(all_bpm):.1f} | std={all_bpm.std():.1f}")
print(f"\n  Persentiller: 10%={np.percentile(all_bpm,10):.0f} "
      f"25%={np.percentile(all_bpm,25):.0f} 50%={np.percentile(all_bpm,50):.0f} "
      f"75%={np.percentile(all_bpm,75):.0f} 90%={np.percentile(all_bpm,90):.0f}")

# saglik kontrolu
# CirCor pediatrik: cocuk kalp hizi tipik 70-140 bpm
# ust sinira yapisma (200) = otokorelasyon basarisiz isareti
at_max = (all_bpm > 190).mean() * 100
at_min = (all_bpm < 45).mean() * 100
healthy = ((all_bpm >= 60) & (all_bpm <= 160)).mean() * 100
print(f"\n  Ust sinira (>190 bpm) yapisan: %{at_max:.1f}")
print(f"  Alt sinira (<45 bpm) yapisan: %{at_min:.1f}")
print(f"  Saglikli aralikta (60-160 bpm): %{healthy:.1f}")

print("\n" + "="*70)
if at_max > 30 or healthy < 40:
    print("[SORUN] Periyot tahmini GUVENILIR DEGIL — cogu segment sinira yapisiyor.")
    print("        Faz kodlamasi gurultu ekliyor -> HybridPCG4 cokusunun nedeni bu.")
    print("        Cozum: faz kodlamasini sadece guvenilir-periyot segmentlerde uygula,")
    print("        veya otokorelasyon yerine daha saglam periyot tespiti kullan.")
else:
    print("[OK] Periyot tahmini saglikli. Cokusun nedeni baska yerde aranmali.")
print("="*70)
