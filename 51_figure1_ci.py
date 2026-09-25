#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
51_figure1_ci.py
Hakem 1 (minör 1) — Sekil 1'i %95 guven araligi hata cubuklariyla yeniden uretir.

Veri kaynagi: reports/stats_revision.json (AUROC + stratified bootstrap %95 GA).
GPU GEREKMEZ, 2 saniyede biter.

Cikti:
  figures/Figure1_CI.png / .pdf   (300 dpi, MDPI icin vektor pdf de var)

Kullanim:
  cd ~/Desktop/pcg_project
  python 51_figure1_ci.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.size": 9, "font.family": "sans-serif",
                 "axes.spines.top": False, "axes.spines.right": False,
                 "axes.linewidth": 0.8})

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
STATS = ROOT / "reports" / "stats_revision.json"

ORDER = ["cnn1d", "cnn2d", "resnet1d", "cnn_bilstm", "hybrid_mlp",
         "hybrid_kan", "efficientnet", "maxvit", "mambaconformer"]
PRETTY = {"cnn1d": "1D-CNN", "cnn2d": "2D-CNN", "resnet1d": "ResNet1D",
          "cnn_bilstm": "CNN-BiLSTM", "hybrid_mlp": "Hybrid-MLP",
          "hybrid_kan": "Hybrid-KAN", "efficientnet": "EfficientNet-B0",
          "maxvit": "MaxViT", "mambaconformer": "PCG-MambaConformer"}

C_INT = "#8C8C8C"; C_CINC = "#4878CF"; C_YAS = "#6ACC65"

d = json.loads(STATS.read_text(encoding="utf-8"))["per_model"]
models = [m for m in ORDER if m in d]
names = [PRETTY[m] for m in models]


def vals(split):
    pt = np.array([d[m][split]["auroc"] for m in models])
    lo = np.array([d[m][split]["auroc_ci"][0] for m in models])
    hi = np.array([d[m][split]["auroc_ci"][1] for m in models])
    # errorbar asimetrik yari-genislikler
    return pt, np.vstack([pt - lo, hi - pt])


x = np.arange(len(models)); w = 0.27
fig, ax = plt.subplots(figsize=(9.5, 4.2))

for off, split, col, lab in [
        (-w, "internal", C_INT, "Internal (CirCor, patient level)"),
        (0.0, "cinc", C_CINC, "External: CinC 2016 (heterogeneous)"),
        (w, "yaseen", C_YAS, "External: Yaseen 2018 (homogeneous)")]:
    pt, err = vals(split)
    ax.bar(x + off, pt, w, color=col, label=lab, zorder=2)
    ax.errorbar(x + off, pt, yerr=err, fmt="none", ecolor="black",
                elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=3)

ax.axhline(0.5, ls="--", c="k", lw=0.7, alpha=0.6, zorder=1)
ax.text(len(models) - 0.45, 0.515, "chance", fontsize=7, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylabel("AUROC")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.16))
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"Figure1_CI.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("[OK] figures/Figure1_CI.png + .pdf")
print("Hata cubuklari: stratified bootstrap %95 GA (2000 resample).")
