#!/usr/bin/env python3
"""
52_revision_figures.py
JCM-4481322 minor revizyon: makale Sekil 2, 3, 4, 5, 6, 7'yi yeniden uretir.

Degisiklikler (43_updated_figures.py ve 31_figures.py'ye gore):
  - Gorsel ici yorum iceren ust basliklar (suptitle) kaldirildi; baslik lejantta.
  - Model adlari makaleyle ayni: PCG-MambaConformer, EfficientNet-B0, Tablo 4 adlari.
  - "clinically usable", "FUSION SYNERGY", "each degenerates", "CONDITIONAL",
    "false confidence" gibi ifadeler kaldirildi.
  - Sekil 2: katlar 1-5; Sens 0.718 / Spec 0.995; panel C GERCEK tahminlerden.
  - Sekil 3: CNN-BiLSTM naif degerleri Tablo 1 ile ayni (0.727 [0.708-0.746],
    0.806 [0.779-0.833]).
  - Sekil 7: Tablo 4 ile ayni uc basamak; tam model 0.718.

ONEMLI: Eski 43_updated_figures.py, Sekil 2 panel C'yi np.random.normal ile
SIMULE ediyordu. Bu script panel C'yi yalnizca kayitli gercek Yaseen
tahminlerinden cizer. Dosya bulunamazsa Sekil 2 URETILMEZ (sahte veriye
dusmez).

Kullanim:
  python 52_revision_figures.py --yaseen-pred YOL/dosya.csv
  (CSV: etiket ve olasilik sutunlari; .npz/.npy de desteklenir)
Cikti: figures_revision/FigureN_*.png (300 dpi) + .pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({"font.size": 10, "font.family": "sans-serif",
                 "axes.spines.top": False, "axes.spines.right": False,
                 "axes.linewidth": 0.8})
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "figures_revision"
FIG_DIR.mkdir(exist_ok=True)

C_BLUE = "#4878CF"; C_RED = "#D65F5F"; C_GREEN = "#6ACC65"
C_GRAY = "#999999"; C_LGRAY = "#BBBBBB"

# ---- Ensemble-Balanced iki modelin adini 41_ensemble.py'den kontrol edip yazin ----
BALANCED_LABEL = "Ensemble-Balanced"


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [OK] {name}.png + .pdf")


# ----------------------------------------------------------------------------
# Gercek Yaseen tahminlerini yukleme
# ----------------------------------------------------------------------------
LABEL_KEYS = ["yas_y", "y_true", "label", "labels", "y", "target", "true", "binary_label"]
PROB_KEYS = ["yas_p", "y_prob", "prob", "probs", "p", "prob_abnormal", "p_abnormal",
             "score", "pred_prob", "ensemble_prob", "mean_prob"]


def _pick(names, keys):
    low = {n.lower(): n for n in names}
    for k in keys:
        if k in low:
            return low[k]
    return None


def load_predictions(path: Path):
    suf = path.suffix.lower()
    if suf == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        lk, pk = _pick(df.columns, LABEL_KEYS), _pick(df.columns, PROB_KEYS)
        if lk is None or pk is None:
            raise SystemExit(f"Sutunlar taninmadi: {list(df.columns)}\n"
                             "LABEL_KEYS / PROB_KEYS listesine sutun adlarini ekleyin.")
        y, p = df[lk].to_numpy(), df[pk].to_numpy(dtype=float)
    elif suf == ".npz":
        d = np.load(path)
        lk, pk = _pick(d.files, LABEL_KEYS), _pick(d.files, PROB_KEYS)
        if lk is None or pk is None:
            raise SystemExit(f"Anahtarlar taninmadi: {d.files}")
        y, p = d[lk], d[pk].astype(float)
    elif suf == ".npy":
        a = np.load(path)
        if a.ndim != 2 or a.shape[1] != 2:
            raise SystemExit(".npy icin (N,2) dizi bekleniyor: [etiket, olasilik]")
        y, p = a[:, 0], a[:, 1].astype(float)
    else:
        raise SystemExit(f"Desteklenmeyen dosya turu: {suf}")
    if p.ndim == 2:            # (N,2) softmax ise abnormal sutunu
        p = p[:, 1]
    y = np.asarray(y)
    if y.dtype.kind in "OUS":  # metin etiket
        y = np.array([0 if str(v).strip().lower() in ("normal", "0", "absent") else 1 for v in y])
    return y.astype(int), p


# ----------------------------------------------------------------------------
# Figure 2 (eski F13): PCG-MambaConformer, Yaseen
# ----------------------------------------------------------------------------
def figure2(pred_path: Path | None):
    if pred_path is None or not pred_path.exists():
        print("  [ATLANDI] Figure 2: --yaseen-pred ile gercek tahmin dosyasi verilmedi.")
        return
    y, p = load_predictions(pred_path)
    if len(y) != 1000:
        print(f"  [UYARI] {len(y)} kayit bulundu; Yaseen icin 1000 bekleniyordu.")
    pred = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p)
    print(f"  Kontrol -> AUROC={auc:.4f}  CM={cm.tolist()}  Sens={sens:.4f}  Spec={spec:.4f}")
    print(f"  Kontrol -> ortalama olasilik: normal={p[y==0].mean():.3f}  abnormal={p[y==1].mean():.3f}"
          "  (makale §3.2: 0.23 ve 0.64)")
    if cm.tolist() != [[199, 1], [226, 574]]:
        print("  [UYARI] Karisiklik matrisi makaledeki 199/1/226/574 ile AYNI DEGIL. "
              "Dosyanin 5-fold ensemble tahmini oldugunu kontrol edin.")

    folds = [0.9912, 0.9053, 0.8252, 0.9179, 0.9339]   # 5 katin Yaseen AUROC'u
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # (A) per-fold
    axes[0].bar(np.arange(1, 6), folds, color=C_RED, alpha=0.85)
    axes[0].axhline(0.985, ls="--", c="k", lw=1, label="Five-fold ensemble (0.985)")
    axes[0].set_xticks(np.arange(1, 6)); axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Yaseen AUROC"); axes[0].set_ylim(0.7, 1.0)
    axes[0].set_title("(A) Per-fold AUROC"); axes[0].legend(fontsize=8, loc="lower right")
    # (B) confusion
    axes[1].imshow(cm, cmap="Reds")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Normal", "Abnormal"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Normal", "Abnormal"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    axes[1].set_title(f"(B) Confusion matrix (Sens {sens:.3f}, Spec {spec:.3f})")
    # (C) GERCEK skor dagilimi
    bins = np.linspace(0, 1, 26)
    axes[2].hist(p[y == 0], bins=bins, alpha=0.6, color=C_BLUE, label="Normal", density=True)
    axes[2].hist(p[y == 1], bins=bins, alpha=0.6, color=C_RED, label="Abnormal", density=True)
    axes[2].axvline(0.5, ls="--", c="k", lw=0.8)
    axes[2].set_xlabel("Predicted abnormal probability"); axes[2].set_ylabel("Density")
    axes[2].set_title("(C) Predicted-probability distributions"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "Figure2_PCG-MambaConformer_Yaseen")


# ----------------------------------------------------------------------------
# Figure 3 (eski F2): DANN
# ----------------------------------------------------------------------------
def figure3():
    groups = ["CinC 2016\n(heterogeneous)", "Yaseen 2018\n(homogeneous)"]
    naive = [0.727, 0.806]; naive_ci = [(0.708, 0.746), (0.779, 0.833)]   # Tablo 1
    dann = [0.675, 0.916];  dann_ci = [(0.653, 0.698), (0.899, 0.932)]
    x = np.arange(2); w = 0.35
    err = lambda v, ci: [[v[i] - ci[i][0] for i in range(2)], [ci[i][1] - v[i] for i in range(2)]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, naive, w, yerr=err(naive, naive_ci), label="Without adaptation", color=C_BLUE, capsize=4)
    ax.bar(x + w/2, dann, w, yerr=err(dann, dann_ci), label="DANN", color=C_RED, capsize=4)
    ax.text(0, 0.775, "p < 0.001", ha="center", fontsize=8)
    ax.text(1, 0.950, "p < 0.001", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("External AUROC (CNN-BiLSTM)"); ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "Figure3_DANN")


# ----------------------------------------------------------------------------
# Figure 4 (eski F3): hastalik-spesifik transfer
# ----------------------------------------------------------------------------
def figure4():
    dis = ["Aortic\nstenosis", "Mitral\nregurgitation", "Mitral\nstenosis", "Mitral valve\nprolapse"]
    sens = [0.925, 0.735, 0.170, 0.055]
    auroc = [0.982, 0.945, 0.780, 0.518]
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w/2, sens, w, label="Sensitivity", color=C_RED)
    ax.bar(x + w/2, auroc, w, label="Class-versus-normal AUROC", color=C_BLUE)
    ax.axhline(0.5, ls="--", c="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(dis, fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "Figure4_disease_specific")


# ----------------------------------------------------------------------------
# Figure 5 (eski F4): belirsizlik
# ----------------------------------------------------------------------------
def figure5():
    methods = ["Deep\nensemble", "MC\ndropout", "Evidential", "KAN\nensemble"]
    ece = [0.133, 0.114, 0.100, 0.106]; ece_e = [0.012, 0.015, 0.044, 0.014]
    sel = [0.861, 0.814, 0.910, 0.833]; sel_e = [0.026, 0.023, 0.021, 0.026]
    ood = [0.099, 0.108, -0.022, 0.085]; ood_e = [0.030, 0.017, 0.038, 0.028]
    cols = [C_BLUE, C_GRAY, C_RED, C_GREEN]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    axes[0].bar(x, ece, yerr=ece_e, color=cols, capsize=3)
    axes[0].set_title("(A) Expected calibration error"); axes[0].set_ylabel("ECE (lower is better)")
    axes[1].bar(x, sel, yerr=sel_e, color=cols, capsize=3)
    axes[1].set_title("(B) Selective prediction (30% abstention)")
    axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0.7, 0.95)
    axes[2].bar(x, ood, yerr=ood_e, color=cols, capsize=3)
    axes[2].axhline(0, c="k", lw=0.8)
    axes[2].set_title("(C) OOD − in-distribution uncertainty")
    axes[2].set_ylabel("Difference in mean uncertainty")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    fig.tight_layout()
    save(fig, "Figure5_uncertainty")


# ----------------------------------------------------------------------------
# Figure 6 (eski F15): ensemble
# ----------------------------------------------------------------------------
def figure6():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    yn = ["PCG-MambaConformer\n(single)", BALANCED_LABEL, "Ensemble-ALL"]
    yv = [0.985, 0.969, 0.959]
    b = axes[0].bar(yn, yv, color=[C_RED, C_LGRAY, C_LGRAY])
    axes[0].set_ylabel("Yaseen AUROC"); axes[0].set_ylim(0.9, 1.0)
    axes[0].set_title("(A) Yaseen 2018 (homogeneous)")
    for bb, v in zip(b, yv):
        axes[0].text(bb.get_x() + bb.get_width()/2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    cn = ["PCG-MambaConformer\n(single)", BALANCED_LABEL, "EfficientNet-B0\n(best single)"]
    cv = [0.601, 0.697, 0.738]
    b2 = axes[1].bar(cn, cv, color=[C_LGRAY, C_LGRAY, C_BLUE])
    axes[1].axhline(0.5, ls="--", c="k", lw=0.6, alpha=0.5)
    axes[1].set_ylabel("CinC AUROC"); axes[1].set_ylim(0.5, 0.8)
    axes[1].set_title("(B) CinC 2016 (heterogeneous)")
    for bb, v in zip(b2, cv):
        axes[1].text(bb.get_x() + bb.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    save(fig, "Figure6_ensemble")


# ----------------------------------------------------------------------------
# Figure 7 (eski F14): ablasyon — revise/ablation_all_eval.log ile ayni
# ----------------------------------------------------------------------------
def figure7():
    single = [("Only\nInceptionTime", 0.966, 0.830), ("Only\nConformer", 0.878, 1.000),
              ("Only\nCWT", 0.768, 1.000), ("Only\nstate-space", 0.969, 0.000)]
    loo = [("Without\nInceptionTime", 0.977), ("Without\nConformer", 0.928),
           ("Without\nCWT", 0.849), ("Without\nstate-space", 0.922)]
    full = ("Full\n(four-branch)", 0.985, 0.718)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [s[0] for s in single] + [full[0]]
    aucs = [s[1] for s in single] + [full[1]]
    sens = [s[2] for s in single] + [full[2]]
    x = np.arange(len(names)); w = 0.38
    axes[0].bar(x - w/2, aucs, w, label="AUROC", color=C_BLUE)
    axes[0].bar(x + w/2, sens, w, label="Sensitivity", color=C_RED)
    axes[0].axhline(0.5, ls="--", c="k", lw=0.6, alpha=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel("Score"); axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(A) Single-branch configurations and full model")
    axes[0].text(3 + w/2, 0.015, "0.000", ha="center", fontsize=7, color=C_RED)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ln = [l[0] for l in loo]; la = [l[1] for l in loo]
    x2 = np.arange(len(ln))
    bars = axes[1].bar(x2, la, color=C_GREEN, alpha=0.85)
    bars[ln.index("Without\nCWT")].set_color(C_RED)
    axes[1].axhline(full[1], ls="--", c=C_RED, lw=1.5, label=f"Full model ({full[1]:.3f})")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(ln, fontsize=8)
    axes[1].set_ylabel("Yaseen AUROC"); axes[1].set_ylim(0.7, 1.0)
    axes[1].set_title("(B) Leave-one-out configurations")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    save(fig, "Figure7_ablation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaseen-pred", type=Path, default=None,
                    help="PCG-MambaConformer 5-fold ensemble Yaseen tahminleri (csv/npz/npy)")
    args = ap.parse_args()
    print("=" * 60); print("REVIZYON SEKILLERI"); print("=" * 60)
    figure2(args.yaseen_pred); figure3(); figure4(); figure5(); figure6(); figure7()
    print(f"\nCikti: {FIG_DIR}")


if __name__ == "__main__":
    main()
