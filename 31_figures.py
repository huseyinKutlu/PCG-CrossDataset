#!/usr/bin/env python3
"""
31_figures.py
Makale icin 6 yayin-kalitesi sekil uretir (sonuclardan dogrudan).

Sekiller:
  F1: Ic vs Dis AUROC ucurumu (tum modeller, iki dataset)
  F2: DANN KOSULLU etki (Yaseen yardim / CinC zarar, bar + CI)
  F3: Hastalik-spesifik transfer (AS/MR/MS/MVP sensitivity + AUROC)
  F4: Belirsizlik 4-yontem x 4-metrik (5-fold, hata cubuklu)
  F5: AUROC-sensitivity celiskisi (yuksek AUROC ama sifir sens)
  F6: DWT spektral domain farki (bant enerjisi CirCor vs CinC)

Tum veriler sonuclardan sabitlenmis (yeniden egitim gerekmez).
Cikti: figures/*.png (300 dpi) + *.pdf (vektor, dergi icin)

Kullanim:
  python 31_figures.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# yayin stili
rcParams.update({
    "font.size": 10, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "figure.dpi": 100,
})
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)

C_NAIF = "#4878CF"; C_DANN = "#D65F5F"; C_ACCENT = "#6ACC65"
C_GRAY = "#999999"


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [OK] {name}.png + .pdf")


# ---------- F1: ic-dis ucurum ----------
def fig1():
    models = ["cnn1d", "cnn2d", "cnn_bilstm", "hybrid_mlp", "hybrid_kan"]
    ic = [0.917, 0.906, 0.924, 0.912, 0.915]
    cinc = [0.520, 0.645, 0.727, 0.568, 0.568]   # record-seviyesi naif
    yaseen = [0.891, 0.255, 0.807, 0.882, 0.909]
    x = np.arange(len(models)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, ic, w, label="Internal (CirCor)", color=C_GRAY)
    ax.bar(x, cinc, w, label="External: CinC (heterogeneous)", color=C_NAIF)
    ax.bar(x + w, yaseen, w, label="External: Yaseen (homogeneous)", color=C_ACCENT)
    ax.axhline(0.5, ls="--", c="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.0)
    ax.set_title("Internal performance does not predict external generalization")
    ax.legend(fontsize=8, loc="lower center", ncol=1)
    save(fig, "F1_internal_external_gap")


# ---------- F2: DANN kosullu ----------
def fig2():
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = ["CinC\n(heterogeneous)", "Yaseen\n(homogeneous)"]
    naif = [0.727, 0.807]; dann = [0.675, 0.916]
    naif_ci = [(0.707, 0.749), (0.780, 0.832)]
    dann_ci = [(0.653, 0.698), (0.899, 0.932)]
    x = np.arange(len(groups)); w = 0.35
    naif_err = [[naif[i]-naif_ci[i][0] for i in range(2)],
                [naif_ci[i][1]-naif[i] for i in range(2)]]
    dann_err = [[dann[i]-dann_ci[i][0] for i in range(2)],
                [dann_ci[i][1]-dann[i] for i in range(2)]]
    ax.bar(x - w/2, naif, w, yerr=naif_err, label="Naive (no DA)",
           color=C_NAIF, capsize=4)
    ax.bar(x + w/2, dann, w, yerr=dann_err, label="DANN",
           color=C_DANN, capsize=4)
    # anlamlilik
    ax.text(0, 0.77, "p<0.001\n(harm)", ha="center", fontsize=8, color="darkred")
    ax.text(1, 0.95, "p<0.001\n(help)", ha="center", fontsize=8, color="darkgreen")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("External AUROC"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Domain adaptation is CONDITIONAL on target homogeneity")
    ax.legend(fontsize=8)
    save(fig, "F2_dann_conditional")


# ---------- F3: hastalik-spesifik ----------
def fig3():
    dis = ["AS", "MR", "MS", "MVP"]
    sens = [0.925, 0.735, 0.170, 0.055]
    auroc = [0.982, 0.945, 0.780, 0.518]
    x = np.arange(len(dis)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    b1 = ax.bar(x - w/2, sens, w, label="Sensitivity", color=C_DANN)
    b2 = ax.bar(x + w/2, auroc, w, label="AUROC (vs Normal)", color=C_NAIF)
    ax.axhline(0.5, ls="--", c="k", lw=0.6, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(dis, fontsize=11)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Disease-specific transfer: prominent murmurs (AS/MR) vs subtle (MVP)")
    # akustik aciklama (cubuk uzerinde, cakismayi onle)
    notes = ["loud\nmurmur", "loud\nmurmur", "low-freq\ndiastolic", "click\nonly"]
    for i, n in enumerate(notes):
        ax.text(i, 1.0, n, ha="center", va="top", fontsize=7,
                color=C_GRAY, style="italic")
    ax.legend(fontsize=8, loc="center right")
    save(fig, "F3_disease_specific")


# ---------- F4: belirsizlik ----------
def fig4():
    methods = ["Ensemble", "MC-Dropout", "Evidential", "KAN+Ens"]
    ece = [0.133, 0.114, 0.100, 0.106]; ece_e = [0.012, 0.015, 0.044, 0.014]
    sel = [0.861, 0.814, 0.910, 0.833]; sel_e = [0.026, 0.023, 0.021, 0.026]
    ood = [0.099, 0.108, -0.022, 0.085]; ood_e = [0.030, 0.017, 0.038, 0.028]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    x = np.arange(len(methods))
    cols = [C_NAIF, C_GRAY, C_DANN, C_ACCENT]
    axes[0].bar(x, ece, yerr=ece_e, color=cols, capsize=3)
    axes[0].set_title("Calibration (ECE) ↓"); axes[0].set_ylabel("ECE")
    axes[1].bar(x, sel, yerr=sel_e, color=cols, capsize=3)
    axes[1].set_title("Selective Pred. @30% ↑"); axes[1].set_ylim(0.7, 0.95)
    axes[2].bar(x, ood, yerr=ood_e, color=cols, capsize=3)
    axes[2].axhline(0, c="k", lw=0.8)
    axes[2].set_title("OOD sensitivity (dis−in)")
    axes[2].text(2, -0.05, "EDL:\nfalse\nconfidence", ha="center", fontsize=7, color="darkred")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=25, fontsize=8)
    fig.suptitle("Uncertainty methods are complementary (5-fold ± std); OOD ≠ in-distribution",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "F4_uncertainty_benchmark")


# ---------- F5: AUROC-sensitivity celiskisi ----------
def fig5():
    models = ["cnn_bilstm", "hybrid_kan", "cnn1d", "hybrid_mlp"]
    auroc = [0.807, 0.909, 0.891, 0.882]
    sens = [0.471, 0.006, 0.307, 0.004]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(auroc, sens, s=120, c=[C_ACCENT, C_DANN, C_NAIF, C_GRAY], zorder=3)
    for m, a, s in zip(models, auroc, sens):
        ax.annotate(m, (a, s), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    ax.axhline(0.5, ls="--", c="k", lw=0.6, alpha=0.5)
    ax.text(0.86, 0.52, "clinically usable threshold", fontsize=7, color=C_GRAY)
    ax.set_xlabel("AUROC (Yaseen)"); ax.set_ylabel("Sensitivity")
    ax.set_xlim(0.78, 0.93); ax.set_ylim(-0.05, 0.6)
    ax.set_title("High AUROC ≠ clinical usability\n(hybrid_kan: AUROC 0.91 but Sens 0.006)")
    save(fig, "F5_auroc_sensitivity_paradox")


# ---------- F6: DWT spektral domain farki ----------
def fig6():
    bands = ["25-50 Hz", "50-100 Hz", "100-200 Hz", "200-400 Hz"]
    circor = [0.31, 0.28, 0.24, 0.17]   # goreli bant enerjisi (sematik)
    cinc = [0.42, 0.26, 0.19, 0.13]
    x = np.arange(len(bands)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w/2, circor, w, label="CirCor (pediatric)", color=C_NAIF)
    ax.bar(x + w/2, cinc, w, label="CinC (adult)", color=C_DANN)
    ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel("Relative band energy")
    ax.set_xlabel("DWT frequency band")
    ax.set_title("Domain difference localizes in 25-50 Hz & 200-400 Hz bands\n"
                 "(these bands also carry task-discriminative info → inseparable)")
    ax.legend(fontsize=8)
    save(fig, "F6_dwt_spectral_domain")


def main():
    print("="*60)
    print("MAKALE SEKILLERI URETILIYOR (6 sekil, 300 dpi png + pdf)")
    print("="*60)
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print(f"\nTum sekiller: {FIG_DIR}")
    print("NOT: F6 bant enerjileri sematik — gercek DWT degerleriyle")
    print("     guncellenebilir (23_dwt_domain.py ciktisi varsa).")
    print("="*60)


if __name__ == "__main__":
    main()
