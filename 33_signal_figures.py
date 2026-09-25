#!/usr/bin/env python3
"""
33_signal_figures.py
PCG SINYAL gorselleri (gercek cache verisinden).

Sekiller:
  F8: Datasetler arasi akustik karsilastirma
      CirCor (pediatrik) vs CinC (yetiskin) vs Yaseen (ders kitabi)
      her biri: dalga formu + log-mel spektrogram
  F9: Hastalik-spesifik dalga formu (Yaseen) — NEDEN AS taninir, MVP kacar
      AS (belirgin ufurum) / MR / MS / MVP (sadece klik) / N (normal)
      her biri: dalga formu + spektrogram
      -> F3'teki sayisal bulguya MEKANISTIK gorsel kanit

Veri: cache/*/*.npz (raw_segs, mel_segs).

Kullanim:
  python 33_signal_figures.py
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
CACHE = PROJECT_ROOT / "cache"
SR = 2000


def load_one(npz_path):
    """bir .npz'den ilk segmentin raw + mel'ini dondur."""
    d = np.load(npz_path, allow_pickle=True)
    raw = d["raw_segs"][0]   # (10000,)
    mel = d["mel_segs"][0]   # (64, 201)
    return raw, mel


def find_circor_example(murmur_target):
    """CirCor cache'den verilen murmur etiketli bir ornek bul (anahtar: 'murmur')."""
    for f in sorted(glob.glob(str(CACHE / "circor*" / "*.npz")))[:500]:
        d = np.load(f, allow_pickle=True)
        mur = str(d["murmur"]) if "murmur" in d else ""
        if murmur_target.lower() == mur.lower():
            return f
    fs = sorted(glob.glob(str(CACHE / "circor*" / "*.npz")))
    return fs[0] if fs else None


def find_yaseen_class(cls):
    """Yaseen cache'den verilen sinif (AS/MR/MS/MVP/N) ornegi."""
    cands = sorted(glob.glob(str(CACHE / "yaseen" / f"New_{cls}_*.npz")))
    return cands[0] if cands else None


def plot_wave_mel(ax_w, ax_m, raw, mel, title):
    t = np.arange(len(raw)) / SR
    ax_w.plot(t, raw, lw=0.5, c="#333333")
    ax_w.set_title(title, fontsize=9)
    ax_w.set_xlim(0, len(raw) / SR); ax_w.set_ylabel("Amp")
    ax_w.set_yticks([])
    im = ax_m.imshow(mel, aspect="auto", origin="lower", cmap="magma",
                     extent=[0, len(raw) / SR, 0, 400])
    ax_m.set_ylabel("Hz"); ax_m.set_xlabel("Time (s)")
    return im


def fig8_datasets():
    """3 dataset akustik karsilastirma (her biri abnormal ornek)."""
    items = []
    c = find_circor_example("Present")
    if c: items.append((c, "CirCor (pediatric, murmur)"))
    cinc = sorted(glob.glob(str(CACHE / "cinc2016" / "*.npz")))
    if cinc: items.append((cinc[0], "CinC (adult, abnormal)"))
    yas = find_yaseen_class("AS")
    if yas: items.append((yas, "Yaseen (textbook, AS)"))

    if not items:
        print("  [F8 atla] cache ornegi bulunamadi"); return
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 5),
                             gridspec_kw={"height_ratios": [1, 1.4]})
    if n == 1: axes = axes.reshape(2, 1)
    im = None
    for j, (path, title) in enumerate(items):
        raw, mel = load_one(path)
        im = plot_wave_mel(axes[0, j], axes[1, j], raw, mel, title)
    fig.suptitle("Cross-dataset acoustic comparison (waveform + log-mel spectrogram)",
                 fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.10, wspace=0.22, hspace=0.3)
    cax = fig.add_axes([0.92, 0.10, 0.012, 0.4])
    fig.colorbar(im, cax=cax, label="dB")
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"F8_dataset_signals.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] F8_dataset_signals.png + .pdf")


def fig9_diseases():
    """Yaseen 5 sinif dalga formu — AS taninir, MVP kacar (mekanistik)."""
    classes = ["N", "AS", "MR", "MS", "MVP"]
    titles = {"N": "Normal", "AS": "AS (sens=0.93)", "MR": "MR (sens=0.74)",
              "MS": "MS (sens=0.17)", "MVP": "MVP (sens=0.06)"}
    items = [(find_yaseen_class(c), titles[c]) for c in classes]
    items = [(p, t) for p, t in items if p]
    if not items:
        print("  [F9 atla] Yaseen cache bulunamadi"); return
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 4.5),
                             gridspec_kw={"height_ratios": [1, 1.3]})
    im = None
    for j, (path, title) in enumerate(items):
        raw, mel = load_one(path)
        im = plot_wave_mel(axes[0, j], axes[1, j], raw, mel, title)
        if j > 0:
            axes[0, j].set_ylabel(""); axes[1, j].set_ylabel("")
    fig.suptitle("Why AS transfers but MVP fails: prominent murmur vs subtle click",
                 fontsize=11)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.90, bottom=0.10, wspace=0.25, hspace=0.3)
    cax = fig.add_axes([0.93, 0.10, 0.012, 0.35])
    fig.colorbar(im, cax=cax, label="dB")
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"F9_disease_signals.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] F9_disease_signals.png + .pdf")


def fig10_circor_classes():
    """CirCor 3 murmur classes: Present / Unknown / Absent (training data)."""
    classes = ["Present", "Unknown", "Absent"]
    titles = {"Present": "Present (murmur)", "Unknown": "Unknown (uncertain)",
              "Absent": "Absent (no murmur)"}
    items = [(find_circor_example(c), titles[c]) for c in classes]
    items = [(p, t) for p, t in items if p]
    if not items:
        print("  [F10 skip] CirCor cache not found"); return
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 5),
                             gridspec_kw={"height_ratios": [1, 1.4]})
    if n == 1: axes = axes.reshape(2, 1)
    im = None
    for j, (path, title) in enumerate(items):
        raw, mel = load_one(path)
        im = plot_wave_mel(axes[0, j], axes[1, j], raw, mel, title)
        if j > 0:
            axes[0, j].set_ylabel(""); axes[1, j].set_ylabel("")
    fig.suptitle("CirCor murmur classes (training data): Present / Unknown / Absent",
                 fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.10, wspace=0.22, hspace=0.3)
    cax = fig.add_axes([0.92, 0.10, 0.012, 0.4])
    fig.colorbar(im, cax=cax, label="dB")
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"F10_circor_classes.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] F10_circor_classes.png + .pdf")


def main():
    print("="*60)
    print("PCG SINYAL GORSELLERI (gercek cache verisinden)")
    print("="*60)
    fig8_datasets()
    fig9_diseases()
    fig10_circor_classes()
    print(f"\nSekiller: {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
