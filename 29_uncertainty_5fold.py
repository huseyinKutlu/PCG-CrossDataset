#!/usr/bin/env python3
"""
29_uncertainty_5fold.py
Belirsizlik benchmark'ini (25) TUM fold'larda calistir, ortalama+-std raporla.
Hakem "tek fold mu?" sorusuna cevap — saglamlik.

Her fold icin 25_uncertainty_benchmark'i train+eval cagirir, sonuclari toplar.
4 yontem x 4 metrik x N_fold -> ortalama+-std tablo.

Kullanim:
  python 29_uncertainty_5fold.py --folds 0 1 2 3 4 --epochs 30
  # veya hizli: python 29_uncertainty_5fold.py --folds 0 1 2 --epochs 30
"""
from __future__ import annotations
import argparse, subprocess, sys, re
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)


def parse_summary(text):
    """25'in ozet tablosunu parse et: yontem -> {ece, sel30, ood, unk}."""
    res = {}
    # satir ornegi: "Ensemble      0.0935    0.8251   +0.0512     EVET"
    for line in text.splitlines():
        m = re.match(r"\s*(Ensemble|MC-Dropout|Evidential|KAN\+Ens)\s+"
                     r"([\d.]+)\s+([\d.]+)\s+([+\-][\d.]+)\s+(\w+)", line)
        if m:
            res[m.group(1)] = {
                "ece": float(m.group(2)), "sel30": float(m.group(3)),
                "ood": float(m.group(4)), "unk": m.group(5)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip_train", action="store_true",
                    help="egitim atла (checkpoint'ler varsa sadece eval)")
    args = ap.parse_args()

    print("="*70)
    print(f"BELIRSIZLIK 5-FOLD SAGLAMLIK | folds={args.folds}")
    print("="*70)

    per_fold = {}
    for fold in args.folds:
        print(f"\n{'#'*60}\n# FOLD {fold}\n{'#'*60}")
        if not args.skip_train:
            print(f"[fold {fold}] egitim...")
            r = subprocess.run(
                [sys.executable, "25_uncertainty_benchmark.py",
                 "--fold", str(fold), "--epochs", str(args.epochs), "--train"],
                capture_output=True, text=True)
            print(r.stdout[-500:] if r.stdout else "")
            if r.returncode != 0:
                print(f"[HATA fold {fold} train]\n{r.stderr[-1000:]}"); continue
        print(f"[fold {fold}] degerlendirme...")
        r = subprocess.run(
            [sys.executable, "25_uncertainty_benchmark.py", "--fold", str(fold), "--eval"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[HATA fold {fold} eval]\n{r.stderr[-1000:]}"); continue
        print(r.stdout[-800:])
        parsed = parse_summary(r.stdout)
        if parsed:
            per_fold[fold] = parsed

    if not per_fold:
        print("\n[HATA] hicbir fold sonucu parse edilemedi."); return

    # === ortalama+-std tablo ===
    methods = ["Ensemble", "MC-Dropout", "Evidential", "KAN+Ens"]
    metrics = ["ece", "sel30", "ood"]
    metric_names = {"ece": "ECE", "sel30": "Sel@30%", "ood": "OOD-fark"}

    print("\n" + "="*70)
    print(f"5-FOLD OZET (ortalama +- std, n={len(per_fold)} fold)")
    print("="*70)
    out = [f"5-FOLD BELIRSIZLIK SAGLAMLIK (n={len(per_fold)} fold)\n"]
    header = f"{'Yontem':<12}" + "".join(f"{metric_names[m]:>16}" for m in metrics) + f"{'Unk-hiz':>10}"
    print(header); out.append(header)
    print("-"*70); out.append("-"*70)
    for meth in methods:
        vals = {m: [] for m in metrics}
        unks = []
        for fold, res in per_fold.items():
            if meth in res:
                for m in metrics:
                    vals[m].append(res[meth][m])
                unks.append(res[meth]["unk"])
        if not vals["ece"]:
            continue
        row = f"{meth:<12}"
        for m in metrics:
            arr = np.array(vals[m])
            row += f"{arr.mean():>8.3f}±{arr.std():<6.3f}"
        # Unknown hizalama: kac fold'da EVET
        n_evet = sum(1 for u in unks if u.upper().startswith("E") or u.upper()=="EVET")
        row += f"{n_evet}/{len(unks):>9}"
        print(row); out.append(row)

    print("\nUnk-hiz: kac fold'da Unknown en yuksek belirsizligi aldi")
    print("OOD-fark: + ise OOD-duyarli, - ise yanlis guven (EDL'de negatif beklenir)")

    rep = REPORT_DIR / "uncertainty_5fold.txt"
    rep.write_text("\n".join(out), encoding="utf-8")
    print(f"\nRapor: {rep}")
    print("=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
