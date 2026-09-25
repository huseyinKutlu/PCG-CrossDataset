#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50_threshold_pr_analysis.py
Hakem 1 (majör 1) + Hakem 2 (madde 3) — isletim noktasi / esik transferi analizi.

Uretir:
  (1) Her model x her veri kumesi icin:
        AUROC, Average Precision (AP), sinif prevalansi (no-skill referansi)
  (2) Uc esikte sens/spec + Wilson %95 GA + dengeli dogruluk:
        - sabit 0.5
        - KAYNAK-turevli Youden-J esigi (ic CirCor OOF tahminlerinden)
        - HEDEF-optimal Youden-J esigi (oracle; hedef etiketleri gerektirir)
      -> "ayrim basarisizligi" ile "esik transfer basarisizligi" ayrisir
  (3) PR egrileri (Sekil 10)
  (4) Tum kayit-duzeyi olasiliklari diske yazar -> bir daha GPU gerekmez

Cikti:
  reports/probs/<model>.npz            (in_p,in_y,cinc_p,cinc_y,yas_p,yas_y)
  reports/threshold_analysis.json
  reports/threshold_analysis.csv
  reports/threshold_analysis.txt       (makaleye yapistirmaya hazir ozet)
  figures/F10_pr_curves.png / .pdf

Kullanim:
  cd ~/Desktop/pcg_project
  python 50_threshold_pr_analysis.py
  python 50_threshold_pr_analysis.py --models cnn_bilstm,mambaconformer   # hizli deneme
  python 50_threshold_pr_analysis.py --reuse                              # diskteki probs'u kullan
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (roc_curve, average_precision_score,
                             precision_recall_curve)

rcParams.update({"font.size": 9, "font.family": "sans-serif",
                 "axes.spines.top": False, "axes.spines.right": False,
                 "axes.linewidth": 0.8})

ev = import_module("16_external_val")
ds_mod = import_module("05_dataset")
st = import_module("47_statistics_revision")

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
PROB_DIR = REPORT_DIR / "probs"; PROB_DIR.mkdir(exist_ok=True)
FIG_DIR = PROJECT_ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
DEVICE = ev.DEVICE
N_FOLDS = ev.N_FOLDS
CKPT_DIR = ev.CKPT_DIR

DEEP_MODELS = st.DEEP_MODELS
PRETTY = {"cnn1d": "1D-CNN", "cnn2d": "2D-CNN", "resnet1d": "ResNet1D",
          "cnn_bilstm": "CNN-BiLSTM", "hybrid_mlp": "Hybrid-MLP",
          "hybrid_kan": "Hybrid-KAN", "efficientnet": "EfficientNet-B0",
          "maxvit": "MaxViT", "mambaconformer": "PCG-MambaConformer"}
SPLITS = ["internal", "cinc", "yaseen"]
SPLIT_LABEL = {"internal": "Internal (CirCor)",
               "cinc": "External: CinC 2016 (heterogeneous)",
               "yaseen": "External: Yaseen 2018 (homogeneous)"}


# ------------------------------------------------------------ toplama
def collect_all(mdl, cv_csv, root, cinc_manifest, yaseen_manifest, num_workers=8):
    """Her fold checkpoint'ini BIR kez yukleyip ic + CinC + Yaseen tahminlerini
    ayni dongude uretir (47'deki collect_probs'un iki kez ic hesaplamasini onler)."""
    need_cwt = (mdl == "mambaconformer")

    def make_ext(kind, manifest):
        DS = ev.YaseenDataset if kind == "yaseen" else ev.CinCDataset
        d = DS(manifest, root, norm_adapt="none")
        d.dwt_norm = False
        if need_cwt:
            d.return_cwt = True
        return DataLoader(d, batch_size=128, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    cinc_loader = make_ext("cinc", cinc_manifest)
    yas_loader = make_ext("yaseen", yaseen_manifest)

    in_p, in_y = [], []
    cinc_folds, cinc_y = [], None
    yas_folds, yas_y = [], None

    for fold in range(N_FOLDS):
        ckpt = CKPT_DIR / f"{mdl}_fold{fold}.pth"
        if not ckpt.exists():
            print(f"  !! eksik checkpoint: {ckpt.name}")
            return None
        model, mode = ev.build_binary_model(mdl)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])

        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root,
                                         fold=fold, split="val")
        if need_cwt:
            va.return_cwt = True
        vl = ds_mod.make_eval_loader(va, batch_size=128, num_workers=num_workers)
        rp, ry, _ = ev.predict_grouped(model, vl, mode, "patient_id", circor=True)
        in_p.append(rp); in_y.append(ry)

        cp, cy, _ = ev.predict_grouped(model, cinc_loader, mode,
                                       "record_id", circor=False)
        cinc_folds.append(cp); cinc_y = cy

        yp, yy, _ = ev.predict_grouped(model, yas_loader, mode,
                                       "record_id", circor=False)
        yas_folds.append(yp); yas_y = yy
        print(f"    fold {fold} bitti")

    return {"in_p": np.concatenate(in_p), "in_y": np.concatenate(in_y),
            "cinc_p": np.mean(cinc_folds, axis=0), "cinc_y": cinc_y,
            "yas_p": np.mean(yas_folds, axis=0), "yas_y": yas_y}


# ------------------------------------------------------------ metrikler
def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(np.asarray(y).astype(int), np.asarray(p, float))
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def op_point(y, p, thr):
    """Verilen esikte sens/spec + Wilson GA + dengeli dogruluk + F1."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p, float) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens, slo, shi = st.wilson_ci(tp, tp + fn)
    spec, plo, phi = st.wilson_ci(tn, tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv == ppv and (ppv + sens) > 0) else float("nan")
    return {"threshold": float(thr),
            "sens": sens, "sens_ci": [slo, shi], "sens_n": [tp, tp + fn],
            "spec": spec, "spec_ci": [plo, phi], "spec_n": [tn, tn + fp],
            "balanced_acc": (sens + spec) / 2.0, "ppv": ppv, "f1": f1}


def bootstrap_ap_ci(y, p, n_boot=2000, seed=42):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        a = float(average_precision_score(y, p)); return a, a, a
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        b[i] = average_precision_score(y[idx], p[idx])
    return (float(average_precision_score(y, p)),
            float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


# ------------------------------------------------------------ ana
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEEP_MODELS))
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--reuse", action="store_true",
                    help="reports/probs/ altindaki kayitli olasiliklari kullan")
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
    yaseen_manifest = root / "manifests" / "yaseen_processed.csv"
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    probs, results = {}, {}

    for mdl in models:
        print(f"\n=== {mdl} ===")
        npz = PROB_DIR / f"{mdl}.npz"
        if args.reuse and npz.exists():
            d = np.load(npz)
            probs[mdl] = {k: d[k] for k in d.files}
            print("  diskteki olasiliklar kullanildi")
        else:
            d = collect_all(mdl, cv_csv, root, cinc_manifest,
                            yaseen_manifest, args.num_workers)
            if d is None:
                print("  atlandi"); continue
            np.savez(npz, **d)
            probs[mdl] = d
            print(f"  kaydedildi -> {npz.relative_to(root)}")

        d = probs[mdl]
        pairs = {"internal": (d["in_y"], d["in_p"]),
                 "cinc": (d["cinc_y"], d["cinc_p"]),
                 "yaseen": (d["yas_y"], d["yas_p"])}

        # kaynak-turevli esik: IC verideki Youden-J
        thr_src = youden_threshold(pairs["internal"][0], pairs["internal"][1])

        res = {"threshold_source_youden": thr_src}
        for sp in SPLITS:
            y, p = pairs[sp]
            auc, alo, ahi = st.bootstrap_auroc_ci(y, p, args.n_boot)
            apv, aplo, aphi = bootstrap_ap_ci(y, p, args.n_boot)
            prev = float(np.mean(np.asarray(y).astype(int)))
            thr_tgt = youden_threshold(y, p)
            res[sp] = {
                "n": int(len(y)), "prevalence": prev,
                "auroc": auc, "auroc_ci": [alo, ahi],
                "ap": apv, "ap_ci": [aplo, aphi],
                "ap_gain_over_noskill": apv - prev,
                "fixed_0.5": op_point(y, p, 0.5),
                "source_youden": op_point(y, p, thr_src),
                "target_oracle": op_point(y, p, thr_tgt),
            }
            f = res[sp]
            print(f"  {sp:<9} AUROC={auc:.3f} AP={apv:.3f} (no-skill {prev:.3f}) | "
                  f"0.5: {f['fixed_0.5']['sens']:.3f}/{f['fixed_0.5']['spec']:.3f} | "
                  f"src-J: {f['source_youden']['sens']:.3f}/{f['source_youden']['spec']:.3f} | "
                  f"oracle: {f['target_oracle']['sens']:.3f}/{f['target_oracle']['spec']:.3f}")
        results[mdl] = res

    # ------------------------------------------------- kayit: json / csv / txt
    (REPORT_DIR / "threshold_analysis.json").write_text(
        json.dumps({"per_model": results, "n_boot": args.n_boot,
                    "note": ("Esikler: sabit 0.5; kaynak-turevli Youden-J (ic CirCor); "
                             "hedef-optimal Youden-J (oracle, hedef etiketi gerektirir). "
                             "GA'lar: AUROC/AP stratified bootstrap, sens/spec Wilson.")},
                   indent=2), encoding="utf-8")

    with open(REPORT_DIR / "threshold_analysis.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "split", "n", "prevalence", "auroc", "auroc_lo", "auroc_hi",
                    "ap", "ap_lo", "ap_hi", "rule", "threshold",
                    "sens", "sens_lo", "sens_hi", "spec", "spec_lo", "spec_hi",
                    "balanced_acc", "ppv", "f1"])
        for mdl, r in results.items():
            for sp in SPLITS:
                d = r[sp]
                for rule in ["fixed_0.5", "source_youden", "target_oracle"]:
                    o = d[rule]
                    w.writerow([PRETTY.get(mdl, mdl), sp, d["n"], f"{d['prevalence']:.4f}",
                                f"{d['auroc']:.4f}", f"{d['auroc_ci'][0]:.4f}", f"{d['auroc_ci'][1]:.4f}",
                                f"{d['ap']:.4f}", f"{d['ap_ci'][0]:.4f}", f"{d['ap_ci'][1]:.4f}",
                                rule, f"{o['threshold']:.4f}",
                                f"{o['sens']:.4f}", f"{o['sens_ci'][0]:.4f}", f"{o['sens_ci'][1]:.4f}",
                                f"{o['spec']:.4f}", f"{o['spec_ci'][0]:.4f}", f"{o['spec_ci'][1]:.4f}",
                                f"{o['balanced_acc']:.4f}", f"{o['ppv']:.4f}", f"{o['f1']:.4f}"])

    lines = []
    def L(s=""):
        print(s); lines.append(s)
    L("=" * 100)
    L("ESIK TRANSFER ANALIZI — makale tablosuna hazir ozet")
    L("=" * 100)
    for sp in SPLITS:
        L("")
        L(f"--- {SPLIT_LABEL[sp]} ---")
        L(f"{'Model':<20}{'AUROC':<10}{'AP':<10}{'NoSkill':<9}"
          f"{'0.5 Sens/Spec':<18}{'srcJ Sens/Spec':<18}{'oracle Sens/Spec':<18}{'BalAcc(0.5->oracle)'}")
        for mdl, r in results.items():
            d = r[sp]
            L(f"{PRETTY.get(mdl, mdl):<20}"
              f"{d['auroc']:.3f}     {d['ap']:.3f}     {d['prevalence']:.3f}    "
              f"{d['fixed_0.5']['sens']:.3f}/{d['fixed_0.5']['spec']:.3f}       "
              f"{d['source_youden']['sens']:.3f}/{d['source_youden']['spec']:.3f}       "
              f"{d['target_oracle']['sens']:.3f}/{d['target_oracle']['spec']:.3f}       "
              f"{d['fixed_0.5']['balanced_acc']:.3f} -> {d['target_oracle']['balanced_acc']:.3f}")
    L("")
    L("Kaynak-turevli Youden-J esikleri (ic CirCor uzerinde):")
    for mdl, r in results.items():
        L(f"  {PRETTY.get(mdl, mdl):<20} thr = {r['threshold_source_youden']:.4f}")
    (REPORT_DIR / "threshold_analysis.txt").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------- Sekil 10: PR egrileri
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    cmap = plt.get_cmap("tab10")
    for ax, sp in zip(axes, SPLITS):
        for i, mdl in enumerate(results):
            d = probs[mdl]
            y, p = ({"internal": (d["in_y"], d["in_p"]),
                     "cinc": (d["cinc_y"], d["cinc_p"]),
                     "yaseen": (d["yas_y"], d["yas_p"])})[sp]
            pr, rc, _ = precision_recall_curve(np.asarray(y).astype(int),
                                               np.asarray(p, float))
            lw = 2.0 if mdl == "mambaconformer" else 1.0
            ax.plot(rc, pr, lw=lw, color=cmap(i % 10),
                    label=f"{PRETTY.get(mdl, mdl)} (AP {results[mdl][sp]['ap']:.2f})")
        prev = results[list(results)[0]][sp]["prevalence"]
        ax.axhline(prev, ls="--", c="k", lw=0.7, alpha=0.6)
        ax.text(0.02, prev + 0.02, f"no-skill = {prev:.2f}", fontsize=7)
        ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision")
        ax.set_title(SPLIT_LABEL[sp], fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    axes[-1].legend(fontsize=6, loc="lower left", frameon=False, ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"F10_pr_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n[OK] figures/F10_pr_curves.png + .pdf")
    print("[OK] reports/threshold_analysis.json / .csv / .txt")
    print("[OK] reports/probs/*.npz  (bundan sonra --reuse ile GPU'suz calisir)")


if __name__ == "__main__":
    main()
