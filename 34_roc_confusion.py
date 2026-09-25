#!/usr/bin/env python3
"""
34_roc_confusion.py
ROC egrileri + Confusion matrix'ler (gercek model tahminlerinden).

Sekiller:
  F11: ROC egrileri — iç (CirCor) vs dış (CinC, Yaseen), cnn_bilstm
       + naif vs DANN overlay (kosullu etkiyi ROC'ta goster)
  F12: Confusion matrix'ler (Yaseen, esik 0.5)
       cnn_bilstm (dengeli) vs hybrid_kan (cokmus: tum abnormal->normal)
       -> AUROC-sensitivity celiskisinin GORSEL kaniti

Gercek checkpoint'lerden tahmin uretir (record-seviyesi).

Kullanim:
  python 34_roc_confusion.py
"""
from __future__ import annotations
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc, confusion_matrix
rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

ext_mod = import_module("16_external_val")
ds_mod = import_module("05_dataset")
PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "figures"; FIG_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CKPT_BINARY = PROJECT_ROOT / "checkpoints" / "binary"
CKPT_DANN = PROJECT_ROOT / "checkpoints" / "dann"


@torch.no_grad()
def predict_external(model_name, ckpt_dir, pattern, ExtDS, root, manifest, seeds=None):
    """record-seviyesi abnormal-olasiligi + label. ensemble."""
    ds = ExtDS(manifest, root, norm_adapt="none")
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)
    ckpts = []
    if seeds is None:
        for f in range(N_FOLDS):
            ck = ckpt_dir / pattern.format(fold=f)
            if ck.exists(): ckpts.append(ck)
    else:
        for s in seeds:
            for f in range(N_FOLDS):
                ck = ckpt_dir / pattern.format(seed=s, fold=f)
                if ck.exists(): ckpts.append(ck)
    if not ckpts:
        return None, None
    all_p, ref_y, ref_id = None, None, None
    for ck in ckpts:
        built = ext_mod.build_binary_model(model_name)
        model, mode = built[0].to(DEVICE), built[1]
        sd = torch.load(ck, map_location=DEVICE)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        model.load_state_dict(sd); model.eval()
        sp, sy, si = [], [], []
        for b in loader:
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE)) if mode == "dual" \
                     else model(b[mode].to(DEVICE))
            sp.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            sy.append(b["label"].numpy()); si.extend(b["record_id"])
        sp = np.concatenate(sp); sy = np.concatenate(sy); si = np.array(si)
        uniq = np.unique(si)
        rp = np.array([sp[si == u].mean() for u in uniq])
        ry = np.array([sy[si == u][0] for u in uniq])
        all_p = rp if all_p is None else all_p + rp
        if ref_y is None: ref_y, ref_id = ry, uniq
    return all_p / len(ckpts), ref_y


@torch.no_grad()
def predict_internal(model_name, root):
    """CirCor held-out (5-fold val birlesimi) — ic ROC icin."""
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    all_p, all_y = [], []
    for fold in range(N_FOLDS):
        ck = CKPT_BINARY / f"{model_name}_fold{fold}.pth"
        if not ck.exists(): continue
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=8)
        built = ext_mod.build_binary_model(model_name)
        model, mode = built[0].to(DEVICE), built[1]
        sd = torch.load(ck, map_location=DEVICE)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        model.load_state_dict(sd); model.eval()
        for b in loader:
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE)) if mode == "dual" \
                     else model(b[mode].to(DEVICE))
            p = torch.softmax(logits, 1).cpu().numpy()
            # 3-sinif -> ikili (abnormal = Present+Unknown)
            all_p.append(p[:, 0] + p[:, 1])
            all_y.append((b["label"].numpy() != 2).astype(int))
    if not all_p:
        return None, None
    return np.concatenate(all_p), np.concatenate(all_y)


def fig11_roc(root):
    """ROC: ic vs dis + naif vs DANN."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    cinc_m = root / "manifests" / "cinc2016_processed.csv"
    yas_m = root / "manifests" / "yaseen_processed.csv"

    # Panel A: ic vs dis (naif cnn_bilstm)
    ax = axes[0]
    pin, yin = predict_internal("cnn_bilstm", root)
    pc, yc = predict_external("cnn_bilstm", CKPT_BINARY, "cnn_bilstm_fold{fold}.pth",
                              ext_mod.CinCDataset, root, cinc_m)
    py, yy = predict_external("cnn_bilstm", CKPT_BINARY, "cnn_bilstm_fold{fold}.pth",
                              ext_mod.YaseenDataset, root, yas_m)
    for p, y, lab, c in [(pin, yin, "Internal (CirCor)", "#333333"),
                          (pc, yc, "External: CinC", "#4878CF"),
                          (py, yy, "External: Yaseen", "#6ACC65")]:
        if p is None: continue
        fpr, tpr, _ = roc_curve(y, p); a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{lab} (AUC={a:.3f})", color=c)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("(A) Internal vs External ROC (cnn_bilstm)")
    ax.legend(fontsize=8, loc="lower right")

    # Panel B: naif vs DANN (Yaseen — DANN'in yardim ettigi yer)
    ax = axes[1]
    pn, yn = predict_external("cnn_bilstm", CKPT_BINARY, "cnn_bilstm_fold{fold}.pth",
                              ext_mod.YaseenDataset, root, yas_m)
    pd_, yd = predict_external("cnn_bilstm", CKPT_DANN,
                               "cnn_bilstm_dann_s{seed}_fold{fold}.pth",
                               ext_mod.YaseenDataset, root, yas_m, seeds=[42, 123, 7])
    for p, y, lab, c in [(pn, yn, "Naive", "#4878CF"),
                         (pd_, yd, "DANN", "#D65F5F")]:
        if p is None: continue
        fpr, tpr, _ = roc_curve(y, p); a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{lab} (AUC={a:.3f})", color=c)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("(B) Naive vs DANN on Yaseen (DA helps here)")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"F11_roc_curves.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] F11_roc_curves.png + .pdf")


def fig12_confusion(root):
    """Confusion matrix: cnn_bilstm (dengeli) vs hybrid_kan (cokmus) — Yaseen."""
    yas_m = root / "manifests" / "yaseen_processed.csv"
    models = [("cnn_bilstm", "cnn_bilstm (balanced)"),
              ("hybrid_kan", "hybrid_kan (AUROC 0.91 but collapsed)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (mname, title) in zip(axes, models):
        p, y = predict_external(mname, CKPT_BINARY, mname + "_fold{fold}.pth",
                                ext_mod.YaseenDataset, root, yas_m)
        if p is None:
            ax.text(0.5, 0.5, f"{mname}\ncheckpoint yok", ha="center"); continue
        pred = (p >= 0.5).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=14, color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Abnormal"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Abnormal"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        # sensitivity hesapla
        sens = cm[1, 1] / max(cm[1].sum(), 1)
        ax.set_title(f"{title}\nSensitivity={sens:.3f}", fontsize=9)
    fig.suptitle("Confusion matrices (Yaseen): high AUROC ≠ clinical usability", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"F12_confusion_matrices.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  [OK] F12_confusion_matrices.png + .pdf")


def main():
    root = PROJECT_ROOT
    print("="*60)
    print("ROC EGRILERI + CONFUSION MATRIX (gercek tahminlerden)")
    print("="*60)
    fig11_roc(root)
    fig12_confusion(root)
    print(f"\nSekiller: {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
