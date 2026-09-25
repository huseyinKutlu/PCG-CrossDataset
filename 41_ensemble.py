#!/usr/bin/env python3
"""
41_ensemble.py
Multi-architecture ensemble degerlendirmesi (mevcut checkpoint'lerden).

UC STRATEJI:
  A) Naif voting (TUM modeller) — zit biaslar dengeleniyor mu, yoksa yaniltici mi?
  B) Sadece DENGELI modeller (cnn_bilstm + mambaconformer)
  C) MambaConformer TEK BASINA vs her ensemble — tek iyi model > ensemble mi?

Her model record-seviyesi abnormal-olasiligi uretir (5-fold kendi ensemble'i),
sonra mimari-arasi soft-voting (olasilik ortalamasi).

Kullanim:
  python 41_ensemble.py --ext_dataset yaseen
  python 41_ensemble.py --ext_dataset cinc
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

ext_mod = import_module("16_external_val")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CKPT = PROJECT_ROOT / "checkpoints" / "binary"

# model -> (mode) eslemesi
MODELS = {
    "cnn1d": "raw", "cnn2d": "mel", "cnn_bilstm": "raw",
    "hybrid_mlp": "dual", "hybrid_kan": "dual",
    "efficientnet": "mel", "maxvit": "mel",
    "resnet1d": "raw", "mambaconformer": "tri",
}
DENGELI = ["cnn_bilstm", "mambaconformer"]  # dengeli (Spec~1, makul Sens)


@torch.no_grad()
def model_record_probs(mname, mode, ExtDS, manifest):
    """Bir modelin 5-fold ensemble record-seviyesi abnormal olasiligi."""
    ds = ExtDS(manifest, PROJECT_ROOT, norm_adapt="none")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    fold_p, ref_y, ref_id = [], None, None
    for fold in range(N_FOLDS):
        ck = CKPT / f"{mname}_fold{fold}.pth"
        if not ck.exists():
            continue
        model, _ = ext_mod.build_binary_model(mname)
        model = model.to(DEVICE)
        sd = torch.load(ck, map_location=DEVICE)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        sp, sy, si = [], [], []
        for b in loader:
            if mode == "tri":
                logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE), b["cwt"].to(DEVICE))
            elif mode == "dual":
                logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE))
            else:
                logits = model(b[mode].to(DEVICE))
            sp.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            sy.append(b["label"].numpy()); si.extend(b["record_id"])
        sp = np.concatenate(sp); sy = np.concatenate(sy); si = np.array(si)
        uniq = np.unique(si)
        rp = np.array([sp[si == u].mean() for u in uniq])
        ry = np.array([sy[si == u][0] for u in uniq])
        fold_p.append(rp)
        if ref_y is None: ref_y, ref_id = ry, uniq
    if not fold_p:
        return None, None, None
    return np.mean(fold_p, 0), ref_y, ref_id


def metrics(y, p, name):
    auc = roc_auc_score(y, p)
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    sens = cm[1, 1] / max(cm[1].sum(), 1)
    spec = cm[0, 0] / max(cm[0].sum(), 1)
    print(f"  {name:32s} AUROC={auc:.4f} | Sens={sens:.3f} Spec={spec:.3f}")
    return auc, sens, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext_dataset", default="yaseen", choices=["cinc", "yaseen"])
    args = ap.parse_args()
    ExtDS = ext_mod.YaseenDataset if args.ext_dataset == "yaseen" else ext_mod.CinCDataset
    manifest = PROJECT_ROOT / "manifests" / (
        "yaseen_processed.csv" if args.ext_dataset == "yaseen" else "cinc2016_processed.csv")

    print("#"*64)
    print(f"# MULTI-ARCHITECTURE ENSEMBLE — {args.ext_dataset.upper()}")
    print("#"*64)

    # tum modellerin olasiliklarini topla
    print("\nModel tahminleri toplaniyor...")
    probs, ref_y = {}, None
    for m, mode in MODELS.items():
        p, y, ids = model_record_probs(m, mode, ExtDS, manifest)
        if p is not None:
            probs[m] = p
            if ref_y is None: ref_y = y
            print(f"  {m}: OK ({len(p)} record)")
        else:
            print(f"  {m}: checkpoint yok, atla")

    print("\n" + "="*64)
    print("BIREYSEL MODELLER (referans)")
    print("="*64)
    for m in probs:
        metrics(ref_y, probs[m], m)

    print("\n" + "="*64)
    print("STRATEJI A: Naif voting (TUM modeller)")
    print("="*64)
    all_stack = np.mean([probs[m] for m in probs], 0)
    metrics(ref_y, all_stack, "Ensemble-ALL (soft voting)")
    print("  >> Yorum: zit-bias modeller (hep-normal + hep-abnormal) birlesti.")
    print("     AUROC iyi gorunse de Sens/Spec dengesine bak — gercek mi yaniltici mi?")

    print("\n" + "="*64)
    print("STRATEJI B: Sadece DENGELI modeller")
    print("="*64)
    dengeli_var = [m for m in DENGELI if m in probs]
    if dengeli_var:
        bal_stack = np.mean([probs[m] for m in dengeli_var], 0)
        metrics(ref_y, bal_stack, f"Ensemble-DENGELI {dengeli_var}")
    print("  >> Yorum: dengeli modeller birlesince MambaConformer'i geciyor mu?")

    print("\n" + "="*64)
    print("STRATEJI C: MambaConformer TEK vs ensemble'lar")
    print("="*64)
    if "mambaconformer" in probs:
        mc_auc = roc_auc_score(ref_y, probs["mambaconformer"])
        all_auc = roc_auc_score(ref_y, all_stack)
        print(f"  MambaConformer tek:    AUROC={mc_auc:.4f}")
        print(f"  Ensemble-ALL:          AUROC={all_auc:.4f}")
        if dengeli_var:
            bal_auc = roc_auc_score(ref_y, bal_stack)
            print(f"  Ensemble-DENGELI:      AUROC={bal_auc:.4f}")
        print(f"  >> Sonuc: tek iyi model ensemble'i geciyor mu?")
        best_ens = max(all_auc, bal_auc if dengeli_var else 0)
        if mc_auc >= best_ens:
            print(f"     EVET — MambaConformer ({mc_auc:.4f}) >= en iyi ensemble ({best_ens:.4f})")
            print(f"     BULGU: tek iyi-tasarlanmis cok-temsil model > naif model-birlestirme")
        else:
            print(f"     HAYIR — ensemble ({best_ens:.4f}) > MambaConformer ({mc_auc:.4f})")

    print("\n" + "#"*64)
    print("# ENSEMBLE DEGERLENDIRME TAMAMLANDI")
    print("#"*64)


if __name__ == "__main__":
    main()
