#!/usr/bin/env python3
"""
20_domain_divergence.py
DA yontemlerinin (MMD/DANN/CORAL) ise yarayip yaramayacagini ONCEDEN teshis et.

FIKIR (A-distance / proxy domain divergence, Ben-David 2007):
  Egitilmis modelin ozellik uzayinda, bir siniflandirici CirCor mu CinC mi
  oldugunu ne kadar kolay ayirt ediyor?
    - Kolay ayirt (A-dist ~2): ozellikler AYRISIK -> DA potansiyel faydali
    - Ayirt edemiyor (A-dist ~0): ozellikler HIZALI -> DA etkisiz, sorun baska

  CORAL kaybi ~0.0005 cikmisti (cok kucuk). Eger A-distance de kucukse,
  ozellikler zaten hizali demektir ve DA yontemleri (MMD/DANN dahil) bosa
  cikar — sorun hizalama degil, ogrenilen ozelliklerin yetiskin patolojisini
  ayirt edememesi. Bu teshis, hangi yone gidecegimizi kesinlestirir.

Kullanim:
  python 20_domain_divergence.py --model hybrid_kan
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader

ds_mod = import_module("05_dataset")
ext_mod = import_module("16_external_val")
coral_mod = import_module("19_coral_da")

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "binary"   # ikili (normal/abnormal) ckpt'ler
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract_feats(model, loader, mode, mdl_name, max_n=2000):
    model.eval()
    feats = []
    for b in loader:
        raw = b["raw"].to(DEVICE); mel = b["mel"].to(DEVICE)
        f, _ = coral_mod.get_features_and_logits(model, raw, mel, mode, mdl_name)
        if f is None: continue
        feats.append(f.cpu().numpy())
        if sum(len(x) for x in feats) >= max_n: break
    return np.concatenate(feats)[:max_n]


def a_distance(src_feat, tgt_feat, seed=42):
    """Lojistik regresyon ile domain ayrimi -> A-distance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X = np.vstack([src_feat, tgt_feat])
    y = np.concatenate([np.zeros(len(src_feat)), np.ones(len(tgt_feat))])
    # standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
    err = 1 - acc
    a_dist = 2 * (1 - 2 * err)
    return acc, err, max(a_dist, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid_kan")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    root = PROJECT_ROOT
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"

    print("="*70)
    print(f"DOMAIN DIVERGENCE TESHISI (A-distance) | model={args.model}")
    print("="*70)

    # naif egitilmis model (CORAL'siz) — checkpoints/{model}_fold{fold}.pth
    ckpt = CKPT_DIR / f"{args.model}_fold{args.fold}.pth"
    if not ckpt.exists():
        print(f"!! ikili checkpoint yok: {ckpt}")
        print(f"   once: python 16_external_val.py --model {args.model} --train")
        return

    model, mode = ext_mod.build_binary_model(args.model)
    model = model.to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(sd["model"] if "model" in sd else sd)

    # CirCor (kaynak) val + CinC (hedef)
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="val")
    src_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=4)
    cinc = ext_mod.CinCDataset(cinc_manifest, root)
    tgt_loader = DataLoader(cinc, batch_size=128, shuffle=True, num_workers=4)

    print("\nOzellikler cikariliyor...")
    src_f = extract_feats(model, src_loader, mode, args.model)
    tgt_f = extract_feats(model, tgt_loader, mode, args.model)
    print(f"  CirCor ozellik: {src_f.shape} | CinC ozellik: {tgt_f.shape}")

    acc, err, a_dist = a_distance(src_f, tgt_f)
    print(f"\nDomain-classifier (CirCor vs CinC ayrimi):")
    print(f"  dogruluk = {acc:.3f} | hata = {err:.3f}")
    print(f"  A-distance = {a_dist:.3f}  (0=hizali, 2=tam ayrisik)")

    print("\n" + "="*70)
    if a_dist < 0.4:
        print("[SONUC] A-distance KUCUK -> ozellikler zaten HIZALI.")
        print("        DA yontemleri (CORAL/MMD/DANN) ETKISIZ kalir.")
        print("        Sorun hizalama DEGIL; ogrenilen ozellikler yetiskin")
        print("        patolojisini ayirt etmiyor. DA pesinde kosmak bosa.")
        print("        -> Durust 'cross-dataset cokusu' makalesine gec.")
    elif a_dist < 1.0:
        print("[SONUC] A-distance ORTA -> kismi ayrisma var.")
        print("        DA kismi fayda saglayabilir ama dramatik degil")
        print("        (CORAL'da gordugumuz gibi). Sinirli getiri beklenir.")
    else:
        print("[SONUC] A-distance BUYUK -> ozellikler AYRISIK.")
        print("        DA yontemleri (MMD/DANN) GERCEK fayda saglayabilir.")
        print("        -> MMD veya DANN denemeye deger.")
    print("="*70)


if __name__ == "__main__":
    main()
