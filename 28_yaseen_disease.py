#!/usr/bin/env python3
"""
28_yaseen_disease.py
Yaseen HASTALIK-SPESIFIK transfer analizi (ozgun klinik katki).

Soru: pediatrik murmur'da egitilen model, yetiskin kapak hastaliklarindan
hangilerini taniyabiliyor? (AS / MS / MR / MVP — N referans)

Yontem:
  - cnn_bilstm (tek klinik-dengeli model) 5-fold ensemble
  - Yaseen'de her record icin abnormal-olasiligi
  - record_id'den hastalik (class5) cikar: New_{SINIF}_{NNN}
  - her hastalik icin: ortalama abnormal-skor + sensitivity (esik 0.5)
  - N (normal) icin: specificity
  - AUROC: her hastalik vs N (ikili ayirt edilebilirlik)

Cikti: hastalik-spesifik tablo (makaleye ozgun Tablo/Sekil).

Kullanim:
  python 28_yaseen_disease.py --model cnn_bilstm
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ext_mod = import_module("16_external_val")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "binary"


@torch.no_grad()
def predict_yaseen(model, loader, mode):
    """record bazinda abnormal-olasiligi + class5 dondur."""
    model.eval()
    seg_pa, ids, c5s = [], [], []
    for b in loader:
        logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE)) if mode == "dual" \
                 else model(b[mode].to(DEVICE))
        prob = torch.softmax(logits, 1).float().cpu().numpy()
        seg_pa.append(prob[:, 1])
        ids.extend(b["record_id"]); c5s.extend(b["class5"])
    seg_pa = np.concatenate(seg_pa); ids = np.array(ids); c5s = np.array(c5s)
    uniq = np.unique(ids)
    rec_p = np.array([seg_pa[ids == u].mean() for u in uniq])
    rec_c5 = np.array([c5s[ids == u][0] for u in uniq])
    return rec_p, rec_c5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_bilstm")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    manifest = root / "manifests" / "yaseen_processed.csv"

    print("="*70)
    print(f"YASEEN HASTALIK-SPESIFIK TRANSFER | {args.model} (5-fold ensemble)")
    print("Soru: pediatrik-egitimli model hangi yetiskin kapak hastaligini taniyor?")
    print("="*70)

    yas = ext_mod.YaseenDataset(manifest, root, norm_adapt="none")
    loader = DataLoader(yas, batch_size=128, shuffle=False, num_workers=args.num_workers)

    # 5-fold ensemble: her fold'un abnormal-olasiligini ortala
    all_p, rec_c5 = None, None
    n_loaded = 0
    for fold in range(N_FOLDS):
        ckpt = CKPT_DIR / f"{args.model}_fold{fold}.pth"
        if not ckpt.exists():
            print(f"  [atla] checkpoint yok: {ckpt.name}"); continue
        built = ext_mod.build_binary_model(args.model)
        model, mode = built[0].to(DEVICE), built[1]
        sd = torch.load(ckpt, map_location=DEVICE)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]   # checkpoint sozluk: {model, epoch, auroc}
        model.load_state_dict(sd)
        p, c5 = predict_yaseen(model, loader, mode)
        all_p = p if all_p is None else all_p + p
        rec_c5 = c5; n_loaded += 1
    if n_loaded == 0:
        print("  HATA: hic checkpoint bulunamadi."); return
    all_p /= n_loaded
    print(f"  {n_loaded}/{N_FOLDS} fold ensemble\n")

    # N (normal) referans skorlari
    p_normal = all_p[rec_c5 == "N"]
    spec = float((p_normal < 0.5).mean())   # normal'i normal bulma

    print(f"{'Hastalik':<8} {'n':>4} {'ort.abn-skor':>13} {'sensitivity':>12} {'AUROC-vs-N':>11}")
    print("-"*52)
    # N satiri (referans)
    print(f"{'N (norm)':<8} {len(p_normal):>4} {p_normal.mean():>13.3f} "
          f"{'(spec='+format(spec,'.3f')+')':>12} {'-':>11}")

    rows = []
    for dis in ["AS", "MS", "MR", "MVP"]:
        m = rec_c5 == dis
        if m.sum() == 0:
            continue
        p_dis = all_p[m]
        sens = float((p_dis >= 0.5).mean())   # hastaligi abnormal bulma
        # AUROC: bu hastalik (1) vs N (0)
        y = np.concatenate([np.ones(m.sum()), np.zeros((rec_c5 == "N").sum())])
        s = np.concatenate([p_dis, p_normal])
        try:
            auroc = roc_auc_score(y, s)
        except Exception:
            auroc = float("nan")
        rows.append((dis, m.sum(), p_dis.mean(), sens, auroc))
        print(f"{dis:<8} {m.sum():>4} {p_dis.mean():>13.3f} {sens:>12.3f} {auroc:>11.3f}")

    print("\n" + "="*70)
    print("YORUM:")
    # en iyi / en kotu taninan
    rows_sorted = sorted(rows, key=lambda r: r[3], reverse=True)
    best, worst = rows_sorted[0], rows_sorted[-1]
    print(f"  En iyi taninan : {best[0]} (sensitivity={best[3]:.3f}, AUROC={best[4]:.3f})")
    print(f"  En kotu taninan: {worst[0]} (sensitivity={worst[3]:.3f}, AUROC={worst[4]:.3f})")
    print(f"  Specificity (N): {spec:.3f}")
    print("\n  Klinik icgoru: model belirgin ufurumlu hastaliklari (yuksek sens)")
    print("  daha iyi tanir; ince/atipik olanlari kacirir. Bu, pediatrik->yetiskin")
    print("  transferin hastalik-bagimli oldugunu gosterir (ozgun bulgu).")
    print("="*70)


if __name__ == "__main__":
    main()
