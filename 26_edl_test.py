#!/usr/bin/env python3
"""
26_edl_test.py
EDL'i IZOLE test et: CNN_BiLSTM gövdesi + evidential loss ile egitim.
Calisiyor mu, kararli mi, OOD-duyarli mi? Kiyasa katmadan once dogrula.

Kontroller:
  1) EDL kayipli egitim KARARLI mi (cokmeden ogreniyor mu)?
  2) Ic performans (W.Acc) makul mu (~0.75-0.80)?
  3) OOD DUYARLILIK: CinC (dis) belirsizligi > CirCor (ic) belirsizligi mi?
     (EDL'in asil vaadi: tanimadigi veride "emin degilim" demeli)
  4) Unknown hizalama: CirCor Unknown sinifi en yuksek belirsizlik mi?

CNN_BiLSTM head'i (Linear) oldugu gibi kullanilir; cikti logit -> evidence.

Kullanim:
  python 26_edl_test.py --fold 0 --epochs 30
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ds_mod = import_module("05_dataset")
me_mod = import_module("06_metrics")
hy_mod = import_module("09_model_hybrid")
bl_mod = import_module("08_train_stable")
ext_mod = import_module("16_external_val")

PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "uncert"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3


def make_smoothed_sampler(ds, beta=0.5):
    from torch.utils.data import WeightedRandomSampler
    labels = np.array(ds.seg_labels)
    c = np.bincount(labels, minlength=N_CLASSES).astype(float)
    w = (c.sum() / (N_CLASSES * np.maximum(c, 1))) ** beta
    return WeightedRandomSampler(torch.DoubleTensor(w[labels]), len(labels), replacement=True)


@torch.no_grad()
def eval_uncertainty(model, loader):
    """EDL: olasilik + belirsizlik (u=K/S) cikar. Segment-seviye."""
    model.eval()
    probs, us, labels = [], [], []
    for b in loader:
        x = b["raw"].to(DEVICE)
        logits = model(x)
        p, u = hy_mod.edl_predict(logits)
        probs.append(p.cpu().numpy()); us.append(u.cpu().numpy())
        labels.append(b["label"].numpy())
    return np.concatenate(probs), np.concatenate(us), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--anneal", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"

    print("="*70)
    print(f"EDL IZOLE TEST | CNN_BiLSTM + evidential loss | fold={args.fold}")
    print("="*70)

    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="train")
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="val")
    sampler = make_smoothed_sampler(tr)
    train_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)

    # CNN_BiLSTM (mevcut, head Linear -> logit; EDL yorumu cikarimda)
    model = bl_mod.CNN_BiLSTM(n_classes=N_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print("\n[EGITIM] evidential loss (Bayes risk + KL annealing)")
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n, nan_ct = 0.0, 0, 0
        for b in train_loader:
            x = b["raw"].to(DEVICE); y = b["label"].to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = hy_mod.edl_loss(logits, y, epoch=ep, anneal=args.anneal)
            if not torch.isfinite(loss):
                nan_ct += 1; continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            # ic degerlendirme
            p, u, lab = eval_uncertainty(model, val_loader)
            pred = p.argmax(1)
            wacc = me_mod.weighted_accuracy(lab, pred)
            print(f"  ep{ep:2d}: loss={tot/max(n,1):.4f} | ic W.Acc={wacc:.4f} | "
                  f"ort.belirsizlik={u.mean():.3f}" + (f" | nan_batch={nan_ct}" if nan_ct else ""))

    # === KONTROL 1-2: ic performans + kararlilik ===
    p_in, u_in, lab_in = eval_uncertainty(model, val_loader)
    pred_in = p_in.argmax(1)
    wacc = me_mod.weighted_accuracy(lab_in, pred_in)
    print(f"\n[KONTROL 1-2] Ic W.Acc={wacc:.4f} (makul: 0.70-0.80) | "
          f"egitim {'KARARLI' if wacc > 0.65 else 'COKTU(?)'}")

    # === KONTROL 3: OOD duyarlilik (CinC belirsizligi > CirCor mu?) ===
    cinc = ext_mod.CinCDataset(cinc_manifest, root)
    cinc_loader = DataLoader(cinc, batch_size=128, shuffle=False, num_workers=args.num_workers)
    _, u_ood, _ = eval_uncertainty(model, cinc_loader)
    print(f"\n[KONTROL 3] OOD DUYARLILIK (EDL'in asil vaadi):")
    print(f"  CirCor (ic)  ort. belirsizlik: {u_in.mean():.4f}")
    print(f"  CinC   (dis) ort. belirsizlik: {u_ood.mean():.4f}")
    if u_ood.mean() > u_in.mean():
        print(f"  >> [BASARILI] dis veri daha belirsiz (fark +{u_ood.mean()-u_in.mean():.4f})")
        print(f"     model tanimadigi popülasyonda 'emin degilim' diyor — OOD-duyarli!")
    else:
        print(f"  >> [ZAYIF] dis veri daha belirsiz DEGIL — EDL OOD yakalamiyor")

    # === KONTROL 4: Unknown hizalama ===
    print(f"\n[KONTROL 4] Unknown-belirsizlik hizalamasi (CirCor ic):")
    for cls, name in [(0, "Present"), (1, "Unknown"), (2, "Absent")]:
        m = lab_in == cls
        if m.sum() > 0:
            print(f"  {name:8s}: ort. belirsizlik = {u_in[m].mean():.4f} (n={m.sum()})")
    u_unknown = u_in[lab_in == 1].mean() if (lab_in == 1).sum() > 0 else 0
    u_others = u_in[lab_in != 1].mean()
    if u_unknown > u_others:
        print(f"  >> [BASARILI] Unknown en yuksek belirsizlik (klinik hizalama)")
    else:
        print(f"  >> Unknown belirsizligi digerlerinden yuksek degil")

    print("\n" + "="*70)
    print("EDL test tamamlandi. Yukaridaki 4 kontrol EDL'in kiyasa hazir")
    print("olup olmadigini gosterir.")
    print("="*70)


if __name__ == "__main__":
    main()
