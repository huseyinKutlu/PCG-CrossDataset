#!/usr/bin/env python3
"""
19_coral_da.py
PCG Q1 Manuscript — Deep CORAL unsupervised domain adaptation.

AMAC: Naif zero-shot transfer cokuyordu (hybrid_kan dis 0.57). CORAL,
kaynak (CirCor) ve hedef (CinC) ozellik dagilimlarinin KOVARYANSLARINI
hizalayarak modeli domain-degismez ozellik ogrenmeye zorlar.

YONTEM (Sun & Saenko 2016):
  L = L_cls(CirCor, etiketli) + lambda * L_coral(CirCor_feat, CinC_feat)
  CinC ETIKETSIZ kullanilir (sadece ozellik dagilimi). -> unsupervised DA.

DURUSTLUK NOTU: Bu artik saf "zero-shot" degil — model CinC ozelliklerini
(etiketsiz) gorur. Makalede "unsupervised domain adaptation" diye raporlanir.
Yontem ONCEDEN secildi (sonuca gore degil) -> test-set overfitting yok.

Kullanim:
  python 19_coral_da.py --model hybrid_kan --lambda_coral 1.0 --epochs 30
  python 19_coral_da.py --model cnn_bilstm --lambda_coral 1.0 --epochs 30
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from importlib import import_module
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ds_mod = import_module("05_dataset")
me_mod = import_module("06_metrics")
hy_mod = import_module("09_model_hybrid")
bl_mod = import_module("08_train_stable")
ext_mod = import_module("16_external_val")   # CinCDataset, build_binary_model

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "coral"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CIRCOR_TO_BINARY = {0: 1, 1: 1, 2: 0}


# ---- CORAL kaybi ----
def coral_loss(source, target):
    """source, target: (n, d) ozellik matrisleri. Kovaryans hizalama kaybi."""
    d = source.shape[1]
    def cov(f):
        f = f - f.mean(0, keepdim=True)
        return (f.t() @ f) / (f.shape[0] - 1)
    cs, ct = cov(source), cov(target)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


# ---- ozellik cikaran sarmalayici (head oncesi temsil) ----
def extract_features(model, raw, mel, mode):
    """Modelin head-oncesi fuzyon ozelligini dondur (CORAL bunun uzerinde calisir)."""
    if mode == "dual":
        # HybridPCG/HybridPCG2: mamba/bilstm + spec -> fuse
        if hasattr(model, "mamba") and hasattr(model, "spec") and hasattr(model, "fuse"):
            f1 = model.mamba(raw); f2 = model.spec(mel)
            return model.fuse(torch.cat([f1, f2], 1))
        if hasattr(model, "bilstm") and hasattr(model, "spec") and hasattr(model, "fuse"):
            f1 = model.bilstm(raw); f2 = model.spec(mel)
            return model.fuse(torch.cat([f1, f2], 1))
    # cnn_bilstm vb: head oncesi -> features
    # baseline modeller icin: son linear oncesi
    return None  # asagida model-tipine gore ele alinir


def lr_lambda_factory(warmup, total):
    def f(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        prog = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    return f


def to_binary(y3):
    out = y3.clone()
    for k, v in CIRCOR_TO_BINARY.items():
        out[y3 == k] = v
    return out


def make_binary_sampler(base_ds, beta=0.5):
    bl = np.array([CIRCOR_TO_BINARY[int(l)] for l in base_ds.seg_labels])
    c = np.bincount(bl, minlength=2).astype(float)
    cw = (c.sum() / (2 * np.maximum(c, 1))) ** beta
    return WeightedRandomSampler(torch.DoubleTensor(cw[bl]), num_samples=len(bl),
                                 replacement=True), cw


class FeatureHybrid(nn.Module):
    """HybridPCG/PCG2'yi sarip head-oncesi ozelligi de dondurebilen wrapper."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def features(self, raw, mel):
        b = self.base
        if hasattr(b, "mamba"):
            f1 = b.mamba(raw)
        elif hasattr(b, "bilstm"):
            f1 = b.bilstm(raw)
        f2 = b.spec(mel)
        return b.fuse(torch.cat([f1, f2], 1))
    def forward(self, raw, mel):
        z = self.features(raw, mel)
        return self.base.head(z)


def get_features_and_logits(model, raw, mel, mode, mdl_name):
    """Model tipine gore head-oncesi ozellik + logits dondur."""
    if mdl_name in ("hybrid_kan", "hybrid_mlp", "hybrid2_kan", "hybrid2_mlp"):
        b = model
        f1 = b.mamba(raw) if hasattr(b, "mamba") else b.bilstm(raw)
        f2 = b.spec(mel)
        z = b.fuse(torch.cat([f1, f2], 1))
        return z, b.head(z)
    if mdl_name == "cnn_bilstm":
        # CNN_BiLSTM: stem -> lstm -> mean -> head; ozellik = head oncesi
        b = model
        h = b.stem(raw); h = h.transpose(1, 2)
        out, _ = b.lstm(h); feat = out.mean(1)
        # head ilk katman(lar): head[0]=Dropout, head[1]=Linear
        logits = b.head(feat)
        return feat, logits
    # diger baseline'lar: genel — forward + son-oncesi yok, basit logits
    logits = model(raw, mel) if mode == "dual" else model(raw if mode == "raw" else mel)
    return None, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid_kan",
                    choices=["hybrid_kan", "hybrid_mlp", "cnn_bilstm"])
    ap.add_argument("--lambda_coral", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sampler_beta", type=float, default=0.5)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
    logs = []
    def log(m=""): print(m); logs.append(str(m))

    log("="*70)
    log(f"DEEP CORAL domain adaptation | model={args.model} | lambda={args.lambda_coral}")
    log("Kaynak: CirCor (etiketli) | Hedef: CinC (ETIKETSIZ ozellik hizalama)")
    log("="*70)

    # CinC hedef (etiketsiz) loader — ozellik hizalama icin
    cinc = ext_mod.CinCDataset(cinc_manifest, root)
    cinc_loader = DataLoader(cinc, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)

    internal_au, external_au = [], []
    cinc_probs_folds = []; cinc_y = None

    for fold in range(N_FOLDS):
        t0 = time.time()
        tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="train")
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        sampler, cw = make_binary_sampler(tr, args.sampler_beta)
        src_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)

        model, mode = ext_mod.build_binary_model(args.model)
        model = model.to(DEVICE)
        loss_w = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup, args.epochs))

        best_auroc, bad = -1.0, 0
        ema = deque(maxlen=3); best_ema = -1.0
        ckpt = CKPT_DIR / f"{args.model}_coral_fold{fold}.pth"

        for ep in range(1, args.epochs + 1):
            model.train()
            cinc_iter = iter(cinc_loader)
            tot_cls, tot_coral, n = 0.0, 0.0, 0
            for b in src_loader:
                raw_s = b["raw"].to(DEVICE); mel_s = b["mel"].to(DEVICE)
                y = to_binary(b["label"]).to(DEVICE)
                # hedef batch (etiketsiz)
                try: bt = next(cinc_iter)
                except StopIteration:
                    cinc_iter = iter(cinc_loader); bt = next(cinc_iter)
                raw_t = bt["raw"].to(DEVICE); mel_t = bt["mel"].to(DEVICE)

                opt.zero_grad()
                feat_s, logits_s = get_features_and_logits(model, raw_s, mel_s, mode, args.model)
                feat_t, _ = get_features_and_logits(model, raw_t, mel_t, mode, args.model)
                loss_cls = crit(logits_s, y)
                loss_coral = coral_loss(feat_s, feat_t) if feat_s is not None else torch.tensor(0.0, device=DEVICE)
                loss = loss_cls + args.lambda_coral * loss_coral
                if not torch.isfinite(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                tot_cls += loss_cls.item() * len(y)
                tot_coral += float(loss_coral.detach()) * len(y); n += len(y)
            sched.step()

            # ic-val AUROC
            rp, ry, _ = ext_mod.predict_grouped(model, val_loader, mode, "patient_id", circor=True)
            bm = me_mod.binary_metrics(ry, rp)
            ema.append(bm["auroc"]); ema_v = float(np.mean(ema))
            if bm["auroc"] > best_auroc:
                best_auroc = bm["auroc"]
                torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
            if ema_v > best_ema + 1e-4: best_ema = ema_v; bad = 0
            else:
                bad += 1
                if bad >= args.patience: break

        # en iyi modelle ic + dis
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
        rp, ry, _ = ext_mod.predict_grouped(model, val_loader, mode, "patient_id", circor=True)
        in_au = me_mod.binary_metrics(ry, rp)["auroc"]; internal_au.append(in_au)
        cinc_eval = DataLoader(cinc, batch_size=128, shuffle=False, num_workers=args.num_workers)
        cp, cy, _ = ext_mod.predict_grouped(model, cinc_eval, mode, "record_id", circor=False)
        cinc_probs_folds.append(cp); cinc_y = cy
        ex_au = me_mod.binary_metrics(cy, cp)["auroc"]; external_au.append(ex_au)
        log(f"  fold {fold}: ic AUROC={in_au:.4f} | dis AUROC={ex_au:.4f} "
            f"(coral={tot_coral/max(n,1):.6f}, λ·coral={args.lambda_coral*tot_coral/max(n,1):.5f}) "
            f"({time.time()-t0:.0f}s)")

    ia, ea = np.array(internal_au), np.array(external_au)
    cinc_ens = np.mean(cinc_probs_folds, axis=0)
    ens_au = me_mod.binary_metrics(cinc_y, cinc_ens)["auroc"]
    log(f"\n  --- {args.model} + CORAL (lambda={args.lambda_coral}) ---")
    log(f"    Ic  AUROC: {ia.mean():.4f} ± {ia.std():.4f}")
    log(f"    Dis AUROC: {ea.mean():.4f} ± {ea.std():.4f}")
    log(f"    Dis (5-fold ens): {ens_au:.4f}")
    log(f"\n  KIYAS (CORAL'siz dis AUROC referans):")
    log(f"    hybrid_kan naif=0.568 | cnn_bilstm naif=0.700")
    log(f"    -> CORAL ile {args.model}: {ens_au:.4f}")

    rep = REPORT_DIR / f"coral_{args.model}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}")
    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
