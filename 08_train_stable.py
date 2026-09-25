#!/usr/bin/env python3
"""
08_train_stable.py
PCG Q1 Manuscript — Adim A+: STABILIZE EDILMIS egitim iskeleti.

07'ye gore degisiklikler (egitim kararsizligini gidermek icin):
  1) Yumusatilmis sampler: weight^beta (varsayilan beta=0.5, karekok).
     -> beta=1.0 her batch'te Unknown'i %33 temsil ediyordu (savrulma kaynagi).
        beta=0.5 ile Unknown ~%14: azinligi ogrenir ama saplanmaz.
  2) Label smoothing (0.05): belirsiz Unknown sinifinda asiri-guveni kirar.
  3) Birlesik secim kriteri: 0.5*W.Acc + 0.5*macro-F1.
     -> tek sinifa saplanip yuksek W.Acc veren epoch'lar elenir.
  4) EMA-tabanli erken durdurma: son 3 epoch hareketli ortalamasi.
     -> sans epoch'una kilitlenmeyi onler.
  5) LR warmup (ilk 2 epoch lineer) + cosine.

Bu iskelet Adim B'de ana hibrit modele AYNEN kullanilacak.

Kullanim:
  python 08_train_stable.py --model cnn2d --fold 0 --epochs 30 --sampler_beta 0.5
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from importlib import import_module
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ds_mod = import_module("05_dataset")
me_mod = import_module("06_metrics")

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
CKPT_DIR   = PROJECT_ROOT / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3


# ---- baseline modeller (07 ile ayni) ----
class CNN1D(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        def block(ci, co, k=7, s=1, p=3):
            return nn.Sequential(nn.Conv1d(ci,co,k,s,p), nn.BatchNorm1d(co),
                                 nn.ReLU(True), nn.MaxPool1d(4))
        self.features = nn.Sequential(block(1,16),block(16,32),block(32,64),
                                      block(64,128),block(128,128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128,n_classes))
    def forward(self,x): return self.head(self.features(x))

class CNN2D(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        def block(ci,co):
            return nn.Sequential(nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co),
                                 nn.ReLU(True), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1,16),block(16,32),block(32,64),block(64,128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128,n_classes))
    def forward(self,x): return self.head(self.features(x))


class CNN_BiLSTM(nn.Module):
    """CNN on-yuz + BiLSTM baseline (ham sinyal icin).
    Mamba kolunun KLASIK recurrent muadili — adil kiyas icin rekabetci kurulur.
    CNN yerel ozellik cikarir, BiLSTM zamansal bagimliligi modeller.
    Mamba kolu (09) ile ayni mantik: stem ile diziyi kisalt, sonra ardisik model."""
    def __init__(self, n_classes=N_CLASSES, hidden=128, n_layers=2):
        super().__init__()
        # Mamba stem'i ile ayni kuculme orani (10000 -> ~156): adil kiyas
        def cb(ci, co, k=7, s=4, p=3):
            return nn.Sequential(nn.Conv1d(ci, co, k, s, p),
                                 nn.GroupNorm(min(8, co), co), nn.GELU())
        self.stem = nn.Sequential(cb(1, 32), cb(32, 64), cb(64, 128))  # (B,128,~156)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden, num_layers=n_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):           # x: (B,1,10000)
        h = self.stem(x)            # (B,128,T)
        h = h.transpose(1, 2)       # (B,T,128) — LSTM (B,L,D) bekler
        out, _ = self.lstm(h)       # (B,T,2*hidden)
        return self.head(out.mean(dim=1))   # zaman havuzu -> head

MODELS = {"cnn1d": (CNN1D,"raw"), "cnn2d": (CNN2D,"mel"),
          "cnn_bilstm": (CNN_BiLSTM, "raw")}


# ---- yumusatilmis sampler ----
def make_smoothed_sampler(ds, beta=0.5):
    """Sinif agirliklarini ^beta ile yumusatip ornek agirligi uretir."""
    counts = np.bincount(ds.seg_labels, minlength=N_CLASSES).astype(float)
    total = counts.sum()
    cw = (total / (N_CLASSES * np.maximum(counts,1))) ** beta
    sw = cw[ds.seg_labels]
    return WeightedRandomSampler(torch.DoubleTensor(sw),
                                 num_samples=len(ds), replacement=True), cw


# ---- warmup + cosine LR ----
def lr_lambda_factory(warmup_epochs, total_epochs):
    def f(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        # cosine
        prog = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    return f


def run_epoch(model, loader, key, crit, opt=None):
    train = opt is not None
    model.train() if train else model.eval()
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for b in loader:
            x = b[key].to(DEVICE, non_blocking=True)
            y = b["label"].to(DEVICE, non_blocking=True)
            if train: opt.zero_grad()
            logits = model(x); loss = crit(logits, y)
            if train: loss.backward(); opt.step()
            tot += loss.item()*len(y); n += len(y)
    return tot/max(n,1)


@torch.no_grad()
def eval_patient(model, loader, key, split_csv, agg="mean"):
    model.eval(); probs, pids = [], []
    for b in loader:
        x = b[key].to(DEVICE, non_blocking=True)
        probs.append(torch.softmax(model(x),1).cpu().numpy())
        pids.extend(b["patient_id"])
    seg = np.concatenate(probs,0)
    pp, ids = ds_mod.aggregate_to_patient(seg, pids, agg)
    truth = ds_mod.patient_true_labels(split_csv)
    yt = np.array([truth[p] for p in ids])
    return me_mod.all_metrics(yt, pp.argmax(1)), me_mod.confusion_matrix(yt, pp.argmax(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="cnn2d")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sampler_beta", type=float, default=0.5)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--ema_window", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.root); cv_csv = root/"splits"/"circor2022_5fold_cv.csv"
    logs = []
    def log(m=""): print(m); logs.append(str(m))

    log("="*70)
    log(f"STABIL egitim: {args.model} | fold {args.fold} | device={DEVICE}")
    log(f"sampler_beta={args.sampler_beta} | label_smooth={args.label_smooth} | "
        f"sel=0.5*W.Acc+0.5*F1 | EMA={args.ema_window}")
    log("="*70)

    ModelCls, key = MODELS[args.model]
    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="train")
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="val")
    log(f"Train {len(tr)} seg / {tr.records['patient_id'].nunique()} hasta | "
        f"Val {len(va)} seg / {va.records['patient_id'].nunique()} hasta")

    sampler, smooth_cw = make_smoothed_sampler(tr, beta=args.sampler_beta)
    log(f"Yumusatilmis sampler agirliklari (P/U/A): {[round(x,3) for x in smooth_cw]}")
    train_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)

    model = ModelCls().to(DEVICE)
    # loss agirligi: yumusatilmis (sampler ile uyumlu, cifte-dengeleme olmasin diye hafif)
    loss_w = torch.tensor((smooth_cw/smooth_cw.mean()), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup, args.epochs))

    def sel_score(m):  # birlesik secim
        return 0.5*m["weighted_accuracy"] + 0.5*m["macro_f1"]

    best_sel, best_ep = -1.0, -1
    ema = deque(maxlen=args.ema_window); best_ema, bad = -1.0, 0
    ckpt = CKPT_DIR/f"stable_{args.model}_fold{args.fold}.pth"

    log("\nEp | tr_loss | W.Acc | F1 | UAR | sel | EMA | (P/U/A) | lr")
    log("-"*70)
    for ep in range(1, args.epochs+1):
        t0=time.time()
        trl = run_epoch(model, train_loader, key, crit, opt)
        sched.step()
        m,_ = eval_patient(model, val_loader, key, cv_csv)
        s = sel_score(m); ema.append(s); ema_v = float(np.mean(ema))
        lr_now = opt.param_groups[0]["lr"]
        log(f"{ep:2d} | {trl:.4f} | {m['weighted_accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['uar']:.4f} | {s:.4f} | {ema_v:.4f} | "
            f"{m['recall_present']:.2f}/{m['recall_unknown']:.2f}/{m['recall_absent']:.2f} | "
            f"{lr_now:.1e} ({time.time()-t0:.0f}s)")

        if s > best_sel:
            best_sel, best_ep = s, ep
            torch.save({"model":model.state_dict(),"epoch":ep,"metrics":m,
                        "args":vars(args)}, ckpt)
        # EMA tabanli erken durdurma
        if ema_v > best_ema + 1e-4:
            best_ema = ema_v; bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                log(f"\nErken durdurma (EMA {args.patience} epoch iyilesmedi).")
                break

    log("\n"+"="*70)
    log(f"En iyi epoch (sel kriteri): {best_ep} | sel={best_sel:.4f}")
    ck = torch.load(ckpt, map_location=DEVICE); model.load_state_dict(ck["model"])
    m, cm = eval_patient(model, val_loader, key, cv_csv)
    log("Final (val, en iyi model):")
    log("   "+me_mod.format_metrics(m))
    log("\nConfusion [gercek x tahmin] (P/U/A):")
    log("        pred_P  pred_U  pred_A")
    for i,cn in enumerate(me_mod.CLASSES):
        log(f"   {cn:>7s} {cm[i,0]:6d} {cm[i,1]:7d} {cm[i,2]:7d}")

    rep = REPORT_DIR/f"stable_{args.model}_fold{args.fold}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}\nCheckpoint: {ckpt}\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
