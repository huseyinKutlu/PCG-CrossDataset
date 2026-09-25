#!/usr/bin/env python3
"""
10_train_hybrid.py
PCG Q1 Manuscript — Hibrit modeli STABIL iskeletle egit (head karsilastirmasi).

08'in TUM stabilizasyonunu korur (yumusatilmis sampler, label smoothing,
birlesik secim, EMA erken durdurma, warmup+cosine). Tek fark: model artik
HybridPCG ve iki girdi alir (raw + mel).

Kullanim:
  python 10_train_hybrid.py --head mlp --fold 0 --epochs 30
  python 10_train_hybrid.py --head kan --fold 0 --epochs 30

Cikti:
  reports/hybrid_<head>_fold<k>.txt
  checkpoints/hybrid_<head>_fold<k>.pth
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
hy_mod = import_module("09_model_hybrid")

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
CKPT_DIR   = PROJECT_ROOT / "checkpoints"; CKPT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3


def make_smoothed_sampler(ds, beta=0.5):
    counts = np.bincount(ds.seg_labels, minlength=N_CLASSES).astype(float)
    total = counts.sum()
    cw = (total / (N_CLASSES * np.maximum(counts, 1))) ** beta
    sw = cw[ds.seg_labels]
    return WeightedRandomSampler(torch.DoubleTensor(sw), num_samples=len(ds),
                                 replacement=True), cw


def lr_lambda_factory(warmup, total):
    def f(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        prog = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    return f


def run_epoch(model, loader, crit, opt=None, use_bf16=False, clip=1.0):
    """Hibrit: hem raw hem mel besle.
    Korumalar: bf16 autocast (float16 DEGIL — taşma yapmaz), gradient clipping,
    nan-nobetcisi (zehirli batch atlanir)."""
    train = opt is not None
    model.train() if train else model.eval()
    tot, n, n_skip = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for b in loader:
            raw = b["raw"].to(DEVICE, non_blocking=True)
            mel = b["mel"].to(DEVICE, non_blocking=True)
            y = b["label"].to(DEVICE, non_blocking=True)
            if train:
                opt.zero_grad()
                if use_bf16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(raw, mel); loss = crit(logits, y)
                else:
                    logits = model(raw, mel); loss = crit(logits, y)
                # nan-nobetcisi: zehirli batch egitimi bozmasin
                if not torch.isfinite(loss):
                    n_skip += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            else:
                logits = model(raw, mel); loss = crit(logits, y)
                if not torch.isfinite(loss):
                    continue
            tot += loss.item() * len(y); n += len(y)
    if n_skip:
        print(f"   [uyari] {n_skip} batch nan/inf nedeniyle atlandi")
    return tot / max(n, 1)


@torch.no_grad()
def eval_patient(model, loader, split_csv, agg="mean"):
    model.eval(); probs, pids = [], []
    for b in loader:
        raw = b["raw"].to(DEVICE, non_blocking=True)
        mel = b["mel"].to(DEVICE, non_blocking=True)
        probs.append(torch.softmax(model(raw, mel), 1).float().cpu().numpy())
        pids.extend(b["patient_id"])
    seg = np.concatenate(probs, 0)
    pp, ids = ds_mod.aggregate_to_patient(seg, pids, agg)
    truth = ds_mod.patient_true_labels(split_csv)
    yt = np.array([truth[p] for p in ids])
    return (me_mod.all_metrics(yt, pp.argmax(1)),
            me_mod.confusion_matrix(yt, pp.argmax(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=list(hy_mod.HEADS), default="mlp")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sampler_beta", type=float, default=0.5)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--ema_window", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--bf16", action="store_true",
                    help="bf16 mixed precision (float16 DEGIL — taşma yapmaz, hizli)")
    ap.add_argument("--clip", type=float, default=0.5, help="gradient clipping max-norm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    root = Path(args.root); cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    logs = []
    def log(m=""): print(m); logs.append(str(m))

    log("="*70)
    log(f"HIBRIT egitim: head={args.head} | fold {args.fold} | device={DEVICE} | seed={args.seed}")
    log(f"sampler_beta={args.sampler_beta} | label_smooth={args.label_smooth} | "
        f"bf16={args.bf16} | clip={args.clip} | d_model={args.d_model}")
    log("="*70)

    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="train")
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="val")
    log(f"Train {len(tr)} seg / {tr.records['patient_id'].nunique()} hasta | "
        f"Val {len(va)} seg / {va.records['patient_id'].nunique()} hasta")

    sampler, cw = make_smoothed_sampler(tr, beta=args.sampler_beta)
    log(f"Yumusatilmis sampler (P/U/A): {[round(x,3) for x in cw]}")
    train_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)

    model = hy_mod.HybridPCG(head=args.head, d_model=args.d_model).to(DEVICE)
    log(f"Model parametre: {sum(p.numel() for p in model.parameters()):,}")
    loss_w = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup, args.epochs))

    def sel_score(m): return 0.5*m["weighted_accuracy"] + 0.5*m["macro_f1"]

    best_sel, best_ep = -1.0, -1
    ema = deque(maxlen=args.ema_window); best_ema, bad = -1.0, 0
    ckpt = CKPT_DIR / f"hybrid_{args.head}_fold{args.fold}.pth"

    log("\nEp | tr_loss | W.Acc | F1 | UAR | sel | EMA | (P/U/A) | lr")
    log("-"*70)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        trl = run_epoch(model, train_loader, crit, opt,
                        use_bf16=args.bf16, clip=args.clip)
        sched.step()
        m, _ = eval_patient(model, val_loader, cv_csv)
        s = sel_score(m); ema.append(s); ema_v = float(np.mean(ema))
        lr_now = opt.param_groups[0]["lr"]
        log(f"{ep:2d} | {trl:.4f} | {m['weighted_accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['uar']:.4f} | {s:.4f} | {ema_v:.4f} | "
            f"{m['recall_present']:.2f}/{m['recall_unknown']:.2f}/{m['recall_absent']:.2f} | "
            f"{lr_now:.1e} ({time.time()-t0:.0f}s)")

        if s > best_sel:
            best_sel, best_ep = s, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                        "args": vars(args)}, ckpt)
        if ema_v > best_ema + 1e-4:
            best_ema = ema_v; bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                log(f"\nErken durdurma (EMA {args.patience} epoch iyilesmedi)."); break

    log("\n" + "="*70)
    log(f"En iyi epoch: {best_ep} | sel={best_sel:.4f}")
    ck = torch.load(ckpt, map_location=DEVICE); model.load_state_dict(ck["model"])
    m, cm = eval_patient(model, val_loader, cv_csv)
    log("Final (val, en iyi model):")
    log("   " + me_mod.format_metrics(m))
    log("\nConfusion [gercek x tahmin] (P/U/A):")
    log("        pred_P  pred_U  pred_A")
    for i, cn in enumerate(me_mod.CLASSES):
        log(f"   {cn:>7s} {cm[i,0]:6d} {cm[i,1]:7d} {cm[i,2]:7d}")

    rep = REPORT_DIR / f"hybrid_{args.head}_fold{args.fold}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}\nCheckpoint: {ckpt}\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
