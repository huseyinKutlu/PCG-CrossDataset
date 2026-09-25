#!/usr/bin/env python3
"""
14_cv5_single.py
PCG Q1 Manuscript — 5-fold tek-model degerlendirme (makale ana tablosu).

Tum modelleri AYNI 5 fold uzerinde kosar, fold-bazinda metrik toplar,
ortalama ± std raporlar. Hem hibrit (mlp/kan) hem baseline (cnn1d/cnn2d).

Bu, makalenin Results bolumunun OMURGASI:
  - hibrit gercekten baseline'i geciyor mu (adil 5-fold kiyas)
  - mlp vs kan hangisi daha iyi (gurultusuz karar)
  - fold-lar arasi varyans ne (stabilite)

Kullanim:
  # tek model, 5 fold
  python 14_cv5_single.py --model hybrid_kan --epochs 30
  python 14_cv5_single.py --model hybrid_mlp --epochs 30
  python 14_cv5_single.py --model cnn2d --epochs 25
  python 14_cv5_single.py --model cnn1d --epochs 25

  # hepsi sirayla
  python 14_cv5_single.py --model all --epochs 30

Cikti: reports/cv5_<model>.txt  (fold detay + ortalama±std)
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
bl_mod = import_module("08_train_stable")   # CNN1D, CNN2D baseline'lar

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "cv5"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3
N_FOLDS = 5


def make_sampler(ds, beta=0.5):
    c = np.bincount(ds.seg_labels, minlength=N_CLASSES).astype(float)
    cw = (c.sum() / (N_CLASSES * np.maximum(c, 1))) ** beta
    return WeightedRandomSampler(torch.DoubleTensor(cw[ds.seg_labels]),
                                 num_samples=len(ds), replacement=True), cw


def lr_lambda_factory(warmup, total):
    def f(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        prog = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    return f


def build_model(name):
    """model adindan (model, iki_girdi_mi) dondur."""
    if name == "hybrid_kan":
        return hy_mod.HybridPCG(head="kan", d_model=128), "dual"
    if name == "hybrid_mlp":
        return hy_mod.HybridPCG(head="mlp", d_model=128), "dual"
    if name == "hybrid2_kan":
        return hy_mod.HybridPCG2(head="kan", d_model=128), "dual"
    if name == "hybrid2_mlp":
        return hy_mod.HybridPCG2(head="mlp", d_model=128), "dual"
    if name == "hybrid3_kan":
        return hy_mod.HybridPCG3(head="kan", d_model=128), "dual"
    if name == "hybrid3_mlp":
        return hy_mod.HybridPCG3(head="mlp", d_model=128), "dual"
    if name == "hybrid4_kan":
        return hy_mod.HybridPCG4(head="kan", d_model=128), "dual"
    if name == "hybrid4_mlp":
        return hy_mod.HybridPCG4(head="mlp", d_model=128), "dual"
    if name == "cnn2d":
        return bl_mod.CNN2D(), "mel"
    if name == "cnn1d":
        return bl_mod.CNN1D(), "raw"
    if name == "cnn_bilstm":
        return bl_mod.CNN_BiLSTM(), "raw"
    raise ValueError(name)


def run_epoch(model, loader, crit, opt, mode, clip=0.5):
    model.train(); tot, n, skip = 0.0, 0, 0
    for b in loader:
        y = b["label"].to(DEVICE, non_blocking=True)
        opt.zero_grad()
        if mode == "dual":
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE))
        else:
            logits = model(b[mode].to(DEVICE))
        loss = crit(logits, y)
        if not torch.isfinite(loss): skip += 1; continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        tot += loss.item() * len(y); n += len(y)
    if skip: print(f"     [uyari] {skip} batch atlandi")
    return tot / max(n, 1)


@torch.no_grad()
def eval_fold(model, loader, cv_csv, mode):
    model.eval(); probs, pids = [], []
    for b in loader:
        if mode == "dual":
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE))
        else:
            logits = model(b[mode].to(DEVICE))
        probs.append(torch.softmax(logits, 1).float().cpu().numpy())
        pids.extend(b["patient_id"])
    seg = np.concatenate(probs, 0)
    pp, ids = ds_mod.aggregate_to_patient(seg, pids, "mean")
    truth = ds_mod.patient_true_labels(cv_csv)
    yt = np.array([truth[p] for p in ids])
    m = me_mod.all_metrics(yt, pp.argmax(1))
    cm = me_mod.confusion_matrix(yt, pp.argmax(1))
    return m, cm, (np.array(ids), yt, pp)   # hasta tahminleri de don


def train_eval_one_fold(args, fold, cv_csv, root):
    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="train")
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
    sampler, cw = make_sampler(tr, args.sampler_beta)
    train_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)

    model, mode = build_model(args.model)
    model = model.to(DEVICE)
    loss_w = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup, args.epochs))

    def sel(m): return 0.5 * m["weighted_accuracy"] + 0.5 * m["macro_f1"]
    best_sel, best_m, best_cm, best_preds = -1.0, None, None, None
    ema = deque(maxlen=3); best_ema, bad = -1.0, 0
    for ep in range(1, args.epochs + 1):
        run_epoch(model, train_loader, crit, opt, mode, clip=args.clip)
        sched.step()
        m, cm, preds = eval_fold(model, val_loader, cv_csv, mode)
        s = sel(m); ema.append(s); ema_v = float(np.mean(ema))
        if s > best_sel: best_sel, best_m, best_cm, best_preds = s, m, cm, preds
        if ema_v > best_ema + 1e-4: best_ema = ema_v; bad = 0
        else:
            bad += 1
            if bad >= args.patience: break
    return best_m, best_cm, best_preds


def run_model_cv5(args, cv_csv, root, log):
    log("\n" + "="*70)
    log(f"5-FOLD: {args.model}")
    log("="*70)
    fold_ms = []
    all_ids, all_yt, all_pp = [], [], []
    for fold in range(N_FOLDS):
        t0 = time.time()
        m, cm, preds = train_eval_one_fold(args, fold, cv_csv, root)
        fold_ms.append(m)
        ids, yt, pp = preds
        all_ids.append(ids); all_yt.append(yt); all_pp.append(pp)
        log(f"  fold {fold}: W.Acc={m['weighted_accuracy']:.4f} "
            f"F1={m['macro_f1']:.4f} UAR={m['uar']:.4f} "
            f"recall P/U/A={m['recall_present']:.2f}/{m['recall_unknown']:.2f}/"
            f"{m['recall_absent']:.2f} ({time.time()-t0:.0f}s)")

    # tum fold hasta tahminlerini birlestir -> bootstrap icin kaydet
    cat_ids = np.concatenate(all_ids)
    cat_yt = np.concatenate(all_yt)
    cat_pp = np.concatenate(all_pp, axis=0)
    pred_path = CKPT_DIR / f"preds_{args.model}.npz"
    np.savez(pred_path, ids=cat_ids, y_true=cat_yt, probs=cat_pp)
    log(f"  (hasta tahminleri kaydedildi: {pred_path.name} — {len(cat_yt)} hasta)")

    # ortalama ± std
    log(f"\n  --- {args.model}: 5-fold ortalama ± std ---")
    keys = ["weighted_accuracy", "macro_f1", "uar",
            "recall_present", "recall_unknown", "recall_absent"]
    summary = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_ms])
        summary[k] = (vals.mean(), vals.std())
        log(f"    {k:20s}: {vals.mean():.4f} ± {vals.std():.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid_kan",
                    choices=["hybrid_kan", "hybrid_mlp", "hybrid2_kan", "hybrid2_mlp",
                             "hybrid3_kan", "hybrid3_mlp", "hybrid4_kan", "hybrid4_mlp",
                             "cnn2d", "cnn1d", "cnn_bilstm", "all"])
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

    root = Path(args.root); cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    logs = []
    def log(m=""): print(m); logs.append(str(m))

    models = (["cnn1d", "cnn2d", "cnn_bilstm", "hybrid_mlp", "hybrid_kan"]
              if args.model == "all" else [args.model])

    log("="*70)
    log(f"5-FOLD TEK-MODEL DEGERLENDIRME | modeller: {models}")
    log("="*70)

    all_summary = {}
    for mdl in models:
        args.model = mdl
        all_summary[mdl] = run_model_cv5(args, cv_csv, root, log)

    # karsilastirma tablosu
    if len(all_summary) > 1:
        log("\n" + "="*70)
        log("KARSILASTIRMA TABLOSU (5-fold W.Acc ± std)")
        log("="*70)
        log(f"  {'model':14s} | {'W.Acc':16s} | {'macro-F1':16s} | {'Unknown recall':16s}")
        log("  " + "-"*66)
        for mdl, s in all_summary.items():
            wa = s["weighted_accuracy"]; f1 = s["macro_f1"]; ur = s["recall_unknown"]
            log(f"  {mdl:14s} | {wa[0]:.3f} ± {wa[1]:.3f}    | "
                f"{f1[0]:.3f} ± {f1[1]:.3f}    | {ur[0]:.3f} ± {ur[1]:.3f}")

    suffix = "all" if len(models) > 1 else models[0]
    rep = REPORT_DIR / f"cv5_{suffix}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}")
    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
