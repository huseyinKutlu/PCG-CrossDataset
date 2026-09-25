#!/usr/bin/env python3
"""
13_ensemble.py
PCG Q1 Manuscript — Deep ensemble + belirsizlik analizi.

KARAR: Ayni hibrit govde (KAN veya MLP head), N farkli seed ile egitilir.
Uyelerin tahminleri birlestirilir; uyeler arasi ANLASMAZLIK belirsizlik
olcusu olarak kullanilir ve Unknown sinifiyla hizalanir.

Iki asama:
  --train : N seed'i egit (her seed ayri checkpoint)
  --eval  : N uyeyi yukle, ensemble + belirsizlik analizi

Belirsizlik olculeri:
  - predictive entropy : toplam belirsizlik
  - mutual information : epistemik (uyeler anlasmazligi) <- Unknown sinyali
  - ECE                : kalibrasyon

Kullanim:
  # mlp icin 5 seed egit
  python 13_ensemble.py --head mlp --n_seeds 5 --fold 0 --epochs 30 --train
  # sonra degerlendir
  python 13_ensemble.py --head mlp --n_seeds 5 --fold 0 --eval
  # ayni sekilde kan icin
  python 13_ensemble.py --head kan --n_seeds 5 --fold 0 --epochs 30 --train
  python 13_ensemble.py --head kan --n_seeds 5 --fold 0 --eval
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
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "ensemble"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3


# ---------- ortak yardimcilar (10_train_hybrid ile ayni mantik) ----------
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


def run_epoch(model, loader, crit, opt, clip=0.5):
    model.train(); tot, n, skip = 0.0, 0, 0
    for b in loader:
        raw = b["raw"].to(DEVICE, non_blocking=True)
        mel = b["mel"].to(DEVICE, non_blocking=True)
        y = b["label"].to(DEVICE, non_blocking=True)
        opt.zero_grad()
        logits = model(raw, mel); loss = crit(logits, y)
        if not torch.isfinite(loss): skip += 1; continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        tot += loss.item() * len(y); n += len(y)
    if skip: print(f"   [uyari] {skip} batch atlandi")
    return tot / max(n, 1)


@torch.no_grad()
def predict_patient_probs(model, loader, agg="mean"):
    """Bir modelin hasta-duzeyi olasiliklarini dondur (sirali pid ile)."""
    model.eval(); probs, pids = [], []
    for b in loader:
        raw = b["raw"].to(DEVICE, non_blocking=True)
        mel = b["mel"].to(DEVICE, non_blocking=True)
        probs.append(torch.softmax(model(raw, mel), 1).float().cpu().numpy())
        pids.extend(b["patient_id"])
    seg = np.concatenate(probs, 0)
    pp, ids = ds_mod.aggregate_to_patient(seg, pids, agg)
    return pp, ids


# ---------- belirsizlik metrikleri ----------
def entropy(p):
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def ece(probs, labels, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ---------- egitim asamasi ----------
def train_one_seed(args, seed, tr, va, cv_csv):
    torch.manual_seed(seed); np.random.seed(seed)
    sampler, cw = make_sampler(tr, args.sampler_beta)
    train_loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)

    model = hy_mod.HybridPCG(head=args.head, d_model=args.d_model).to(DEVICE)
    loss_w = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup, args.epochs))

    def sel_score(m): return 0.5 * m["weighted_accuracy"] + 0.5 * m["macro_f1"]
    best_sel, best_ep = -1.0, -1
    ema = deque(maxlen=3); best_ema, bad = -1.0, 0
    ckpt = CKPT_DIR / f"{args.head}_fold{args.fold}_seed{seed}.pth"

    for ep in range(1, args.epochs + 1):
        run_epoch(model, train_loader, crit, opt, clip=args.clip)
        sched.step()
        pp, ids = predict_patient_probs(model, val_loader)
        truth = ds_mod.patient_true_labels(cv_csv)
        yt = np.array([truth[p] for p in ids])
        m = me_mod.all_metrics(yt, pp.argmax(1))
        s = sel_score(m); ema.append(s); ema_v = float(np.mean(ema))
        if s > best_sel:
            best_sel, best_ep = s, ep
            torch.save({"model": model.state_dict(), "epoch": ep, "metrics": m,
                        "seed": seed}, ckpt)
        if ema_v > best_ema + 1e-4: best_ema = ema_v; bad = 0
        else:
            bad += 1
            if bad >= args.patience: break
    return ckpt, best_sel, best_ep


def stage_train(args, tr, va, cv_csv, log):
    log(f"\n=== EGITIM: {args.n_seeds} seed, head={args.head} ===")
    seeds = [42 + i * 100 for i in range(args.n_seeds)]
    for i, sd in enumerate(seeds):
        t0 = time.time()
        ckpt, bs, be = train_one_seed(args, sd, tr, va, cv_csv)
        log(f"  seed {sd} ({i+1}/{args.n_seeds}): best_sel={bs:.4f} @ep{be} "
            f"({time.time()-t0:.0f}s) -> {ckpt.name}")
    log("  [OK] tum seed'ler egitildi.")
    return seeds


# ---------- degerlendirme asamasi ----------
def stage_eval(args, va, cv_csv, log):
    seeds = [42 + i * 100 for i in range(args.n_seeds)]
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)
    truth = ds_mod.patient_true_labels(cv_csv)

    log(f"\n=== DEGERLENDIRME: ensemble ({args.n_seeds} seed), head={args.head} ===")

    # her uyenin hasta-olasiliklarini topla
    member_probs = []; ids_ref = None; single_metrics = []
    for sd in seeds:
        ckpt = CKPT_DIR / f"{args.head}_fold{args.fold}_seed{sd}.pth"
        if not ckpt.exists():
            log(f"  !! eksik checkpoint: {ckpt.name} — once --train calistir."); return
        model = hy_mod.HybridPCG(head=args.head, d_model=args.d_model).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
        pp, ids = predict_patient_probs(model, val_loader)
        if ids_ref is None: ids_ref = ids
        assert ids == ids_ref, "uye hasta sirasi tutarsiz!"
        member_probs.append(pp)
        yt = np.array([truth[p] for p in ids])
        sm = me_mod.all_metrics(yt, pp.argmax(1))
        single_metrics.append(sm)

    member_probs = np.stack(member_probs, 0)   # (N, n_hasta, 3)
    yt = np.array([truth[p] for p in ids_ref])

    # tek-model ortalama performans (referans)
    log("\n--- Tek-model performans (uye ortalamasi) ---")
    avg_single = {k: np.mean([m[k] for m in single_metrics]) for k in single_metrics[0]}
    std_single = {k: np.std([m[k] for m in single_metrics]) for k in single_metrics[0]}
    log(f"  W.Acc={avg_single['weighted_accuracy']:.4f}±{std_single['weighted_accuracy']:.4f} | "
        f"F1={avg_single['macro_f1']:.4f}±{std_single['macro_f1']:.4f} | "
        f"UAR={avg_single['uar']:.4f}")
    log(f"  Unknown recall (tek-model ort): {avg_single['recall_unknown']:.4f}")

    # ENSEMBLE: olasilik ortalamasi
    ens_prob = member_probs.mean(0)            # (n_hasta, 3)
    ens_pred = ens_prob.argmax(1)
    ens_m = me_mod.all_metrics(yt, ens_pred)
    ens_cm = me_mod.confusion_matrix(yt, ens_pred)
    log("\n--- ENSEMBLE performans (olasilik ortalamasi) ---")
    log("  " + me_mod.format_metrics(ens_m))

    # kazanim
    dW = ens_m['weighted_accuracy'] - avg_single['weighted_accuracy']
    dF = ens_m['macro_f1'] - avg_single['macro_f1']
    log(f"  Kazanim: ΔW.Acc={dW:+.4f} | ΔF1={dF:+.4f}")

    log("\n  Confusion [gercek x tahmin] (P/U/A):")
    log("          pred_P  pred_U  pred_A")
    for i, cn in enumerate(me_mod.CLASSES):
        log(f"     {cn:>7s} {ens_cm[i,0]:6d} {ens_cm[i,1]:7d} {ens_cm[i,2]:7d}")

    # --- BELIRSIZLIK analizi ---
    log("\n--- BELIRSIZLIK analizi ---")
    total_unc = entropy(ens_prob)                       # predictive entropy
    exp_ent = entropy(member_probs).mean(0)             # ortalama uye entropisi
    mutual_info = total_unc - exp_ent                   # epistemik (anlasmazlik)
    member_preds = member_probs.argmax(-1)              # (N, n_hasta)
    disagree = np.array([1 - np.bincount(member_preds[:, i], minlength=N_CLASSES).max()/args.n_seeds
                         for i in range(len(yt))])

    # Belirsizlik Unknown'i yakaliyor mu? Sinif bazinda ortalama belirsizlik
    log("  Sinif bazinda ortalama belirsizlik (yuksek=model emin degil):")
    log("    sinif    | entropy | mutual_info | oylama_anlasmazlik")
    for c, cn in enumerate(me_mod.CLASSES):
        mask = (yt == c)
        if mask.sum() > 0:
            log(f"    {cn:>8s} |  {total_unc[mask].mean():.3f}  |    {mutual_info[mask].mean():.3f}    "
                f"|      {disagree[mask].mean():.3f}")
    log("  >> BEKLENTI: Unknown sinifi en yuksek belirsizlik/anlasmazligi gostermeli")
    log("     (model Unknown'da 'emin degilim' diyorsa hizalama dogru).")

    # kalibrasyon
    ens_ece = ece(ens_prob, yt)
    single_ece = np.mean([ece(member_probs[i], yt) for i in range(args.n_seeds)])
    log(f"\n  ECE (kalibrasyon, dusuk=iyi): tek-model={single_ece:.4f} -> ensemble={ens_ece:.4f}")
    log(f"  {'[OK] ensemble daha iyi kalibre' if ens_ece < single_ece else '[!] ensemble kalibrasyonu kotulesti'}")

    # belirsizlik-tabanli reddetme (selective prediction) on-analiz
    log("\n--- Selective prediction on-analiz ---")
    log("  En belirsiz %X hastayi 'Unknown/incele' diye reddedersek dogruluk:")
    order = np.argsort(-total_unc)  # en belirsizden aza
    for frac in [0.0, 0.1, 0.2, 0.3]:
        n_reject = int(frac * len(yt))
        keep = order[n_reject:]
        if len(keep) > 0:
            acc_kept = (ens_pred[keep] == yt[keep]).mean()
            log(f"    reddet %{int(frac*100):2d} -> kalan {len(keep):3d} hasta, dogruluk={acc_kept:.3f}")

    # ozet kaydet
    rep = REPORT_DIR / f"ensemble_{args.head}_fold{args.fold}.txt"
    return ens_m, avg_single, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=list(hy_mod.HEADS), default="kan")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sampler_beta", type=float, default=0.5)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--train", action="store_true", help="seed'leri egit")
    ap.add_argument("--eval", action="store_true", help="ensemble degerlendir")
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    if not (args.train or args.eval):
        args.train = args.eval = True   # ikisini de yap

    root = Path(args.root); cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    logs = []
    def log(m=""): print(m); logs.append(str(m))

    log("="*70)
    log(f"DEEP ENSEMBLE | head={args.head} | n_seeds={args.n_seeds} | fold={args.fold}")
    log("="*70)

    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="train")
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="val")
    log(f"Train {len(tr)} seg | Val {len(va)} seg / {va.records['patient_id'].nunique()} hasta")

    if args.train:
        stage_train(args, tr, va, cv_csv, log)
    if args.eval:
        result = stage_eval(args, va, cv_csv, log)
        if result:
            _, _, rep = result
            rep.write_text("\n".join(logs), encoding="utf-8")
            log(f"\nRapor: {rep}")

    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
