#!/usr/bin/env python3
"""
25_uncertainty_benchmark.py
Belirsizlik kestirim yontemleri KIYASLAMASI (Q1 metodolojik eksen).

KIYASLANAN 4 YONTEM:
  1) Deep Ensemble (N model, tahmin yayilimi)
  2) MC Dropout (tek model, N forward, dropout acik)
  3) Evidential (EDL, tek model, Dirichlet belirsizligi u=K/S)
  4) KAN + Ensemble (mevcut yaklasim)

4 METRIK (belirsizligin "iyiligi"):
  A) ECE (kalibrasyon) — dusuk=iyi
  B) Selective prediction — en belirsiz %X reddedilince dogruluk artisi
  C) OOD duyarliligi — CinC (dis) belirsizligi > CirCor (ic) belirsizligi mi?
     (en ozgun metrik: model tanimadigi veride "emin degilim" diyor mu?)
  D) Unknown hizalamasi — CirCor 'Unknown' sinifi en yuksek belirsizligi aliyor mu?

NOT: bu script 3-sinifli (Present/Unknown/Absent) modeller uzerinde calisir
(Unknown hizalamasi icin). cnn_bilstm omurga.

Kullanim:
  python 25_uncertainty_benchmark.py --train     # 4 yontemi egit
  python 25_uncertainty_benchmark.py --eval      # kiyasla
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
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "uncert"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3


# ---------- belirsizlik metrikleri ----------
def predictive_entropy(probs):
    """probs: (N, K) -> (N,) Shannon entropi (toplam belirsizlik)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(1)


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: guven ile dogrulugun uyumu. Dusuk=iyi."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return ece


def selective_prediction(probs, labels, uncert, rejects=(0.1, 0.2, 0.3)):
    """En belirsiz %X reddedilince kalan dogruluk."""
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    order = np.argsort(uncert)   # dusuk belirsizlik -> guvenilir
    out = {}
    base = correct.mean()
    out["0.0"] = base
    for r in rejects:
        keep = order[:int(len(order) * (1 - r))]
        out[f"{r}"] = correct[keep].mean()
    return out


# ---------- yontem-bazli belirsizlik cikarma ----------
@torch.no_grad()
def infer_ensemble(models, loader, mode="raw"):
    """N model -> ortalama olasilik + epistemik belirsizlik (yayilim)."""
    all_probs = []
    for m in models:
        m.eval()
        ps = []
        for b in loader:
            x = b["raw"].to(DEVICE) if mode == "raw" else b["mel"].to(DEVICE)
            logits = m(x)
            ps.append(F.softmax(logits, 1).cpu().numpy())
        all_probs.append(np.concatenate(ps))
    all_probs = np.stack(all_probs)        # (M, N, K)
    mean_p = all_probs.mean(0)             # (N, K)
    # epistemik: tahminler arasi yayilim (vote disagreement / varyans)
    epi = all_probs.var(0).sum(1)          # (N,)
    ent = predictive_entropy(mean_p)
    return mean_p, ent, epi


@torch.no_grad()
def infer_mc_dropout(model, loader, mode="raw", n_samples=20):
    """Tek model, dropout ACIK, N forward."""
    model.train()   # dropout acik kalsin (ama BN'yi eval moduna almak gerek)
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            mod.eval()
    all_probs = []
    for _ in range(n_samples):
        ps = []
        for b in loader:
            x = b["raw"].to(DEVICE) if mode == "raw" else b["mel"].to(DEVICE)
            logits = model(x)
            ps.append(F.softmax(logits, 1).cpu().numpy())
        all_probs.append(np.concatenate(ps))
    all_probs = np.stack(all_probs)
    mean_p = all_probs.mean(0)
    epi = all_probs.var(0).sum(1)
    ent = predictive_entropy(mean_p)
    return mean_p, ent, epi


@torch.no_grad()
def infer_evidential(model, loader, mode="raw"):
    """EDL: tek forward, Dirichlet belirsizligi u=K/S."""
    model.eval()
    probs, us = [], []
    for b in loader:
        x = b["raw"].to(DEVICE) if mode == "raw" else b["mel"].to(DEVICE)
        logits = model(x)
        p, u = hy_mod.edl_predict(logits)
        probs.append(p.cpu().numpy()); us.append(u.cpu().numpy())
    mean_p = np.concatenate(probs)
    u = np.concatenate(us)   # Dirichlet vacuity (OOD-duyarli)
    return mean_p, u, u


def aggregate_patient(probs, uncert, loader):
    """segment -> hasta (ortalama). loader sirasi korunur varsayimi."""
    # bu script segment-seviyesinde calisir; basitlik icin segment-seviye raporlar
    return probs, uncert


# ---------- ana ----------
def train_cnn_bilstm(args, tr, va, head, loss_type, seed):
    """cnn_bilstm egit. head: 'linear'|'kan'. loss_type: 'ce'|'edl'."""
    torch.manual_seed(seed); np.random.seed(seed)
    if head == "kan":
        # cnn_bilstm govde + KAN head: HybridPCG2 degil, dogrudan cnn_bilstm'e KAN tak
        model = CNNBiLSTM_KAN(n_classes=N_CLASSES).to(DEVICE)
    else:
        model = bl_mod.CNN_BiLSTM(n_classes=N_CLASSES).to(DEVICE)
    sampler = make_smoothed_sampler(tr, args.sampler_beta)
    loader = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)
    labels = np.array(tr.seg_labels)
    c = np.bincount(labels, minlength=N_CLASSES).astype(float)
    w = torch.tensor((c.sum()/(N_CLASSES*np.maximum(c,1))), dtype=torch.float32).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_w, best = None, -1
    for ep in range(1, args.epochs+1):
        model.train()
        for b in loader:
            x = b["raw"].to(DEVICE); y = b["label"].to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            if loss_type == "edl":
                loss = hy_mod.edl_loss(logits, y, epoch=ep, anneal=args.anneal)
            else:
                loss = crit(logits, y)
            if not torch.isfinite(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        sched.step()
        # secim: ic W.Acc
        if ep % 3 == 0 or ep == args.epochs:
            model.eval()
            ps, ls = [], []
            with torch.no_grad():
                for b in val_loader:
                    lo = model(b["raw"].to(DEVICE))
                    ps.append(F.softmax(lo,1).cpu().numpy()); ls.append(b["label"].numpy())
            wacc = me_mod.weighted_accuracy(np.concatenate(ls), np.concatenate(ps).argmax(1))
            if wacc > best:
                best = wacc; best_w = {k: v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_w)
    return model, best


class CNNBiLSTM_KAN(nn.Module):
    """cnn_bilstm govde + KAN head (kiyas icin)."""
    def __init__(self, n_classes=N_CLASSES, hidden=128, n_layers=2):
        super().__init__()
        def cb(ci, co, k=7, s=4, p=3):
            return nn.Sequential(nn.Conv1d(ci, co, k, s, p),
                                 nn.GroupNorm(min(8, co), co), nn.GELU())
        self.stem = nn.Sequential(cb(1,32), cb(32,64), cb(64,128))
        self.lstm = nn.LSTM(128, hidden, n_layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.head = hy_mod.KANHead(hidden*2, n_classes)
    def forward(self, x):
        h = self.stem(x).transpose(1,2)
        out, _ = self.lstm(h)
        return self.head(out.mean(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_ensemble", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--anneal", type=int, default=10)
    ap.add_argument("--sampler_beta", type=float, default=0.5)
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
    logs = []
    def log(m=""): print(m); logs.append(str(m))

    log("="*70)
    log("BELIRSIZLIK YONTEMLERI KIYASLAMASI | cnn_bilstm omurga | fold=%d" % args.fold)
    log("Ensemble(5) | MC-Dropout | Evidential(EDL) | KAN+Ensemble(5)")
    log("="*70)

    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="train")
    va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="val")
    val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)
    cinc = ext_mod.CinCDataset(cinc_manifest, root)
    cinc_loader = DataLoader(cinc, batch_size=128, shuffle=False, num_workers=args.num_workers)

    seeds = [42 + i*100 for i in range(args.n_ensemble)]

    if args.train:
        log("\n[EGITIM] ensemble icin %d seed (CE), EDL icin 1, KAN icin %d seed..." % (args.n_ensemble, args.n_ensemble))
        # Ensemble (linear head, CE) — n_ensemble seed
        for i, sd in enumerate(seeds):
            m, w = train_cnn_bilstm(args, tr, va, "linear", "ce", sd)
            torch.save(m.state_dict(), CKPT_DIR / f"ens_linear_f{args.fold}_s{sd}.pth")
            log(f"  ensemble seed{sd} ({i+1}/{args.n_ensemble}): ic W.Acc={w:.4f}")
        # EDL (linear head, edl loss) — 1 model
        m, w = train_cnn_bilstm(args, tr, va, "linear", "edl", 42)
        torch.save(m.state_dict(), CKPT_DIR / f"edl_f{args.fold}.pth")
        log(f"  EDL: ic W.Acc={w:.4f}")
        # KAN (kan head, CE) — n_ensemble seed
        for i, sd in enumerate(seeds):
            m, w = train_cnn_bilstm(args, tr, va, "kan", "ce", sd)
            torch.save(m.state_dict(), CKPT_DIR / f"kan_f{args.fold}_s{sd}.pth")
            log(f"  KAN seed{sd} ({i+1}/{args.n_ensemble}): ic W.Acc={w:.4f}")
        log("  [OK] tum modeller egitildi.")

    if args.eval:
        log("\n[DEGERLENDIRME] 4 yontem, 4 metrik")
        results = {}

        # --- ENSEMBLE ---
        ens_models = []
        for sd in seeds:
            m = bl_mod.CNN_BiLSTM(n_classes=N_CLASSES).to(DEVICE)
            m.load_state_dict(torch.load(CKPT_DIR / f"ens_linear_f{args.fold}_s{sd}.pth", map_location=DEVICE))
            ens_models.append(m)
        p_in, ent_in, epi_in = infer_ensemble(ens_models, val_loader)
        p_ood, ent_ood, epi_ood = infer_ensemble(ens_models, cinc_loader)
        lab_in = np.concatenate([b["label"].numpy() for b in val_loader])
        results["Ensemble"] = eval_method(p_in, ent_in, lab_in, ent_ood, log)

        # --- MC DROPOUT (ensemble[0] modeli) ---
        p_in, ent_in, epi_in = infer_mc_dropout(ens_models[0], val_loader, n_samples=args.mc_samples)
        p_ood, ent_ood, _ = infer_mc_dropout(ens_models[0], cinc_loader, n_samples=args.mc_samples)
        results["MC-Dropout"] = eval_method(p_in, ent_in, lab_in, ent_ood, log)

        # --- EDL ---
        edl_m = bl_mod.CNN_BiLSTM(n_classes=N_CLASSES).to(DEVICE)
        edl_m.load_state_dict(torch.load(CKPT_DIR / f"edl_f{args.fold}.pth", map_location=DEVICE))
        p_in, u_in, _ = infer_evidential(edl_m, val_loader)
        p_ood, u_ood, _ = infer_evidential(edl_m, cinc_loader)
        results["Evidential"] = eval_method(p_in, u_in, lab_in, u_ood, log)

        # --- KAN + Ensemble ---
        kan_models = []
        for sd in seeds:
            m = CNNBiLSTM_KAN(n_classes=N_CLASSES).to(DEVICE)
            m.load_state_dict(torch.load(CKPT_DIR / f"kan_f{args.fold}_s{sd}.pth", map_location=DEVICE))
            kan_models.append(m)
        p_in, ent_in, _ = infer_ensemble(kan_models, val_loader)
        p_ood, ent_ood, _ = infer_ensemble(kan_models, cinc_loader)
        results["KAN+Ens"] = eval_method(p_in, ent_in, lab_in, ent_ood, log)

        # --- OZET TABLO ---
        log("\n" + "="*70)
        log("OZET TABLO (4 yontem x 4 metrik)")
        log("="*70)
        log(f"{'Yontem':<12} {'ECE':>7} {'Sel@30%':>9} {'OOD-fark':>9} {'Unk-hiz':>8}")
        log("-"*50)
        for name, r in results.items():
            log(f"{name:<12} {r['ece']:>7.4f} {r['sel30']:>9.4f} "
                f"{r['ood_diff']:>+9.4f} {r['unk_align']:>8s}")
        log("\nOOD-fark: dis-ic belirsizlik (+ ise OOD-duyarli, - ise yanlis guven)")
        log("Sel@30%: en belirsiz %30 reddedilince dogruluk")

    rep = REPORT_DIR / f"uncertainty_benchmark_f{args.fold}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}")
    log("=== TAMAMLANDI ===")


def eval_method(probs_in, uncert_in, labels_in, uncert_ood, log):
    """Bir yontem icin 4 metrik hesapla."""
    ece = expected_calibration_error(probs_in, labels_in)
    sel = selective_prediction(probs_in, labels_in, uncert_in)
    ood_diff = float(uncert_ood.mean() - uncert_in.mean())
    # Unknown hizalama
    u_unk = uncert_in[labels_in==1].mean() if (labels_in==1).sum()>0 else 0
    u_oth = uncert_in[labels_in!=1].mean()
    unk_align = "EVET" if u_unk > u_oth else "hayir"
    return {"ece": ece, "sel30": sel["0.3"], "ood_diff": ood_diff, "unk_align": unk_align}


def make_smoothed_sampler(ds, beta=0.5):
    from torch.utils.data import WeightedRandomSampler
    labels = np.array(ds.seg_labels)
    c = np.bincount(labels, minlength=N_CLASSES).astype(float)
    w = (c.sum() / (N_CLASSES * np.maximum(c, 1))) ** beta
    return WeightedRandomSampler(torch.DoubleTensor(w[labels]), len(labels), replacement=True)


if __name__ == "__main__":
    main()
