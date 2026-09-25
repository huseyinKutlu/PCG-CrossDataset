#!/usr/bin/env python3
"""
21_dann_da.py
PCG Q1 Manuscript — DANN (Domain-Adversarial Neural Network).

GEREKCE: A-distance teshisi 1.70 cikti (ozellikler AYRISIK). CORAL (sadece
kovaryans = 2. moment hizalar) bu farki yakalayamadi (coral~0.0005, etkisiz).
DANN, bir domain-classifier'i gradient reversal ile KANDIRARAK feature
extractor'i domain-invariant olmaya zorlar — A-distance'i dogrudan dusurur.

MIMARI:
  feature_extractor -> task_classifier  (normal/abnormal, CirCor etiketli)
                    -> GRL -> domain_classifier  (CirCor/CinC ayrimi)
  L = L_task + lambda * L_domain
  GRL (gradient reversal): ileri=identity, geri=-lambda*grad
  -> feature extractor domain ayrimini ZORLASTIRAN ozellik ogrenir.

CinC ETIKETSIZ (sadece domain etiketi=1). -> unsupervised domain adaptation.
lambda egitim boyunca 0->1 rampa (DANN standardi, erken egitimde kararlilik).

Kullanim:
  python 21_dann_da.py --model hybrid_kan --dann_lambda 1.0 --epochs 30
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
ext_mod = import_module("16_external_val")
coral_mod = import_module("19_coral_da")   # get_features_and_logits, to_binary, sampler

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "dann"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CIRCOR_TO_BINARY = {0: 1, 1: 1, 2: 0}


# ---- Gradient Reversal Layer ----
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


# ---- domain classifier (feature -> CirCor/CinC) ----
class DomainClassifier(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d_model, 2))
    def forward(self, x): return self.net(x)


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


def lr_lambda_factory(warmup, total):
    def f(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        prog = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid_kan",
                    choices=["hybrid_kan", "hybrid_mlp", "cnn_bilstm"])
    ap.add_argument("--dann_lambda", type=float, default=1.0,
                    help="domain kaybi max agirligi (0->bu degere rampa)")
    ap.add_argument("--dwt_norm", action="store_true",
                    help="DWT bant-normalizasyonu (DANN ile birlikte, hem train hem test)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sampler_beta", type=float, default=0.5)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42,
                    help="reproduktibilite + coklu-seed DANN varyans olcumu icin")
    ap.add_argument("--ext_dataset", default="cinc", choices=["cinc", "yaseen"],
                    help="hedef dis dataset: cinc veya yaseen")
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    # seed set (DANN coklu-seed varyans olcumu)
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    if args.ext_dataset == "yaseen":
        cinc_manifest = root / "manifests" / "yaseen_processed.csv"
        ExtDS = ext_mod.YaseenDataset
    else:
        cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
        ExtDS = ext_mod.CinCDataset
    logs = []
    def log(m=""): print(m); logs.append(str(m))

    log("="*70)
    log(f"DANN domain-adversarial | model={args.model} | dann_lambda={args.dann_lambda}")
    log(f"Kaynak: CirCor (etiketli) | Hedef: {args.ext_dataset.upper()} (ETIKETSIZ, domain-adversarial)")
    log("A-distance=1.70 -> DANN bunu dusurmeye calisir")
    log("="*70)

    cinc = ExtDS(cinc_manifest, root)
    cinc.dwt_norm = args.dwt_norm   # DWT bant-normalizasyonu (test tarafi)
    cinc_loader = DataLoader(cinc, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, drop_last=True)

    internal_au, external_au = [], []
    cinc_probs_folds = []; cinc_y = None

    total_steps = args.epochs

    for fold in range(N_FOLDS):
        t0 = time.time()
        tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="train")
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        sampler, cw = make_binary_sampler(tr, args.sampler_beta)
        train_ds = ext_mod.DWTCircor(tr) if args.dwt_norm else tr
        src_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        va_ds = ext_mod.DWTCircor(va) if args.dwt_norm else va
        val_loader = ds_mod.make_eval_loader(va_ds, batch_size=128, num_workers=args.num_workers)

        model, mode = ext_mod.build_binary_model(args.model)
        model = model.to(DEVICE)
        # feature boyutunu ilk batch'ten algila (hybrid=128, cnn_bilstm=256)
        with torch.no_grad():
            _b = next(iter(src_loader))
            _f, _ = coral_mod.get_features_and_logits(
                model, _b["raw"].to(DEVICE), _b["mel"].to(DEVICE), mode, args.model)
            feat_dim = _f.shape[1]
        domain_clf = DomainClassifier(feat_dim).to(DEVICE)
        loss_w = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
        crit_dom = nn.CrossEntropyLoss()
        # DANN dengesi: domain_clf task'i feature extractor'dan HIZLI ogrenmeli
        opt = torch.optim.AdamW([
            {"params": model.parameters(), "lr": args.lr},
            {"params": domain_clf.parameters(), "lr": args.lr * 2.0},
        ], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup, args.epochs))

        best_auroc, bad = -1.0, 0
        ema = deque(maxlen=5); best_ema = -1.0   # daha genis EMA (tohum-duyarliligi azalt)
        dwt_sfx = "_dwt" if args.dwt_norm else ""
        ckpt = CKPT_DIR / f"{args.model}_dann{dwt_sfx}_s{args.seed}_fold{fold}.pth"

        for ep in range(1, args.epochs + 1):
            model.train(); domain_clf.train()
            # DANN lambda rampa: YAVAS + gecikmeli (feature extractor once task ogrensin)
            # ilk %20 epoch lambda~0, sonra yavas yukseli (-5 katsayisi, -10 yerine)
            p_raw = (ep - 1) / max(total_steps - 1, 1)
            p = max(0.0, (p_raw - 0.2) / 0.8)   # ilk %20 gecikme
            lamb = args.dann_lambda * (2.0 / (1.0 + np.exp(-5 * p)) - 1.0)  # yavas rampa
            cinc_iter = iter(cinc_loader)
            tot_task, tot_dom, n = 0.0, 0.0, 0
            for b in src_loader:
                raw_s = b["raw"].to(DEVICE); mel_s = b["mel"].to(DEVICE)
                y = to_binary(b["label"]).to(DEVICE)
                try: bt = next(cinc_iter)
                except StopIteration:
                    cinc_iter = iter(cinc_loader); bt = next(cinc_iter)
                raw_t = bt["raw"].to(DEVICE); mel_t = bt["mel"].to(DEVICE)

                opt.zero_grad()
                # kaynak: ozellik + task logits
                feat_s, logits_s = coral_mod.get_features_and_logits(model, raw_s, mel_s, mode, args.model)
                feat_t, _ = coral_mod.get_features_and_logits(model, raw_t, mel_t, mode, args.model)
                if feat_s is None:
                    log("!! bu model ozellik cikaramiyor, DANN uygulanamaz"); return
                loss_task = crit(logits_s, y)
                # domain: GRL -> domain_clf, kaynak=0 hedef=1
                feat_all = torch.cat([feat_s, feat_t], 0)
                dom_y = torch.cat([torch.zeros(len(feat_s)), torch.ones(len(feat_t))]).long().to(DEVICE)
                dom_logits = domain_clf(grad_reverse(feat_all, lamb))
                loss_dom = crit_dom(dom_logits, dom_y)
                loss = loss_task + loss_dom
                if not torch.isfinite(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(domain_clf.parameters()), args.clip)
                opt.step()
                tot_task += loss_task.item() * len(y)
                tot_dom += loss_dom.item() * len(y); n += len(y)
            sched.step()

            rp, ry, _ = ext_mod.predict_grouped(model, val_loader, mode, "patient_id", circor=True)
            au = me_mod.binary_metrics(ry, rp)["auroc"]
            ema.append(au); ema_v = float(np.mean(ema))
            if au > best_auroc:
                best_auroc = au
                torch.save({"model": model.state_dict(), "epoch": ep}, ckpt)
            if ema_v > best_ema + 1e-4: best_ema = ema_v; bad = 0
            else:
                bad += 1
                if bad >= args.patience: break

        model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])
        rp, ry, _ = ext_mod.predict_grouped(model, val_loader, mode, "patient_id", circor=True)
        in_au = me_mod.binary_metrics(ry, rp)["auroc"]; internal_au.append(in_au)
        cinc_eval = DataLoader(cinc, batch_size=128, shuffle=False, num_workers=args.num_workers)
        cp, cy, _ = ext_mod.predict_grouped(model, cinc_eval, mode, "record_id", circor=False)
        cinc_probs_folds.append(cp); cinc_y = cy
        ex_au = me_mod.binary_metrics(cy, cp)["auroc"]; external_au.append(ex_au)
        log(f"  fold {fold}: ic AUROC={in_au:.4f} | dis AUROC={ex_au:.4f} "
            f"(task={tot_task/max(n,1):.3f} dom={tot_dom/max(n,1):.3f} λ_max={lamb:.2f}) "
            f"({time.time()-t0:.0f}s)")

    ia, ea = np.array(internal_au), np.array(external_au)
    cinc_ens = np.mean(cinc_probs_folds, axis=0)
    ens_au = me_mod.binary_metrics(cinc_y, cinc_ens)["auroc"]
    log(f"\n  --- {args.model} + DANN (lambda={args.dann_lambda}) ---")
    log(f"    Ic  AUROC: {ia.mean():.4f} ± {ia.std():.4f}")
    log(f"    Dis AUROC: {ea.mean():.4f} ± {ea.std():.4f}")
    log(f"    Dis (5-fold ens): {ens_au:.4f}")
    log(f"\n  KIYAS (dis AUROC):")
    log(f"    hybrid_kan: naif=0.568 | CORAL(λ=10)=0.709 | DANN={ens_au:.4f}")
    log(f"    cnn_bilstm: naif=0.700")

    rep = REPORT_DIR / f"dann_{args.model}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}")
    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
