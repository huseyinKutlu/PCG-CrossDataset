#!/usr/bin/env python3
"""
42_ablation.py
PCG-MambaConformer ablation: hangi dal Yaseen basarisini (0.972) sagliyor?

8 konfigurasyon:
  Single-branch (4): sadece incep / conf / convnext / mamba
  Leave-one-out (4): tam model eksi her bir dal
  + referans: tam model (zaten egitildi, checkpoint'ten)

Her konfig: 5-fold CirCor egitim + Yaseen & CinC cross-dataset test.
16_external_val.py altyapisini yeniden kullanir.

Kullanim:
  python 42_ablation.py                 # tum ablation (uzun)
  python 42_ablation.py --config conf   # tek konfig
  python 42_ablation.py --quick         # 2 fold, hizli on-bakis
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

ev = import_module("16_external_val")
ds_mod = import_module("05_dataset")
mc_mod = import_module("37_mambaconformer")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "ablation"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
ALL_BRANCHES = ("incep", "conf", "cwt", "mamba")

# konfigurasyonlar
CONFIGS = {
    "only_incep": ("incep",),
    "only_conf": ("conf",),
    "only_cwt": ("cwt",),
    "only_mamba": ("mamba",),
    "no_incep": ("conf", "cwt", "mamba"),
    "no_conf": ("incep", "cwt", "mamba"),
    "no_cwt": ("incep", "conf", "mamba"),   # CWT'siz — CWT kritik mi?
    "no_mamba": ("incep", "conf", "cwt"),
}


def log(m): print(m, flush=True)


def train_config(cfg_name, branches, n_folds, epochs, root):
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    for fold in range(n_folds):
        ck = CKPT_DIR / f"{cfg_name}_fold{fold}.pth"
        if ck.exists():
            continue
        tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="train")
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        tr.return_cwt = True; va.return_cwt = True
        sampler, cw = ev.make_binary_sampler(tr, 0.5)
        cw_t = torch.tensor(np.asarray(cw), dtype=torch.float32).to(DEVICE)
        tl = DataLoader(tr, batch_size=48, sampler=sampler, num_workers=8, drop_last=True)
        vl = ds_mod.make_eval_loader(va, batch_size=96, num_workers=8)
        model = mc_mod.PCGMambaConformer(n_classes=2, pretrained=True,
                                         use_branches=branches).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss(weight=cw_t, label_smoothing=0.05)
        best_auc, best_state = 0, None
        for ep in range(epochs):
            ev.run_epoch(model, tl, crit, opt, "tri", clip=0.5, circor=True)
            sched.step()
            # ic val AUROC
            auc = eval_internal(model, vl)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save({"model": best_state, "auroc": best_auc}, ck)
        log(f"    {cfg_name} fold{fold}: ic-AUROC={best_auc:.4f}")


@torch.no_grad()
def eval_internal(model, loader):
    model.eval(); ps, ys = [], []
    for b in loader:
        logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE), b["cwt"].to(DEVICE))
        p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        ps.append(p); ys.append(ev.to_binary_labels(b["label"]).numpy())
    return roc_auc_score(np.concatenate(ys), np.concatenate(ps))


@torch.no_grad()
def eval_external(cfg_name, branches, ExtDS, manifest, n_folds, root):
    ds = ExtDS(manifest, root, norm_adapt="none")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    fold_p, ref_y, ref_id = [], None, None
    for fold in range(n_folds):
        ck = CKPT_DIR / f"{cfg_name}_fold{fold}.pth"
        if not ck.exists(): continue
        model = mc_mod.PCGMambaConformer(n_classes=2, pretrained=False,
                                         use_branches=branches).to(DEVICE)
        sd = torch.load(ck, map_location=DEVICE)
        model.load_state_dict(sd["model"]); model.eval()
        sp, sy, si = [], [], []
        for b in loader:
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE), b["cwt"].to(DEVICE))
            sp.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            sy.append(b["label"].numpy()); si.extend(b["record_id"])
        sp = np.concatenate(sp); sy = np.concatenate(sy); si = np.array(si)
        uniq = np.unique(si)
        fold_p.append(np.array([sp[si == u].mean() for u in uniq]))
        if ref_y is None:
            ref_y = np.array([sy[si == u][0] for u in uniq])
    if not fold_p: return None
    ens = np.mean(fold_p, 0)
    auc = roc_auc_score(ref_y, ens)
    cm = confusion_matrix(ref_y, (ens >= 0.5).astype(int), labels=[0, 1])
    sens = cm[1, 1] / max(cm[1].sum(), 1); spec = cm[0, 0] / max(cm[0].sum(), 1)
    return auc, sens, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    n_folds = 2 if args.quick else 5
    epochs = 10 if args.quick else args.epochs
    root = PROJECT_ROOT
    yas_m = root / "manifests" / "yaseen_processed.csv"
    cinc_m = root / "manifests" / "cinc2016_processed.csv"

    configs = CONFIGS if args.config == "all" else {args.config: CONFIGS[args.config]}
    log("#"*64)
    log("# MAMBACONFORMER ABLATION")
    log(f"# {n_folds} fold, {epochs} epoch | konfig: {list(configs.keys())}")
    log("#"*64)

    results = {}
    for cfg, branches in configs.items():
        log(f"\n=== {cfg}: {branches} ===")
        t0 = time.time()
        train_config(cfg, branches, n_folds, epochs, root)
        yas = eval_external(cfg, branches, ev.YaseenDataset, yas_m, n_folds, root)
        cinc = eval_external(cfg, branches, ev.CinCDataset, cinc_m, n_folds, root)
        results[cfg] = (yas, cinc)
        if yas: log(f"  Yaseen: AUROC={yas[0]:.4f} Sens={yas[1]:.3f} Spec={yas[2]:.3f}")
        if cinc: log(f"  CinC:   AUROC={cinc[0]:.4f} Sens={cinc[1]:.3f} Spec={cinc[2]:.3f}")
        log(f"  ({time.time()-t0:.0f}s)")

    # ozet tablo
    log("\n" + "="*64)
    log("ABLATION OZET (referans: tam model Yaseen=0.972, CinC=0.607)")
    log("="*64)
    log(f"  {'konfig':16s} | {'Yaseen AUROC':14s} | {'CinC AUROC':12s}")
    log("  " + "-"*48)
    for cfg, (yas, cinc) in results.items():
        ya = f"{yas[0]:.4f} (Se{yas[1]:.2f})" if yas else "—"
        ca = f"{cinc[0]:.4f}" if cinc else "—"
        log(f"  {cfg:16s} | {ya:14s} | {ca:12s}")
    log("\n[BITTI] Yorum: tam modele en yakin/en uzak konfig hangisi?")


if __name__ == "__main__":
    main()
