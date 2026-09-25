#!/usr/bin/env python3
"""
49_reciprocal_transfer.py
RECIPROCAL (cift-yonlu) cross-dataset transfer — Hakem-onleyici, yon-asimetrisi testi.

Mevcut makale: CirCor (egit) -> CinC/Yaseen (test).  [ileri yon]
Bu script:     CinC   (egit) -> CirCor (test) + Yaseen (test).  [ters yon]

NEDEN sadece CinC<->CirCor (Yaseen egitime SOKULMAZ):
  - CinC ve CirCor ikisi de GENEL normal/abnormal etiketli -> etiket tanimi uyumlu.
  - Yaseen 5 spesifik KAPAK hastaligi -> Yaseen'de egitmek "kapak dedektoru" yapar,
    "genel-anormal dedektoru" degil; CirCor'a transferi etiket-uyusmazligini olcer,
    domain shift'i DEGIL. Bu yuzden Yaseen yalniz TEST hedefi olarak kalir.
  - CinC'te patient_id YOK -> kayit-disjoint 5-fold (her kayit bagimsiz; CinC std.).

ORIJINAL checkpoint'leri EZMEZ: checkpoints/reciprocal_cinc/ 'e yazar.

Kullanim:
  python 49_reciprocal_transfer.py --model cnn_bilstm
  python 49_reciprocal_transfer.py --model mambaconformer
"""
from __future__ import annotations
import argparse, time, json
from pathlib import Path
from importlib import import_module
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, confusion_matrix

ev = import_module("16_external_val")
ds_mod = import_module("05_dataset")
me_mod = import_module("06_metrics")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = ev.DEVICE
N_FOLDS = 5
RANDOM_STATE = 42

# ileri yon (mevcut makale) referans degerleri — karsilastirma icin
FORWARD = {
    "cnn_bilstm":     {"cinc": 0.727, "yaseen": 0.806},   # CirCor->X
    "mambaconformer": {"cinc": 0.601, "yaseen": 0.985},
}


def make_cinc_folds(manifest_csv, out_csv):
    """CinC kayit-disjoint stratified 5-fold (patient_id yok -> kayit bazli)."""
    from sklearn.model_selection import StratifiedKFold
    df = pd.read_csv(manifest_csv).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold = np.full(len(df), -1, dtype=int)
    ybin = (df["label"] == "abnormal").astype(int).values
    for fi, (_, val_idx) in enumerate(skf.split(df, ybin)):
        fold[val_idx] = fi
    df["fold"] = fold
    df.to_csv(out_csv, index=False)
    return df


def cinc_fold_manifest(full_df, fold, split, tmp_dir):
    """Bir fold icin train/val alt-manifest yaz (CinCDataset bunu okur)."""
    if split == "train":
        sub = full_df[full_df["fold"] != fold]
    else:
        sub = full_df[full_df["fold"] == fold]
    p = tmp_dir / f"cinc_fold{fold}_{split}.csv"
    sub.to_csv(p, index=False)
    return p


def binary_sampler_from_cinc(ds, beta=0.5):
    """CinCDataset.index -> ikili etiketlerden weighted sampler."""
    bl = np.array([e[2] for e in ds.index], dtype=int)   # (ri,si,lab,rid)
    c = np.bincount(bl, minlength=2).astype(float)
    cw = (c.sum() / (2 * np.maximum(c, 1))) ** beta
    return WeightedRandomSampler(torch.DoubleTensor(cw[bl]), num_samples=len(bl),
                                 replacement=True), cw


def bootstrap_ci(y, p, n_boot=2000, seed=42):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        a = roc_auc_score(y, p); return a, a, a
    bs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        bs[b] = roc_auc_score(y[idx], p[idx])
    return roc_auc_score(y, p), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def sens_spec(y, p, thr=0.5):
    cm = confusion_matrix(y, (np.asarray(p) >= thr).astype(int), labels=[0, 1])
    return cm[1, 1] / max(cm[1].sum(), 1), cm[0, 0] / max(cm[0].sum(), 1)


def train_on_cinc(args, mdl, full_df, tmp_dir, ckpt_dir, root, log):
    """CinC 5-fold egitim (val = tutulan CinC fold). Orijinali ezmez."""
    need_cwt = (mdl == "mambaconformer")
    log(f"\n=== TERS-YON EGITIM (CinC uzerinde): {mdl} ===")
    for fold in range(N_FOLDS):
        t0 = time.time()
        ck = ckpt_dir / f"{mdl}_cinc_fold{fold}.pth"
        if ck.exists():
            log(f"  fold {fold}: checkpoint var, atla"); continue
        tr_man = cinc_fold_manifest(full_df, fold, "train", tmp_dir)
        va_man = cinc_fold_manifest(full_df, fold, "val", tmp_dir)
        tr = ev.CinCDataset(tr_man, root, norm_adapt="none")
        va = ev.CinCDataset(va_man, root, norm_adapt="none")
        sampler, cw = binary_sampler_from_cinc(tr, args.sampler_beta)
        tl = DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        vl = DataLoader(va, batch_size=128, shuffle=False, num_workers=args.num_workers)

        model, mode = ev.build_binary_model(mdl); model = model.to(DEVICE)
        loss_w = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, ev.lr_lambda_factory(args.warmup, args.epochs))

        best_auroc, bad = -1.0, 0
        ema = deque(maxlen=3); best_ema = -1.0
        for ep in range(1, args.epochs + 1):
            # circor=False: CinC etiketleri ZATEN ikili (donusum yok)
            ev.run_epoch(model, tl, crit, opt, mode, args.clip, circor=False)
            sched.step()
            rp, ry, _ = ev.predict_grouped(model, vl, mode, "record_id", circor=False)
            au = me_mod.binary_metrics(ry, rp)["auroc"]
            ema.append(au); ema_v = float(np.mean(ema))
            if au > best_auroc:
                best_auroc = au
                torch.save({"model": model.state_dict(), "epoch": ep, "auroc": au}, ck)
            if ema_v > best_ema + 1e-4: best_ema = ema_v; bad = 0
            else:
                bad += 1
                if bad >= args.patience: break
        log(f"  fold {fold}: ic-val(CinC) AUROC={best_auroc:.4f} ({time.time()-t0:.0f}s)")


@torch.no_grad()
def eval_external(mdl, ckpt_dir, target, root, nw):
    """CinC-egitimli 5-fold ensemble -> hedef (circor tum / yaseen)."""
    need_cwt = (mdl == "mambaconformer")
    if target == "circor":
        cv = root / "splits" / "circor2022_5fold_cv.csv"
        ds = ds_mod.CircorSegmentDataset(cv, project_root=root, split="all")
        if need_cwt: ds.return_cwt = True
        loader = ds_mod.make_eval_loader(ds, batch_size=128, num_workers=nw)
        id_key, circor = "patient_id", True
    else:  # yaseen
        man = root / "manifests" / "yaseen_processed.csv"
        ds = ev.YaseenDataset(man, root, norm_adapt="none")
        if need_cwt: ds.return_cwt = True
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=nw)
        id_key, circor = "record_id", False

    fold_p, ref_y = [], None
    for fold in range(N_FOLDS):
        ck = ckpt_dir / f"{mdl}_cinc_fold{fold}.pth"
        if not ck.exists(): continue
        model, mode = ev.build_binary_model(mdl); model = model.to(DEVICE)
        model.load_state_dict(torch.load(ck, map_location=DEVICE)["model"])
        rp, ry, _ = ev.predict_grouped(model, loader, mode, id_key, circor=circor)
        fold_p.append(rp); ref_y = ry
    if not fold_p: return None
    ens = np.mean(fold_p, 0)
    return ens, ref_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_bilstm")
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
    root = Path(args.root); mdl = args.model; nw = args.num_workers

    tmp_dir = root / "splits" / "_reciprocal_tmp"; tmp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = root / "checkpoints" / "reciprocal_cinc"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir = root / "reports"; report_dir.mkdir(exist_ok=True)
    logs = []
    def log(m=""): print(m, flush=True); logs.append(str(m))

    log("#"*70)
    log(f"# RECIPROCAL TRANSFER — CinC (egit) -> CirCor + Yaseen (test) | {mdl}")
    log(f"# Orijinal checkpoint'ler korunur; ters-yon -> {ckpt_dir}")
    log("#"*70)

    # 1) CinC kayit-disjoint 5-fold
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
    fold_csv = root / "splits" / "cinc2016_5fold_train.csv"
    full_df = make_cinc_folds(cinc_manifest, fold_csv)
    log(f"\nCinC egitim seti: {len(full_df)} kayit | fold dagilimi:")
    log(pd.crosstab(full_df["fold"], full_df["label"]).to_string())

    # 2) CinC uzerinde egit
    train_on_cinc(args, mdl, full_df, tmp_dir, ckpt_dir, root, log)

    # 3) ters-yon harici degerlendirme
    log("\n" + "="*70)
    log(f"TERS-YON SONUClAR — {mdl} (CinC-egitimli 5-fold ensemble)")
    log("="*70)
    out = {"model": mdl, "forward": FORWARD.get(mdl, {}), "reverse": {}}
    for tgt, name in [("circor", "CirCor (tum 2531 kayit)"), ("yaseen", "Yaseen")]:
        r = eval_external(mdl, ckpt_dir, tgt, root, nw)
        if r is None:
            log(f"  {name}: checkpoint yok"); continue
        ens, y = r
        au, lo, hi = bootstrap_ci(y, ens)
        se, sp = sens_spec(y, ens)
        log(f"  CinC -> {name:24s}: AUROC={au:.3f} [{lo:.3f},{hi:.3f}] | Sens={se:.3f} Spec={sp:.3f} "
            f"(n+={int((y==1).sum())}, n-={int((y==0).sum())})")
        out["reverse"][tgt] = {"auroc": [au, lo, hi], "sens": se, "spec": sp}

    # 4) yon-karsilastirma ozeti
    log("\n" + "-"*70)
    log("YON KARSILASTIRMASI (AUROC)")
    log("-"*70)
    fwd = FORWARD.get(mdl, {})
    log(f"  Ileri  CirCor->CinC  : {fwd.get('cinc','?')}")
    if "circor" in out["reverse"]:
        log(f"  Ters   CinC->CirCor  : {out['reverse']['circor']['auroc'][0]:.3f}")
    log(f"  Ileri  CirCor->Yaseen: {fwd.get('yaseen','?')}")
    if "yaseen" in out["reverse"]:
        log(f"  Ters   CinC->Yaseen  : {out['reverse']['yaseen']['auroc'][0]:.3f}")
    log("\nYorum (verigörünce): CinC->CirCor de zayifsa -> asimetri CirCor'a ozgu degil,")
    log("genel cross-dataset kirilganligi; CinC->Yaseen iyi/kotu ise homojenlik-hedef")
    log("tezini ayrica sinar. Sonucu GORMEDEN yorum yazma.")

    (report_dir / f"reciprocal_{mdl}.json").write_text(json.dumps(out, indent=2))
    (report_dir / f"reciprocal_{mdl}.txt").write_text("\n".join(logs), encoding="utf-8")
    log(f"\n[OK] reports/reciprocal_{mdl}.json + .txt")


if __name__ == "__main__":
    main()
