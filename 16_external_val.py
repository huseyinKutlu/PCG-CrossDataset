#!/usr/bin/env python3
"""
16_external_val.py
PCG Q1 Manuscript — Cross-dataset (zero-shot) dis validasyon.

ANA FIKIR: Modeli CirCor'da IKILI (normal/abnormal) egit, CinC 2016'yi
HIC GORMEDEN test et. Ic (CirCor) vs dis (CinC) + degradation.

Ikili etiket:
  CirCor: Present+Unknown -> abnormal(1) ; Absent -> normal(0)
  CinC  : abnormal->1 ; normal->0

Hipotez: Mamba'li hibrit domain shift'e daha dayanikli (az degradation).

Asamalar:
  --train : CirCor ikili 5-fold egit (model basina 5 checkpoint)
  --eval  : CirCor ic-test (val foldlari) + CinC zero-shot dis-test

Kullanim:
  python 16_external_val.py --model all --epochs 30 --train
  python 16_external_val.py --model all --eval
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
aug_mod = import_module("17_augment")
_dwt_mod = import_module("24_dwt_transform")

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "binary"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CIRCOR_TO_BINARY = {0: 1, 1: 1, 2: 0}   # P,U->abnormal ; Absent->normal


# ============================================================
# CinC dis-test dataset
# ============================================================
class CinCDataset(Dataset):
    def __init__(self, manifest_csv, project_root, norm_adapt="none",
                 cir_mel_mean=-26.86, cir_mel_std=10.27,
                 cinc_mel_mean=-35.38, cinc_mel_std=14.45):
        self.root = Path(project_root)
        self.norm_adapt = norm_adapt   # none | dataset | instance
        self.cir_mm, self.cir_ms = cir_mel_mean, cir_mel_std
        self.cinc_mm, self.cinc_ms = cinc_mel_mean, cinc_mel_std
        df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.records = df
        self.index = []
        for ri, row in df.iterrows():
            lab = 1 if row["label"] == "abnormal" else 0
            for si in range(int(row["n_segments"])):
                self.index.append((ri, si, lab, str(row["record_id"])))

    def __len__(self): return len(self.index)

    def _adapt_mel(self, mel):
        """Test-ani mel domain adaptasyonu."""
        if self.norm_adapt == "dataset":
            # CinC dagilimini CirCor egitim dagilimina hizala
            z = (mel - self.cinc_mm) / self.cinc_ms
            return z * self.cir_ms + self.cir_mm
        if self.norm_adapt == "instance":
            # her segment kendi icinde z-score, sonra CirCor olcegine
            m, s = mel.mean(), mel.std() + 1e-6
            z = (mel - m) / s
            return z * self.cir_ms + self.cir_mm
        return mel

    def __getitem__(self, i):
        ri, si, lab, rid = self.index[i]
        row = self.records.iloc[ri]
        d = np.load(self.root / row["cache_path"], allow_pickle=True)
        raw = d["raw_segs"][si].astype(np.float32)
        if getattr(self, "dwt_norm", False):
            raw = _dwt_mod.dwt_band_normalize(raw)
        mel = self._adapt_mel(d["mel_segs"][si].astype(np.float32))
        raw = torch.from_numpy(raw).unsqueeze(0)
        mel = torch.from_numpy(mel).unsqueeze(0)
        out = {"raw": raw, "mel": mel, "label": lab, "record_id": rid}
        if "cwt_segs" in d:   # PCG-MambaConformer 3. dali icin
            cwt = d["cwt_segs"][si].astype(np.float32)
            out["cwt"] = torch.from_numpy(cwt).unsqueeze(0)
        return out


# ============================================================
# Modeller (ikili: n_classes=2)
# ============================================================
def build_binary_model(name):
    if name == "hybrid_kan": return hy_mod.HybridPCG(head="kan", d_model=128, n_classes=2), "dual"
    if name == "hybrid_mlp": return hy_mod.HybridPCG(head="mlp", d_model=128, n_classes=2), "dual"
    if name == "hybrid2_kan": return hy_mod.HybridPCG2(head="kan", d_model=128, n_classes=2), "dual"
    if name == "hybrid2_mlp": return hy_mod.HybridPCG2(head="mlp", d_model=128, n_classes=2), "dual"
    if name == "hybrid3_kan": return hy_mod.HybridPCG3(head="kan", d_model=128, n_classes=2), "dual"
    if name == "hybrid3_mlp": return hy_mod.HybridPCG3(head="mlp", d_model=128, n_classes=2), "dual"
    if name == "hybrid4_kan": return hy_mod.HybridPCG4(head="kan", d_model=128, n_classes=2), "dual"
    if name == "hybrid4_mlp": return hy_mod.HybridPCG4(head="mlp", d_model=128, n_classes=2), "dual"
    if name == "cnn2d":      return bl_mod.CNN2D(n_classes=2), "mel"
    if name == "cnn1d":      return bl_mod.CNN1D(n_classes=2), "raw"
    if name == "cnn_bilstm": return bl_mod.CNN_BiLSTM(n_classes=2), "raw"
    if name == "efficientnet":
        tm = import_module("35_timm_models")
        return tm.TimmPCG("efficientnet_b0", n_classes=2, pretrained=True), "mel"
    if name == "maxvit":
        tm = import_module("35_timm_models")
        return tm.TimmPCG("maxvit_tiny_tf_224", n_classes=2, pretrained=True), "mel"
    if name == "mambaconformer":
        mc = import_module("37_mambaconformer")
        return mc.PCGMambaConformer(n_classes=2, pretrained=True), "tri"
    if name == "resnet1d":
        rn = import_module("39_resnet1d")
        return rn.ResNet1D(n_classes=2), "raw"
    raise ValueError(name)


def lr_lambda_factory(warmup, total):
    def f(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        prog = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))
    return f


# ============================================================
# CirCor ikili sampler (3-sinif datasetin seg_labels'ini ikiliye cevir)
# ============================================================
def binary_seg_labels(base_ds):
    return np.array([CIRCOR_TO_BINARY[int(l)] for l in base_ds.seg_labels], dtype=np.int64)


class YaseenDataset(CinCDataset):
    """Yaseen ikinci dis dataset. CinCDataset ile ayni .npz formati;
    ek olarak class5 (AS/MR/MS/MVP/N) dondurur — hastalik-spesifik analiz icin."""
    def __init__(self, manifest_csv, project_root, norm_adapt="none",
                 yas_mel_mean=None, yas_mel_std=None):
        super().__init__(manifest_csv, project_root, norm_adapt=norm_adapt)
        # class5 indexini de sakla
        self.class5_index = []
        for ri, row in self.records.iterrows():
            c5 = str(row.get("class5", "?"))
            for si in range(int(row["n_segments"])):
                self.class5_index.append(c5)

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["class5"] = self.class5_index[i]
        return item


class AugmentedCircor(Dataset):
    """CirCor egitim datasetini on-the-fly augmentation ile sarar.
    Sadece egitimde kullanilir; raw/mel augmente edilir, etiket korunur."""
    def __init__(self, base_ds, augmenter):
        self.base = base_ds
        self.aug = augmenter
        self.seg_labels = base_ds.seg_labels   # sampler icin

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        raw = item["raw"].numpy()   # (1,10000)
        mel = item["mel"].numpy()   # (1,64,201)
        raw_a, mel_a = self.aug(raw, mel)
        item["raw"] = torch.from_numpy(raw_a).unsqueeze(0)   # (1,10000)
        item["mel"] = torch.from_numpy(mel_a).unsqueeze(0)   # (1,64,201)
        return item


class DWTCircor(Dataset):
    """CirCor egitimini DWT bant-normalizasyonu ile sarar (raw uzerinde).
    Hem CirCor hem CinC ayni domain-robust temsile gelir -> domain bant-farki silinir."""
    def __init__(self, base_ds):
        self.base = base_ds
        self.seg_labels = base_ds.seg_labels
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        item = dict(self.base[i])
        raw = item["raw"].numpy().squeeze()       # (10000,)
        raw_t = _dwt_mod.dwt_band_normalize(raw)
        item["raw"] = torch.from_numpy(raw_t).unsqueeze(0)
        return item


def make_binary_sampler(base_ds, beta=0.5):
    bl = binary_seg_labels(base_ds)
    c = np.bincount(bl, minlength=2).astype(float)
    cw = (c.sum() / (2 * np.maximum(c, 1))) ** beta
    return WeightedRandomSampler(torch.DoubleTensor(cw[bl]), num_samples=len(bl),
                                 replacement=True), cw


# ============================================================
# Egitim / tahmin (etiketler runtime'da ikiliye cevrilir)
# ============================================================
def to_binary_labels(y3):
    """3-sinif label tensorunu ikiliye cevir."""
    out = y3.clone()
    for k, v in CIRCOR_TO_BINARY.items():
        out[y3 == k] = v
    return out


def run_epoch(model, loader, crit, opt, mode, clip=0.5, circor=True):
    model.train(); tot, n, skip = 0.0, 0, 0
    for b in loader:
        y = to_binary_labels(b["label"]).to(DEVICE) if circor else b["label"].to(DEVICE)
        opt.zero_grad()
        if mode == "tri":
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE), b["cwt"].to(DEVICE))
        elif mode == "dual":
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
def predict_grouped(model, loader, mode, id_key, circor=True):
    """Segment abnormal-olasiliklarini id (hasta/record) bazinda ortala.
    Doner: (record_p_abnormal, record_true_label, ids)."""
    model.eval(); seg_pa, ids, seg_y = [], [], []
    for b in loader:
        if mode == "tri":
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE), b["cwt"].to(DEVICE))
        elif mode == "dual":
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE))
        else:
            logits = model(b[mode].to(DEVICE))
        prob = torch.softmax(logits, 1).float().cpu().numpy()
        seg_pa.append(prob[:, 1])    # abnormal olasiligi
        ids.extend(b[id_key])
        y = to_binary_labels(b["label"]) if circor else b["label"]
        seg_y.extend(y.numpy().tolist())
    seg_pa = np.concatenate(seg_pa)
    ids = np.array(ids); seg_y = np.array(seg_y)
    uniq = np.unique(ids)
    rec_p = np.array([seg_pa[ids == u].mean() for u in uniq])
    rec_y = np.array([seg_y[ids == u][0] for u in uniq])
    return rec_p, rec_y, uniq


# ============================================================
# Egitim asamasi
# ============================================================
def train_model_cv(args, mdl, cv_csv, root, log):
    log(f"\n=== EGITIM (ikili): {mdl} ===")
    need_cwt = (mdl == "mambaconformer")   # tri-mod modeller CWT ister
    for fold in range(N_FOLDS):
        t0 = time.time()
        tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="train")
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        if need_cwt:
            tr.return_cwt = True; va.return_cwt = True
        sampler, cw = make_binary_sampler(tr, args.sampler_beta)
        # augmentation (yalnizca egitim) — sampler ICIN orijinal tr'nin seg_labels'i lazim
        train_ds = tr
        if args.augment:
            augmenter = aug_mod.PCGAugment(seed=fold)   # fold basina farkli seed
            train_ds = AugmentedCircor(tr, augmenter)
        if args.dwt_norm:
            train_ds = DWTCircor(train_ds)   # DWT en dista (augment'ten sonra)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)

        model, mode = build_binary_model(mdl); model = model.to(DEVICE)
        loss_w = torch.tensor(cw / cw.mean(), dtype=torch.float32).to(DEVICE)
        crit = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=args.label_smooth)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(args.warmup, args.epochs))

        best_auroc, bad = -1.0, 0
        ema = deque(maxlen=3); best_ema = -1.0
        aug_sfx = "_aug" if args.augment else ""
        dwt_sfx = "_dwt" if args.dwt_norm else ""
        ckpt = CKPT_DIR / f"{mdl}{aug_sfx}{dwt_sfx}_fold{fold}.pth"
        for ep in range(1, args.epochs + 1):
            run_epoch(model, train_loader, crit, opt, mode, args.clip, circor=True)
            sched.step()
            rec_p, rec_y, _ = predict_grouped(model, val_loader, mode, "patient_id", circor=True)
            bm = me_mod.binary_metrics(rec_y, rec_p)
            ema.append(bm["auroc"]); ema_v = float(np.mean(ema))
            if bm["auroc"] > best_auroc:
                best_auroc = bm["auroc"]
                torch.save({"model": model.state_dict(), "epoch": ep, "auroc": bm["auroc"]}, ckpt)
            if ema_v > best_ema + 1e-4: best_ema = ema_v; bad = 0
            else:
                bad += 1
                if bad >= args.patience: break
        log(f"  fold {fold}: ic-val AUROC={best_auroc:.4f} ({time.time()-t0:.0f}s)")


# ============================================================
# Degerlendirme: ic (CirCor val) + dis (CinC zero-shot)
# ============================================================
def eval_model(args, mdl, cv_csv, cinc_manifest, root, log):
    # Dis-test loader (CinC veya Yaseen) — norm_adapt ile
    ExtDS = YaseenDataset if args.ext_dataset == "yaseen" else CinCDataset
    cinc = ExtDS(cinc_manifest, root, norm_adapt=args.norm_adapt)
    cinc.dwt_norm = args.dwt_norm   # DWT bant-normalizasyonu (test tarafi)
    cinc_loader = DataLoader(cinc, batch_size=128, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    internal_auroc, internal_bacc = [], []
    external_auroc, external_bacc = [], []
    # ic: her fold'un val'inde; dis: her fold modeliyle CinC, sonra ortala
    cinc_probs_per_fold = []; cinc_y = None
    for fold in range(N_FOLDS):
        aug_sfx = "_aug" if args.augment else ""
        dwt_sfx = "_dwt" if args.dwt_norm else ""
        ckpt = CKPT_DIR / f"{mdl}{aug_sfx}{dwt_sfx}_fold{fold}.pth"
        if not ckpt.exists():
            log(f"  !! eksik checkpoint {ckpt.name} — once --train"); return None
        model, mode = build_binary_model(mdl); model = model.to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])

        # ic-test (CirCor val fold)
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        if mdl == "mambaconformer":
            va.return_cwt = True
        vl = ds_mod.make_eval_loader(va, batch_size=128, num_workers=args.num_workers)
        rp, ry, _ = predict_grouped(model, vl, mode, "patient_id", circor=True)
        bm_in = me_mod.binary_metrics(ry, rp)
        internal_auroc.append(bm_in["auroc"]); internal_bacc.append(bm_in["bacc"])

        # dis-test (CinC zero-shot)
        cp, cy, _ = predict_grouped(model, cinc_loader, mode, "record_id", circor=False)
        cinc_probs_per_fold.append(cp); cinc_y = cy
        bm_ex = me_mod.binary_metrics(cy, cp)
        external_auroc.append(bm_ex["auroc"]); external_bacc.append(bm_ex["bacc"])

    # 5-fold ensemble dis tahmini (foldlarin ortalamasi)
    cinc_ens = np.mean(cinc_probs_per_fold, axis=0)
    bm_ens = me_mod.binary_metrics(cinc_y, cinc_ens)

    ia, ib = np.array(internal_auroc), np.array(internal_bacc)
    ea, eb = np.array(external_auroc), np.array(external_bacc)
    log(f"\n  {mdl}:")
    log(f"    Ic  (CirCor val) : AUROC={ia.mean():.4f}±{ia.std():.4f} | BAcc={ib.mean():.4f}")
    log(f"    Dis (CinC 0-shot): AUROC={ea.mean():.4f}±{ea.std():.4f} | BAcc={eb.mean():.4f}")
    log(f"    Dis (5-fold ens) : AUROC={bm_ens['auroc']:.4f} | Sens={bm_ens['sensitivity']:.3f} "
        f"| Spec={bm_ens['specificity']:.3f}")
    log(f"    >> Degradation (ic-dis AUROC): {ia.mean()-ea.mean():+.4f} "
        f"(dusuk=iyi genelleme)")
    return {"model": mdl, "in_auroc": ia.mean(), "in_std": ia.std(),
            "ex_auroc": ea.mean(), "ex_std": ea.std(),
            "ens_auroc": bm_ens["auroc"], "degradation": ia.mean() - ea.mean(),
            "ens_sens": bm_ens["sensitivity"], "ens_spec": bm_ens["specificity"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hybrid_kan",
                    choices=["hybrid_kan","hybrid_mlp","hybrid2_kan","hybrid2_mlp",
                             "hybrid3_kan","hybrid3_mlp","hybrid4_kan","hybrid4_mlp","cnn2d","cnn1d","cnn_bilstm",
                             "efficientnet","maxvit","mambaconformer","resnet1d","all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sampler_beta", type=float, default=0.5)
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--norm_adapt", default="none",
                    choices=["none", "dataset", "instance"],
                    help="CinC test-ani mel domain adaptasyonu")
    ap.add_argument("--augment", action="store_true",
                    help="egitimde on-the-fly PCG augmentation (domain dayaniklilik)")
    ap.add_argument("--dwt_norm", action="store_true",
                    help="DWT alt-bant normalizasyonu (domain-robust temsil, hem train hem test)")
    ap.add_argument("--ext_dataset", default="cinc", choices=["cinc", "yaseen"],
                    help="dis-test dataseti: cinc (yetiskin/karisik) veya yaseen (5 kapak hastaligi)")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()
    if not (args.train or args.eval): args.train = args.eval = True

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    if args.ext_dataset == "yaseen":
        cinc_manifest = root / "manifests" / "yaseen_processed.csv"
    else:
        cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
    models = (["cnn1d","cnn2d","cnn_bilstm","hybrid_mlp","hybrid_kan"]
              if args.model == "all" else [args.model])

    logs = []
    def log(m=""): print(m); logs.append(str(m))
    log("="*70)
    log("CROSS-DATASET DIS VALIDASYON (ikili normal/abnormal)")
    log(f"Egit: CirCor | Zero-shot test: {args.ext_dataset.upper()} | norm_adapt={args.norm_adapt} | augment={args.augment} | dwt_norm={args.dwt_norm}")
    log("="*70)

    if args.train:
        for mdl in models:
            train_model_cv(args, mdl, cv_csv, root, log)

    if args.eval:
        log("\n" + "="*70); log("SONUCLAR: ic vs dis (degradation)"); log("="*70)
        rows = []
        for mdl in models:
            r = eval_model(args, mdl, cv_csv, cinc_manifest, root, log)
            if r: rows.append(r)
        if rows:
            log("\n" + "="*70); log("OZET TABLO"); log("="*70)
            log(f"  {'model':12s} | {'ic AUROC':14s} | {'dis AUROC':14s} | {'degr.':8s} | dis Sens/Spec")
            log("  " + "-"*72)
            for r in rows:
                log(f"  {r['model']:12s} | {r['in_auroc']:.3f}±{r['in_std']:.3f}   | "
                    f"{r['ex_auroc']:.3f}±{r['ex_std']:.3f}   | {r['degradation']:+.3f}  | "
                    f"{r['ens_sens']:.2f}/{r['ens_spec']:.2f}")

    rep = REPORT_DIR / f"external_{'all' if len(models)>1 else models[0]}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}")
    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
