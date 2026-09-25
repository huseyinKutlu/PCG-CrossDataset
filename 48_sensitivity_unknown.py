#!/usr/bin/env python3
"""
48_sensitivity_unknown.py
Hakem Yorum 1 — "unknown" etiketi duyarlilik analizi.

Soru: CirCor'daki 131 "unknown" kaydini "present" ile gruplamak (->abnormal)
sonuclari suruluyor mu? Bunu gostermek icin temsili bir modeli (varsayilan
cnn_bilstm) "unknown" KAYITLARI DISLAYARAK yeniden egitiyoruz (yalnizca
Present-vs-Absent), sonra CinC ve Yaseen'de harici AUROC'u olcup ORIJINAL
(unknown-dahil) degerlerle karsilastiriyoruz.

ONEMLI: Orijinal checkpoint'leri EZMEZ. Ayri klasore yazar:
        checkpoints/binary_nounknown/
Orijinal sonuclar (unknown-dahil, makaleden):
   cnn_bilstm : ic(pooled)=0.903 | CinC=0.727 | Yaseen=0.806
   mambaconformer: ic=0.862 | CinC=0.601 | Yaseen=0.985

Kullanim:
   python 48_sensitivity_unknown.py                 # cnn_bilstm
   python 48_sensitivity_unknown.py --model mambaconformer
   python 48_sensitivity_unknown.py --epochs 30
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from importlib import import_module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ev = import_module("16_external_val")
me = import_module("06_metrics")
ds_mod = import_module("05_dataset")
PROJECT_ROOT = Path(__file__).resolve().parent

# orijinal (unknown-dahil) referans degerler — karsilastirma icin
ORIG = {
    "cnn_bilstm":     {"internal": 0.903, "cinc": 0.727, "yaseen": 0.806},
    "mambaconformer": {"internal": 0.862, "cinc": 0.601, "yaseen": 0.985},
    "cnn1d":          {"internal": 0.832, "cinc": 0.513, "yaseen": 0.891},
    "hybrid_kan":     {"internal": 0.900, "cinc": 0.571, "yaseen": 0.909},
}


def bootstrap_ci(y, p, n_boot=2000, seed=42):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        a = me._auroc(y, p); return a, a, a
    bs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
        bs[b] = me._auroc(y[idx], p[idx])
    return me._auroc(y, p), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def collect(mdl, cv_csv, root, manifest, kind, nw):
    need_cwt = (mdl == "mambaconformer")
    ExtDS = ev.YaseenDataset if kind == "yaseen" else ev.CinCDataset
    ext = ExtDS(manifest, root, norm_adapt="none"); ext.dwt_norm = False
    if need_cwt: ext.return_cwt = True
    el = DataLoader(ext, batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    in_p, in_y, ext_pf, ext_y = [], [], [], None
    for fold in range(ev.N_FOLDS):
        ckpt = ev.CKPT_DIR / f"{mdl}_fold{fold}.pth"
        if not ckpt.exists():
            print(f"  !! eksik: {ckpt}"); return None
        model, mode = ev.build_binary_model(mdl); model = model.to(ev.DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=ev.DEVICE)["model"])
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        if need_cwt: va.return_cwt = True
        vl = ds_mod.make_eval_loader(va, batch_size=128, num_workers=nw)
        rp, ry, _ = ev.predict_grouped(model, vl, mode, "patient_id", circor=True)
        in_p.append(rp); in_y.append(ry)
        cp, cy, _ = ev.predict_grouped(model, el, mode, "record_id", circor=False)
        ext_pf.append(cp); ext_y = cy
    return (np.concatenate(in_p), np.concatenate(in_y)), (np.mean(ext_pf, axis=0), ext_y)


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
    ap.add_argument("--norm_adapt", default="none")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--dwt_norm", action="store_true")
    ap.add_argument("--ext_dataset", default="cinc")
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    mdl = args.model
    nw = args.num_workers

    cv_orig = root / "splits" / "circor2022_5fold_cv.csv"
    cv_nou = root / "splits" / "circor2022_5fold_cv_nounknown.csv"
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
    yaseen_manifest = root / "manifests" / "yaseen_processed.csv"

    # 1) unknown'i disla, yeni CV csv yaz
    df = pd.read_csv(cv_orig)
    n0 = len(df)
    df2 = df[df["murmur"] != "Unknown"].reset_index(drop=True)
    df2.to_csv(cv_nou, index=False)
    print(f"[unknown disla] {n0} -> {len(df2)} kayit ({n0-len(df2)} unknown cikarildi)")
    print(f"  yeni murmur dagilimi: {df2['murmur'].value_counts().to_dict()}")

    # 2) checkpoint'leri AYRI klasore yaz (orijinali ezme!)
    orig_ckpt = ev.CKPT_DIR
    ev.CKPT_DIR = root / "checkpoints" / "binary_nounknown"
    ev.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] sensitivity modelleri -> {ev.CKPT_DIR} (orijinal {orig_ckpt} korunur)")

    logs = []
    def log(m=""): print(m); logs.append(str(m))

    # 3) yeniden egit (unknown'siz)
    log(f"\n=== YENIDEN EGITIM (unknown'siz): {mdl} ===")
    ev.train_model_cv(args, mdl, cv_nou, root, log)

    # 4) harici degerlendir (unknown'siz CV ile ic; ayni harici setler)
    cinc = collect(mdl, cv_nou, root, cinc_manifest, "cinc", nw)
    yas = collect(mdl, cv_nou, root, yaseen_manifest, "yaseen", nw)
    if cinc is None or yas is None:
        log("  degerlendirme atlandi"); return
    (in_p, in_y), (cp, cy) = cinc
    (_, _), (yp, yy) = yas

    ia, ilo, ihi = bootstrap_ci(in_y, in_p)
    ca, clo, chi = bootstrap_ci(cy, cp)
    ya, ylo, yhi = bootstrap_ci(yy, yp)

    o = ORIG.get(mdl, {})
    log("\n" + "="*72)
    log(f"DUYARLILIK ANALIZI — {mdl} (unknown DISLANMIS vs orijinal)")
    log("="*72)
    log(f"{'Hedef':<12}{'Orijinal':<12}{'Unknown-siz (95% GA)':<30}{'Fark'}")
    for tgt, (val, lo, hi) in [("Ic", (ia, ilo, ihi)), ("CinC", (ca, clo, chi)), ("Yaseen", (ya, ylo, yhi))]:
        key = {"Ic": "internal", "CinC": "cinc", "Yaseen": "yaseen"}[tgt]
        ov = o.get(key, float("nan"))
        log(f"{tgt:<12}{ov:<12.3f}{val:.3f} [{lo:.3f},{hi:.3f}]{'':<8}{val-ov:+.3f}")
    log("="*72)
    log("Yorum: farklar kucukse (orn. |Δ|<0.03 ve GA'lar orijinali iceriyorsa),")
    log("'unknown' gruplamasi sonuclari SURUKLEMIYOR demektir (Yorum 1 kapanir).")

    out = {"model": mdl, "orig": o,
           "nounknown": {"internal": [ia, ilo, ihi], "cinc": [ca, clo, chi], "yaseen": [ya, ylo, yhi]},
           "n_excluded": int(n0 - len(df2))}
    (ev.REPORT_DIR / f"sensitivity_unknown_{mdl}.json").write_text(json.dumps(out, indent=2))
    (ev.REPORT_DIR / f"sensitivity_unknown_{mdl}.txt").write_text("\n".join(logs), encoding="utf-8")
    print(f"\n[OK] reports/sensitivity_unknown_{mdl}.json + .txt")


if __name__ == "__main__":
    main()
