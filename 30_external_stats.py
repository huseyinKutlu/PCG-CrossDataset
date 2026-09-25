#!/usr/bin/env python3
"""
30_external_stats.py
Dis-test (cross-dataset) AUROC farklarinin ISTATISTIKSEL anlamliligi.

Testler:
  1) Naif vs DANN (cnn_bilstm) — DANN iyilesmesi anlamli mi? (DeLong)
     Hem CinC hem Yaseen.
  2) Model ciftleri: cnn_bilstm vs hybrid_kan dis AUROC farki anlamli mi?
  3) Bootstrap %95 CI her AUROC icin.

DeLong: ayni test setinde iki korelasyonlu AUROC farki icin standart test.
Hasta/record-seviyesi olasiliklar uzerinde.

Kullanim:
  python 30_external_stats.py --ext_dataset cinc
  python 30_external_stats.py --ext_dataset yaseen
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
from math import erf, sqrt

ext_mod = import_module("16_external_val")
PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CKPT_BINARY = PROJECT_ROOT / "checkpoints" / "binary"


# ---------- DeLong (Sun & Xu 2014 fast) ----------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_test(preds_a, preds_b, labels):
    """iki korelasyonlu AUROC farki. return (aucA, aucB, z, p)."""
    labels = np.asarray(labels)
    pos = labels == 1; neg = labels == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([preds_a, preds_b])
    tx = np.array([compute_midrank(preds[r, pos]) for r in range(2)])
    ty = np.array([compute_midrank(preds[r, neg]) for r in range(2)])
    tz = np.array([compute_midrank(preds[r]) for r in range(2)])
    aucs = (tz[:, pos].sum(1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, pos] - tx) / n
    v10 = 1 - (tz[:, neg] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var_diff <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return aucs[0], aucs[1], z, p


def bootstrap_auc_ci(preds, labels, n_boot=2000, seed=42):
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels); preds = np.asarray(preds)
    n = len(labels); aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], preds[idx]))
    return np.percentile(aucs, [2.5, 97.5])


# ---------- model dis-test tahminleri (record-seviyesi, eslesmis) ----------
@torch.no_grad()
def collect_preds(model_name, ckpt_dir, ckpt_pattern, ExtDS, root, manifest, seeds=None):
    """5-fold (ve coklu-seed) ensemble record-seviyesi abnormal-olasiligi.
    Ayni record sirasini (sorted uniq) dondurur — eslestirme icin sart."""
    ds = ExtDS(manifest, root, norm_adapt="none")
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)
    # tum checkpoint'leri topla
    ckpts = []
    if seeds is None:
        for fold in range(N_FOLDS):
            ck = ckpt_dir / ckpt_pattern.format(fold=fold)
            if ck.exists(): ckpts.append(ck)
    else:
        for sd in seeds:
            for fold in range(N_FOLDS):
                ck = ckpt_dir / ckpt_pattern.format(seed=sd, fold=fold)
                if ck.exists(): ckpts.append(ck)
    if not ckpts:
        return None, None, None

    all_p = None; ref_ids = None; ref_lab = None
    for ck in ckpts:
        built = ext_mod.build_binary_model(model_name)
        model, mode = built[0].to(DEVICE), built[1]
        sd = torch.load(ck, map_location=DEVICE)
        if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
        model.load_state_dict(sd); model.eval()
        seg_p, seg_y, seg_id = [], [], []
        for b in loader:
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE)) if mode == "dual" \
                     else model(b[mode].to(DEVICE))
            seg_p.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            seg_y.append(b["label"].numpy())
            seg_id.extend(b["record_id"])
        seg_p = np.concatenate(seg_p); seg_y = np.concatenate(seg_y)
        seg_id = np.array(seg_id)
        uniq = np.unique(seg_id)
        rec_p = np.array([seg_p[seg_id == u].mean() for u in uniq])
        rec_y = np.array([seg_y[seg_id == u][0] for u in uniq])
        all_p = rec_p if all_p is None else all_p + rec_p
        if ref_ids is None: ref_ids, ref_lab = uniq, rec_y
    all_p /= len(ckpts)
    return all_p, ref_lab, ref_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext_dataset", default="cinc", choices=["cinc", "yaseen"])
    ap.add_argument("--model", default="cnn_bilstm")
    ap.add_argument("--dann_seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    if args.ext_dataset == "yaseen":
        manifest = root / "manifests" / "yaseen_processed.csv"
        ExtDS = ext_mod.YaseenDataset
    else:
        manifest = root / "manifests" / "cinc2016_processed.csv"
        ExtDS = ext_mod.CinCDataset

    logs = []
    def log(m=""): print(m); logs.append(str(m))
    log("="*70)
    log(f"DIS-TEST ISTATISTIKSEL ANLAMLILIK | {args.ext_dataset.upper()} | {args.model}")
    log("DeLong (korelasyonlu AUROC farki) + bootstrap %95 CI")
    log("="*70)

    # 1) NAIF tahminleri (checkpoints/binary)
    naif_dir = PROJECT_ROOT / "checkpoints" / "binary"
    naif_p, naif_y, naif_ids = collect_preds(
        args.model, naif_dir, args.model + "_fold{fold}.pth", ExtDS, root, manifest)
    if naif_p is None:
        log("[HATA] naif checkpoint bulunamadi."); _save(logs, args); return

    # 2) DANN tahminleri (checkpoints/dann, coklu-seed)
    dann_dir = PROJECT_ROOT / "checkpoints" / "dann"
    dann_p, dann_y, dann_ids = collect_preds(
        args.model, dann_dir, args.model + "_dann_s{seed}_fold{fold}.pth",
        ExtDS, root, manifest, seeds=args.dann_seeds)
    if dann_p is None:
        log("[HATA] DANN checkpoint bulunamadi. Once DANN'i bu dataset'te egitin.")
        _save(logs, args); return

    # eslestirme kontrolu (ayni record kumesi + sira)
    assert np.array_equal(naif_ids, dann_ids), "record kumeleri eslesmali!"
    labels = naif_y

    from sklearn.metrics import roc_auc_score
    auc_naif = roc_auc_score(labels, naif_p)
    auc_dann = roc_auc_score(labels, dann_p)
    ci_naif = bootstrap_auc_ci(naif_p, labels)
    ci_dann = bootstrap_auc_ci(dann_p, labels)

    log(f"\n--- AUROC + bootstrap %95 CI (n={len(labels)} record) ---")
    log(f"  Naif : AUROC={auc_naif:.4f}  CI=[{ci_naif[0]:.3f}, {ci_naif[1]:.3f}]")
    log(f"  DANN : AUROC={auc_dann:.4f}  CI=[{ci_dann[0]:.3f}, {ci_dann[1]:.3f}]")
    ci_overlap = not (ci_dann[0] > ci_naif[1] or ci_naif[0] > ci_dann[1])
    log(f"  CI ortusmesi: {'EVET (fark zayif)' if ci_overlap else 'HAYIR (fark guclu)'}")

    # DeLong: naif vs DANN (esit record uzerinde)
    a_n, a_d, z, p = delong_test(naif_p, dann_p, labels)
    log(f"\n--- DeLong testi: Naif vs DANN ---")
    log(f"  AUROC naif={a_n:.4f} | DANN={a_d:.4f} | fark={a_d-a_n:+.4f}")
    log(f"  z={z:.3f} | p={p:.4f}  -> {'ANLAMLI (p<0.05)' if p < 0.05 else 'anlamsiz'}")
    if a_d > a_n and p < 0.05:
        log(f"  >> DANN, naif'i ISTATISTIKSEL OLARAK anlamli sekilde gecti.")
    elif a_d > a_n:
        log(f"  >> DANN daha iyi ama fark anlamlilik esigine ulasmadi (orneklem).")

    _save(logs, args)


def _save(logs, args):
    rep = REPORT_DIR / f"external_stats_{args.ext_dataset}.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    print(f"\nRapor: {rep}")


if __name__ == "__main__":
    main()
