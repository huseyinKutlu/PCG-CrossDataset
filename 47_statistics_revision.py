#!/usr/bin/env python3
"""
47_statistics_revision.py
Hakem revizyonu — istatistiksel raporlama (Yorum 4, 5, 6).

Uretir:
  (1) Her model icin AUROC + %95 GA (stratified bootstrap, 2000 resample)
      - Ic (CirCor val, hasta-duzeyi, 5-fold birlesik)
      - CinC (5-fold ensemble, record-duzeyi)
      - Yaseen (5-fold ensemble, record-duzeyi)
  (2) Sens/Spec + Wilson %95 GA — ic, CinC ve Yaseen (Yorum 5, 6)
  (3) DeLong testleri (Yorum 4):
      - Yaseen'de PCG-MambaConformer vs her rakip
      - CinC'de en iyi modeller ikili
      - Holm-Bonferroni cokluluk duzeltmesi
  (4) "—" girdilerini netlestiren tani (hangi model dejenere)

Cikti:
  reports/stats_revision.json  (makaleye islemek icin)
  reports/stats_revision.txt   (insan-okur ozet)

Kullanim:
  python 47_statistics_revision.py
  python 47_statistics_revision.py --models cnn1d,cnn_bilstm,mambaconformer
  python 47_statistics_revision.py --n_boot 2000
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader

ev = import_module("16_external_val")
me = import_module("06_metrics")
ds_mod = import_module("05_dataset")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = ev.DEVICE
N_FOLDS = ev.N_FOLDS
CKPT_DIR = ev.CKPT_DIR
REPORT_DIR = ev.REPORT_DIR

# makaledeki 11 yaklasimin model anahtarlari (klasik ML haric; onlar ayri)
DEEP_MODELS = ["cnn1d", "cnn2d", "resnet1d", "cnn_bilstm",
               "hybrid_mlp", "hybrid_kan", "efficientnet", "maxvit",
               "mambaconformer"]


# --------------------------------------------------- olasilik toplama
def collect_probs(mdl, cv_csv, root, ext_manifest, ext_kind, num_workers=8):
    """Bir model icin ic (val, hasta) + harici (record, 5-fold ensemble)
    kayit-duzeyi olasilik ve gercek etiketleri dondur.

    Doner: dict(
       internal=(p, y),         # 5 fold'un val tahminleri BIRLESTIRILMIS
       external=(p_ens, y),     # 5-fold ensemble (ortalama)
    )
    """
    need_cwt = (mdl == "mambaconformer")
    # harici loader
    ExtDS = ev.YaseenDataset if ext_kind == "yaseen" else ev.CinCDataset
    ext = ExtDS(ext_manifest, root, norm_adapt="none")
    ext.dwt_norm = False
    if need_cwt:
        ext.return_cwt = True
    ext_loader = DataLoader(ext, batch_size=128, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    in_p_all, in_y_all = [], []
    ext_probs_per_fold, ext_y = [], None

    for fold in range(N_FOLDS):
        ckpt = CKPT_DIR / f"{mdl}_fold{fold}.pth"
        if not ckpt.exists():
            print(f"  !! eksik checkpoint: {ckpt.name}")
            return None
        model, mode = ev.build_binary_model(mdl)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model"])

        # ic (CirCor val fold, hasta-duzeyi)
        va = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=fold, split="val")
        if need_cwt:
            va.return_cwt = True
        vl = ds_mod.make_eval_loader(va, batch_size=128, num_workers=num_workers)
        rp, ry, _ = ev.predict_grouped(model, vl, mode, "patient_id", circor=True)
        in_p_all.append(rp); in_y_all.append(ry)

        # harici (record-duzeyi)
        cp, cy, _ = ev.predict_grouped(model, ext_loader, mode, "record_id", circor=False)
        ext_probs_per_fold.append(cp); ext_y = cy

    internal = (np.concatenate(in_p_all), np.concatenate(in_y_all))
    external = (np.mean(ext_probs_per_fold, axis=0), ext_y)
    return {"internal": internal, "external": external}


# --------------------------------------------------- AUROC + bootstrap GA
def auroc(y, p):
    return me._auroc(np.asarray(y).astype(int), np.asarray(p).astype(float))


def bootstrap_auroc_ci(y, p, n_boot=2000, seed=42, alpha=0.05):
    """Stratified bootstrap %95 GA (sinif-katmanli yeniden ornekleme)."""
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        a = auroc(y, p); return a, a, a
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pi = rng.choice(pos_idx, len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        boots[b] = auroc(y[idx], p[idx])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return auroc(y, p), float(lo), float(hi)


# --------------------------------------------------- Wilson GA (sens/spec)
def wilson_ci(k, n, z=1.96):
    """Oran k/n icin Wilson %95 GA."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return phat, max(0.0, centre - half), min(1.0, centre + half)


def sens_spec_wilson(y, p, thr=0.5):
    """0.5 esikte sens/spec + Wilson GA; sayisal payda/pay da dondur."""
    y = np.asarray(y).astype(int); pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens, slo, shi = wilson_ci(tp, tp + fn)
    spec, plo, phi = wilson_ci(tn, tn + fp)
    return {"sens": sens, "sens_ci": [slo, shi], "sens_n": [tp, tp + fn],
            "spec": spec, "spec_ci": [plo, phi], "spec_n": [tn, tn + fp]}


# --------------------------------------------------- DeLong testi
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2


def delong_var(y, p):
    """Tek bir modelin AUROC ve DeLong varyansi (kovaryans icin yapi)."""
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    pos = p[y == 1]; neg = p[y == 0]
    return pos, neg


def delong_test(y, p1, p2):
    """Iki AUROC'un esitligi (ayni y, eslesmis). z ve iki-yonlu p dondur.
    Implementasyon: DeLong (1988) hizli midrank (Sun & Xu 2014)."""
    y = np.asarray(y).astype(int)
    pos = y == 1; neg = ~pos
    m = int(pos.sum()); n = int(neg.sum())
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])  # (2, N)
    X = preds[:, pos]; Y = preds[:, neg]   # (2,m),(2,n)
    k = 2
    tx = np.array([_compute_midrank(X[r]) for r in range(k)])
    ty = np.array([_compute_midrank(Y[r]) for r in range(k)])
    txy = np.array([_compute_midrank(np.concatenate([X[r], Y[r]])) for r in range(k)])
    aucs = np.array([(txy[r, :m].sum() - m * (m + 1) / 2) / (m * n) for r in range(k)])
    v01 = (txy[:, :m] - tx) / n
    v10 = 1 - (txy[:, m:] - ty) / m
    s01 = np.cov(v01); s10 = np.cov(v10)
    S = s01 / m + s10 / n
    S = np.atleast_2d(S)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0] - aucs[1], 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    from math import erf, sqrt
    pval = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(z), float(pval)


def holm_correction(pairs):
    """pairs: list of (label, p). Holm-Bonferroni duzeltilmis p dondur."""
    valid = [(lab, pv) for lab, pv in pairs if pv == pv]  # nan ele
    order = sorted(range(len(valid)), key=lambda i: valid[i][1])
    m = len(valid); adj = {}
    prev = 0.0
    for rank, idx in enumerate(order):
        lab, pv = valid[idx]
        a = min(1.0, (m - rank) * pv)
        a = max(a, prev); prev = a
        adj[lab] = a
    return adj


# --------------------------------------------------- ana
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEEP_MODELS))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    cinc_manifest = root / "manifests" / "cinc2016_processed.csv"
    yaseen_manifest = root / "manifests" / "yaseen_processed.csv"
    models = args.models.split(",")

    results = {}
    store_probs = {}   # DeLong icin: model -> {'yaseen':(p,y), 'cinc':(p,y)}

    for mdl in models:
        print(f"\n=== {mdl} ===")
        cinc = collect_probs(mdl, cv_csv, root, cinc_manifest, "cinc", args.num_workers)
        yas = collect_probs(mdl, cv_csv, root, yaseen_manifest, "yaseen", args.num_workers)
        if cinc is None or yas is None:
            print(f"  atlandi (eksik checkpoint)"); continue

        in_p, in_y = cinc["internal"]   # ic ayni (CirCor), cinc'ten al
        cinc_p, cinc_y = cinc["external"]
        yas_p, yas_y = yas["external"]

        # AUROC + bootstrap GA
        ia, ilo, ihi = bootstrap_auroc_ci(in_y, in_p, args.n_boot)
        ca, clo, chi = bootstrap_auroc_ci(cinc_y, cinc_p, args.n_boot)
        ya, ylo, yhi = bootstrap_auroc_ci(yas_y, yas_p, args.n_boot)

        # sens/spec + Wilson — ic, cinc, yaseen
        in_ss = sens_spec_wilson(in_y, in_p)
        cinc_ss = sens_spec_wilson(cinc_y, cinc_p)
        yas_ss = sens_spec_wilson(yas_y, yas_p)

        results[mdl] = {
            "internal": {"auroc": ia, "auroc_ci": [ilo, ihi], **in_ss},
            "cinc":     {"auroc": ca, "auroc_ci": [clo, chi], **cinc_ss},
            "yaseen":   {"auroc": ya, "auroc_ci": [ylo, yhi], **yas_ss},
        }
        store_probs[mdl] = {"yaseen": (yas_p, yas_y), "cinc": (cinc_p, cinc_y)}
        print(f"  ic AUROC={ia:.3f} [{ilo:.3f},{ihi:.3f}] | "
              f"CinC={ca:.3f} [{clo:.3f},{chi:.3f}] | Yaseen={ya:.3f} [{ylo:.3f},{yhi:.3f}]")
        print(f"  Yaseen Sens={yas_ss['sens']:.3f} {yas_ss['sens_ci']} "
              f"Spec={yas_ss['spec']:.3f} {yas_ss['spec_ci']}")
        print(f"  CinC   Sens={cinc_ss['sens']:.3f} {cinc_ss['sens_ci']} "
              f"Spec={cinc_ss['spec']:.3f} {cinc_ss['spec_ci']}")

    # ---- DeLong: Yaseen'de mambaconformer vs her rakip
    delong = {"yaseen_vs_mambaconformer": {}, "cinc_top": {}}
    if "mambaconformer" in store_probs:
        yp_ref, yy_ref = store_probs["mambaconformer"]["yaseen"]
        pairs = []
        for mdl in store_probs:
            if mdl == "mambaconformer": continue
            yp, yy = store_probs[mdl]["yaseen"]
            # ayni kayit kumesi varsayimi (ayni Yaseen, ayni sira)
            z, pv = delong_test(yy_ref, yp_ref, yp)
            delong["yaseen_vs_mambaconformer"][mdl] = {"z": z, "p": pv}
            pairs.append((mdl, pv))
        adj = holm_correction(pairs)
        for mdl, a in adj.items():
            delong["yaseen_vs_mambaconformer"][mdl]["p_holm"] = a

    # ---- DeLong: CinC'de en iyi iki model (efficientnet vs maxvit vs cnn_bilstm)
    cinc_models = [m for m in ["efficientnet", "maxvit", "cnn_bilstm"] if m in store_probs]
    cpairs = []
    for i in range(len(cinc_models)):
        for j in range(i + 1, len(cinc_models)):
            m1, m2 = cinc_models[i], cinc_models[j]
            p1, y1 = store_probs[m1]["cinc"]; p2, _ = store_probs[m2]["cinc"]
            z, pv = delong_test(y1, p1, p2)
            key = f"{m1}_vs_{m2}"
            delong["cinc_top"][key] = {"z": z, "p": pv}
            cpairs.append((key, pv))
    if cpairs:
        adjc = holm_correction(cpairs)
        for key, a in adjc.items():
            delong["cinc_top"][key]["p_holm"] = a

    out = {"per_model": results, "delong": delong,
           "n_boot": args.n_boot,
           "note": "AUROC GA: stratified bootstrap; Sens/Spec GA: Wilson; DeLong+Holm."}
    (REPORT_DIR / "stats_revision.json").write_text(json.dumps(out, indent=2))

    # ---- insan-okur ozet
    lines = []
    def L(s=""): print(s); lines.append(s)
    L("="*78); L("ISTATISTIK REVIZYON OZETI (Yorum 4-5-6)"); L("="*78)
    L(f"{'Model':<16}{'Ic AUROC (95% GA)':<26}{'CinC AUROC (95% GA)':<26}{'Yaseen AUROC (95% GA)'}")
    for mdl, r in results.items():
        def fmt(d): return f"{d['auroc']:.3f} [{d['auroc_ci'][0]:.3f},{d['auroc_ci'][1]:.3f}]"
        L(f"{mdl:<16}{fmt(r['internal']):<26}{fmt(r['cinc']):<26}{fmt(r['yaseen'])}")
    L("")
    L("--- Sens/Spec (0.5 esik) + Wilson 95% GA ---")
    for mdl, r in results.items():
        for tgt in ("internal", "cinc", "yaseen"):
            d = r[tgt]
            L(f"  {mdl:<16}{tgt:<10} Sens={d['sens']:.3f} [{d['sens_ci'][0]:.3f},{d['sens_ci'][1]:.3f}] "
              f"Spec={d['spec']:.3f} [{d['spec_ci'][0]:.3f},{d['spec_ci'][1]:.3f}]")
    L("")
    L("--- DeLong: Yaseen, MambaConformer vs rakip (Holm duzeltmeli) ---")
    for mdl, d in delong["yaseen_vs_mambaconformer"].items():
        ph = d.get("p_holm", float("nan"))
        L(f"  vs {mdl:<14} z={d['z']:+.2f}  p={d['p']:.2e}  p_Holm={ph:.2e}")
    L("")
    L("--- DeLong: CinC en iyi modeller (Holm duzeltmeli) ---")
    for key, d in delong["cinc_top"].items():
        ph = d.get("p_holm", float("nan"))
        L(f"  {key:<28} z={d['z']:+.2f}  p={d['p']:.2e}  p_Holm={ph:.2e}")
    L("")
    L("="*78)
    L("NOT (Yorum 6 '—' aciklamasi): asagidaki modellerde harici duyarlilik/ozgulluk")
    L("dejenere isletim noktasi gosteriyorsa, tabloda — yerine gercek deger + GA yazilacak.")
    (REPORT_DIR / "stats_revision.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] reports/stats_revision.json + .txt yazildi")


if __name__ == "__main__":
    main()
