#!/usr/bin/env python3
"""
15_stats_test.py
PCG Q1 Manuscript — istatistiksel anlamlilik analizi.

14_cv5_single.py'nin kaydettigi hasta-duzeyi tahminleri (preds_*.npz) uzerinden:
  1) Her model icin hasta-seviyesi bootstrap %95 CI (W.Acc, macro-F1, UAR)
  2) Model ciftleri arasi PAIRED bootstrap testi (ana model digerlerini
     anlamli geciyor mu — n=hasta sayisi gucuyle, n=5 fold degil)

Neden paired bootstrap: 5 fold ile Wilcoxon gucu dusuk (n=5).
Ayni hastalar uzerinde iki modelin farkini bootstrap'lamak cok daha guclu.

Kullanim (once 14_cv5_single --model all kosmali):
  python 15_stats_test.py --ref hybrid_kan
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module

import numpy as np

me_mod = import_module("06_metrics")

PROJECT_ROOT = Path(__file__).resolve().parent
PRED_DIR = PROJECT_ROOT / "checkpoints" / "cv5"
REPORT_DIR = PROJECT_ROOT / "reports"; REPORT_DIR.mkdir(exist_ok=True)


def load_preds(model):
    p = PRED_DIR / f"preds_{model}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["ids"], d["y_true"], d["probs"]


def metric_from_probs(y_true, probs, which="weighted_accuracy"):
    pred = probs.argmax(1)
    return me_mod.all_metrics(y_true, pred)[which]


def bootstrap_ci(y_true, probs, which="weighted_accuracy", n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)   # hasta-seviyesi resample
        vals.append(metric_from_probs(y_true[idx], probs[idx], which))
    point = metric_from_probs(y_true, probs, which)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def paired_bootstrap(y_true, probs_a, probs_b, which="weighted_accuracy",
                     n_boot=10000, seed=42):
    """A - B farki icin paired bootstrap. Ayni hasta indexleri her iki modele.
    Doner: fark (A-B), %95 CI, tek-yonlu p (A<=B olasiligi)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ma = metric_from_probs(y_true[idx], probs_a[idx], which)
        mb = metric_from_probs(y_true[idx], probs_b[idx], which)
        diffs.append(ma - mb)
    diffs = np.array(diffs)
    point = (metric_from_probs(y_true, probs_a, which)
             - metric_from_probs(y_true, probs_b, which))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_one_sided = float((diffs <= 0).mean())   # A, B'den iyi DEGIL olasiligi
    return point, lo, hi, p_one_sided


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="hybrid_kan", help="referans (ana) model")
    ap.add_argument("--models", nargs="*",
                    default=["cnn1d", "cnn2d", "cnn_bilstm", "hybrid_mlp", "hybrid_kan"])
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    logs = []
    def log(m=""): print(m); logs.append(str(m))

    log("="*70)
    log("ISTATISTIKSEL ANLAMLILIK (hasta-seviyesi bootstrap)")
    log("="*70)

    # yukle
    data = {}
    for mdl in args.models:
        r = load_preds(mdl)
        if r is None:
            log(f"  [atla] {mdl}: preds bulunamadi (once 14_cv5_single kos)")
            continue
        data[mdl] = r
    if args.ref not in data:
        log(f"  !! referans model '{args.ref}' tahminleri yok. Cikiliyor."); return

    # hizalama kontrolu: ayni hasta kumesi mi (paired test icin sart)
    ref_ids = data[args.ref][0]
    log(f"\nReferans: {args.ref} | {len(ref_ids)} hasta tahmini")

    # 1) her model icin bootstrap CI
    log("\n--- Model bazinda hasta-seviyesi bootstrap %95 CI ---")
    log(f"  {'model':14s} | {'W.Acc (95% CI)':24s} | {'macro-F1 (95% CI)':24s}")
    log("  " + "-"*64)
    for mdl, (ids, yt, pp) in data.items():
        wp, wlo, whi = bootstrap_ci(yt, pp, "weighted_accuracy", args.n_boot)
        fp, flo, fhi = bootstrap_ci(yt, pp, "macro_f1", args.n_boot)
        log(f"  {mdl:14s} | {wp:.3f} [{wlo:.3f}, {whi:.3f}]   | "
            f"{fp:.3f} [{flo:.3f}, {fhi:.3f}]")

    # 2) referans vs digerleri paired bootstrap
    log(f"\n--- Paired bootstrap: {args.ref} vs digerleri (W.Acc farki) ---")
    log("  pozitif fark = referans daha iyi | p<0.05 = anlamli ustunluk")
    log(f"  {'kiyas':24s} | {'Δ W.Acc (95% CI)':28s} | {'p':8s} | anlamli?")
    log("  " + "-"*72)
    ref_ids, ref_yt, ref_pp = data[args.ref]
    for mdl, (ids, yt, pp) in data.items():
        if mdl == args.ref:
            continue
        # paired test icin ayni hasta sirasi gerekir
        if not np.array_equal(ids, ref_ids):
            # ortak id'lere hizala
            common = np.intersect1d(ids, ref_ids)
            iref = np.array([np.where(ref_ids == c)[0][0] for c in common])
            iother = np.array([np.where(ids == c)[0][0] for c in common])
            yt_use = ref_yt[iref]; a = ref_pp[iref]; b = pp[iother]
        else:
            yt_use = ref_yt; a = ref_pp; b = pp
        d, lo, hi, p = paired_bootstrap(yt_use, a, b, "weighted_accuracy", args.n_boot)
        sig = "EVET" if p < 0.05 else "hayir"
        log(f"  {args.ref} vs {mdl:11s} | {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  | "
            f"{p:.4f} | {sig}")

    log("\n  NOT: p, referansin digerinden 'iyi olmama' olasiligi (tek yonlu).")
    log("       95% CI 0'i icermiyorsa fark anlamlidir.")

    rep = REPORT_DIR / "stats_test.txt"
    rep.write_text("\n".join(logs), encoding="utf-8")
    log(f"\nRapor: {rep}")
    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
