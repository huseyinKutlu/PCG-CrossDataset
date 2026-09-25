#!/usr/bin/env python3
"""
04_make_circor_splits.py
PCG Q1 Manuscript — CirCor 2022 icin patient-disjoint split uretimi.

STRATEJI (itiraza kapali, literatur-karsilastirmali):
  1) Hasta duzeyinde sabit held-out TEST seti (%20) ayrilir.
     -> Resmi challenge tarzi held-out karsilastirma icin.
  2) Kalan %80 train uzerinde 5-fold patient-disjoint stratified CV.
     -> Bootstrap CI + fold ort±std icin (saglam istatistik).
  3) Held-out test HICBIR fold'a / egitime girmez.

KRITIK KURALLAR:
  - Stratifikasyon HASTA duzeyi murmur (Present/Unknown/Absent, RESMI 3-sinif).
  - Bolme HASTA duzeyinde (patient_id) -> ayni hastanin AV/MV/PV/TV kayitlari
    asla farkli bolmelere dagilmaz (sizinti yok).
  - Otomatik sizinti assert'leri: hicbir hasta iki yerde olamaz.
  - Sabit seed -> tam tekrarlanabilir.

Cikti:
  splits/circor2022_heldout_test.csv      (kayit duzeyi, test seti)
  splits/circor2022_5fold_cv.csv          (kayit duzeyi, train+fold)
  reports/circor_split_report.txt         (denetim raporu)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# -----------------------------------------------------------------------------
# Yollar — script proje kokunde calisir varsayimi (02/03 ile ayni desen)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = PROJECT_ROOT / "manifests"
SPLIT_DIR    = PROJECT_ROOT / "splits"
REPORT_DIR   = PROJECT_ROOT / "reports"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Sabitler
# -----------------------------------------------------------------------------
RANDOM_STATE  = 42
N_SPLITS      = 5
TEST_SIZE     = 0.20          # hasta duzeyinde held-out test orani
MURMUR_CLASSES = ["Present", "Unknown", "Absent"]   # RESMI 3-sinif sirasi

REPORT = []
def log(m=""):
    print(m); REPORT.append(str(m))


def load_patient_table(manifest_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kayit-duzeyi manifest -> (kayit df, hasta-duzeyi df)."""
    df = pd.read_csv(manifest_csv)
    df["patient_id"] = df["patient_id"].astype(str)

    # Hasta basina murmur tutarliligi kontrolu (CirCor'da garanti ama yine de assert)
    incon = df.groupby("patient_id")["murmur"].nunique()
    bad = incon[incon > 1]
    assert len(bad) == 0, f"Hasta icinde celisen murmur etiketi var: {list(bad.index)[:10]}"

    pat = (df.groupby("patient_id")
             .agg(murmur=("murmur", "first"),
                  outcome=("outcome", "first"),
                  n_records=("record_id", "count"),
                  n_segments=("n_segments", "sum"))
             .reset_index())
    return df, pat


def make_heldout_test(pat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hasta duzeyinde stratified held-out test ayir."""
    train_pat, test_pat = train_test_split(
        pat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=pat["murmur"],     # 3-sinif stratify
        shuffle=True,
    )
    return train_pat.reset_index(drop=True), test_pat.reset_index(drop=True)


def make_cv_folds(train_pat: pd.DataFrame) -> pd.DataFrame:
    """Train hastalari uzerinde patient-disjoint stratified 5-fold."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=RANDOM_STATE)
    train_pat = train_pat.copy()
    fold_col = np.full(len(train_pat), -1, dtype=np.int8)
    # groups = patient_id ama her hasta zaten tek satir; yine de patient-disjoint
    # garantisi icin grup olarak patient_id veriyoruz (idempotent, guvenli).
    for fold_idx, (_, val_idx) in enumerate(
            sgkf.split(train_pat,
                       train_pat["murmur"],
                       groups=train_pat["patient_id"])):
        fold_col[val_idx] = fold_idx
    train_pat["fold"] = fold_col
    assert (train_pat["fold"] >= 0).all(), "Bazi hastalara fold atanmadi!"
    return train_pat


def expand_to_records(record_df: pd.DataFrame,
                      pat_with_assignment: pd.DataFrame,
                      assignment_col: str) -> pd.DataFrame:
    """Hasta-duzeyi atamayi (fold veya test) kayit duzeyine yay."""
    amap = dict(zip(pat_with_assignment["patient_id"],
                    pat_with_assignment[assignment_col]))
    out = record_df[record_df["patient_id"].isin(amap)].copy()
    out[assignment_col] = out["patient_id"].map(amap)
    return out


def verify_no_leakage(test_pat: pd.DataFrame, train_pat: pd.DataFrame) -> None:
    """Sizinti assert'leri — Q1 hakeminin ilk bakacagi yer."""
    test_ids  = set(test_pat["patient_id"])
    train_ids = set(train_pat["patient_id"])
    overlap = test_ids & train_ids
    assert len(overlap) == 0, f"SIZINTI: test+train ortak hasta: {list(overlap)[:10]}"

    # fold'lar arasi disjoint
    for i in range(N_SPLITS):
        fi = set(train_pat[train_pat["fold"] == i]["patient_id"])
        for j in range(i + 1, N_SPLITS):
            fj = set(train_pat[train_pat["fold"] == j]["patient_id"])
            ov = fi & fj
            assert len(ov) == 0, f"SIZINTI: fold {i} & {j} ortak hasta: {list(ov)[:5]}"
    log("   [OK] Sizinti yok: test/train disjoint, fold'lar disjoint.")


def dist_table(pat: pd.DataFrame, label_col: str, group_col: str) -> str:
    """Grup x etiket capraz tablosu (hasta sayisi)."""
    return pd.crosstab(pat[group_col], pat[label_col]).to_string()


def main():
    manifest_csv = MANIFEST_DIR / "circor2022_processed.csv"
    if not manifest_csv.exists():
        # uploads ortaminda manifest yan klasorde olabilir; esnek ara
        alt = PROJECT_ROOT / "circor2022_processed.csv"
        manifest_csv = alt if alt.exists() else manifest_csv
    log("="*70)
    log("CirCor 2022 — Patient-disjoint split uretimi")
    log("="*70)
    log(f"Manifest: {manifest_csv}")
    log(f"Seed={RANDOM_STATE}, n_splits={N_SPLITS}, test_size={TEST_SIZE}")
    log(f"Murmur siniflari (resmi): {MURMUR_CLASSES}")

    record_df, pat = load_patient_table(manifest_csv)
    log(f"\nToplam: {len(record_df)} kayit | {len(pat)} hasta | "
        f"{int(record_df['n_segments'].sum())} segment")
    log("\nHasta-duzeyi murmur dagilimi:")
    log(pat["murmur"].value_counts().reindex(MURMUR_CLASSES).to_string())
    log("\nHasta-duzeyi outcome dagilimi:")
    log(pat["outcome"].value_counts().to_string())

    # 1) Held-out test
    log("\n" + "-"*70)
    log("ADIM 1: Held-out test ayrimi (hasta duzeyi, stratified)")
    log("-"*70)
    train_pat, test_pat = make_heldout_test(pat)
    log(f"   Train hasta: {len(train_pat)} | Test hasta: {len(test_pat)}")
    log("\n   Test murmur dagilimi:")
    log(test_pat["murmur"].value_counts().reindex(MURMUR_CLASSES).to_string())
    log("\n   Train murmur dagilimi:")
    log(train_pat["murmur"].value_counts().reindex(MURMUR_CLASSES).to_string())

    # 2) CV folds
    log("\n" + "-"*70)
    log("ADIM 2: 5-fold patient-disjoint CV (train uzerinde)")
    log("-"*70)
    train_pat = make_cv_folds(train_pat)
    log("\n   Fold x murmur (hasta sayisi):")
    log(dist_table(train_pat, "murmur", "fold"))
    log("\n   Fold x outcome (hasta sayisi):")
    log(dist_table(train_pat, "outcome", "fold"))

    # 3) Sizinti kontrolu
    log("\n" + "-"*70)
    log("ADIM 3: Sizinti dogrulama")
    log("-"*70)
    verify_no_leakage(test_pat, train_pat)

    # 4) Kayit duzeyine yay + yaz
    test_records  = expand_to_records(record_df, test_pat,  "murmur")  # test isaretle
    test_records  = test_records.merge(
        test_pat[["patient_id", "outcome"]].rename(columns={"outcome": "outcome_pat"}),
        on="patient_id", how="left")
    test_records["split"] = "heldout_test"

    cv_records = expand_to_records(record_df, train_pat, "fold")

    # segment-bazli sinif agirligi (record_df'den, train kismi icin)
    seg = cv_records.groupby("murmur")["n_segments"].sum()
    total_seg = seg.sum(); n_cls = len(seg)
    seg_w = {c: float(total_seg / (n_cls * seg[c])) for c in seg.index}
    log("\n   Segment-bazli sinif agirliklari (train, CV kismi):")
    for c in MURMUR_CLASSES:
        if c in seg_w:
            log(f"     {c}: segment={int(seg[c])}, weight={seg_w[c]:.4f}")

    out_test = SPLIT_DIR / "circor2022_heldout_test.csv"
    out_cv   = SPLIT_DIR / "circor2022_5fold_cv.csv"
    test_cols = ["record_id","patient_id","location","murmur","outcome",
                 "duration_s","n_segments","cache_path","split"]
    cv_cols   = ["record_id","patient_id","location","murmur","outcome",
                 "duration_s","n_segments","cache_path","fold"]
    test_records[test_cols].to_csv(out_test, index=False)
    cv_records[cv_cols].to_csv(out_cv, index=False)
    log(f"\n   Yazildi: {out_test}  ({len(test_records)} kayit)")
    log(f"   Yazildi: {out_cv}  ({len(cv_records)} kayit)")

    # Ozet sayisal saglama
    log("\n" + "-"*70)
    log("SAGLAMA")
    log("-"*70)
    log(f"   Test kayit + CV kayit = {len(test_records)} + {len(cv_records)} "
        f"= {len(test_records)+len(cv_records)} (beklenen {len(record_df)})")
    assert len(test_records) + len(cv_records) == len(record_df), \
        "Kayit sayisi tutmuyor — bir yerde kayip/cift var!"
    log("   [OK] Tum kayitlar tam olarak bir bolmede.")

    rep = REPORT_DIR / "circor_split_report.txt"
    rep.write_text("\n".join(REPORT), encoding="utf-8")
    log(f"\nDenetim raporu: {rep}")
    log("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
