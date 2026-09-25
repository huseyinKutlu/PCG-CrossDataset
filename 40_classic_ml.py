#!/usr/bin/env python3
"""
40_classic_ml.py
Klasik ML baseline: SVM + Random Forest.
Oznitelikler: MFCC + DWT bant enerjileri + spektral oznitelikler.

Derin ogrenme modelleriyle BIREBIR karsilastirilabilir:
  - Ayni CirCor 5-fold split
  - Ayni record-seviyesi degerlendirme
  - Ayni CinC + Yaseen cross-dataset zero-shot test

Amac: "derin ogrenme klasik ML'den gercekten iyi mi?" + "klasik ML
cross-dataset'te daha stabil mi?" sorularini yanitlamak (Q1 hakem bekler).

Kullanim:
  python 40_classic_ml.py            # SVM + RF, tam degerlendirme
  python 40_classic_ml.py --test     # ilk birkac record (hizli test)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE = PROJECT_ROOT / "cache"
SR = 2000
N_FOLDS = 5


# ---------- OZNITELIK CIKARIMI ----------
def extract_features(raw):
    """Tek segment (10000,) -> oznitelik vektoru (MFCC+DWT+spektral)."""
    import librosa
    feats = []
    # 1) MFCC (13 katsayi, mean+std)
    mfcc = librosa.feature.mfcc(y=raw, sr=SR, n_mfcc=13, n_fft=512, hop_length=256)
    feats.extend(mfcc.mean(1)); feats.extend(mfcc.std(1))
    # 2) delta MFCC mean
    dmfcc = librosa.feature.delta(mfcc)
    feats.extend(dmfcc.mean(1))
    # 3) spektral oznitelikler
    sc = librosa.feature.spectral_centroid(y=raw, sr=SR, n_fft=512, hop_length=256)
    sb = librosa.feature.spectral_bandwidth(y=raw, sr=SR, n_fft=512, hop_length=256)
    sr_ = librosa.feature.spectral_rolloff(y=raw, sr=SR, n_fft=512, hop_length=256)
    zcr = librosa.feature.zero_crossing_rate(raw, frame_length=512, hop_length=256)
    for f in [sc, sb, sr_, zcr]:
        feats.extend([f.mean(), f.std()])
    # 4) DWT bant enerjileri (5 seviye)
    import pywt
    coeffs = pywt.wavedec(raw, "db4", level=5)
    for c in coeffs:
        feats.append(float(np.sum(c ** 2)))   # bant enerjisi
        feats.append(float(np.std(c)))
    # 5) zaman-alani istatistikleri
    feats.extend([raw.mean(), raw.std(), np.percentile(np.abs(raw), 90),
                  float(np.sqrt(np.mean(raw ** 2)))])  # RMS
    return np.array(feats, dtype=np.float32)


def record_features(npz_path, label_key):
    """Bir record'un tum segmentlerinden ozellik cikar, ortala (record-seviyesi)."""
    d = np.load(npz_path, allow_pickle=True)
    raw = d["raw_segs"]
    seg_feats = np.stack([extract_features(raw[i]) for i in range(len(raw))])
    rec_feat = seg_feats.mean(0)  # segmentler uzeri ortalama
    # etiket
    if label_key in d:
        lab_raw = str(d[label_key])
    else:
        lab_raw = None
    return rec_feat, lab_raw


# ---------- DATASET YUKLEME ----------
def load_circor(test=False):
    """CirCor: record-seviyesi ozellik + ikili etiket + fold."""
    cv = pd.read_csv(PROJECT_ROOT / "splits" / "circor2022_5fold_cv.csv")
    if test: cv = cv.head(20)
    X, y, folds = [], [], []
    for _, row in cv.iterrows():
        p = PROJECT_ROOT / row["cache_path"]
        if not p.exists(): continue
        feat, _ = record_features(p, "murmur")
        # ikili: Present/Unknown -> abnormal(1), Absent -> normal(0)
        mur = str(row["murmur"])
        lab = 0 if mur == "Absent" else 1
        X.append(feat); y.append(lab); folds.append(int(row["fold"]))
    return np.array(X), np.array(y), np.array(folds)


def load_external(manifest_name, label_col, test=False):
    """CinC/Yaseen: record-seviyesi ozellik + ikili etiket."""
    man = pd.read_csv(PROJECT_ROOT / "manifests" / manifest_name)
    if test: man = man.head(20)
    X, y = [], []
    for _, row in man.iterrows():
        p = PROJECT_ROOT / row["cache_path"]
        if not p.exists(): continue
        feat, lab_raw = record_features(p, "label")
        lab = lab_raw if lab_raw else str(row[label_col])
        ybin = 0 if lab.lower() in ("normal", "absent", "-1", "0") else 1
        X.append(feat); y.append(ybin)
    return np.array(X), np.array(y)


# ---------- DEGERLENDIRME ----------
def evaluate():
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, confusion_matrix

    print("CirCor ozellikleri cikariliyor (record-seviyesi)...")
    Xc, yc, folds = load_circor()
    print(f"  CirCor: {Xc.shape[0]} record, {Xc.shape[1]} oznitelik")
    print("CinC ozellikleri...")
    Xcinc, ycinc = load_external("cinc2016_processed.csv", "label")
    print(f"  CinC: {Xcinc.shape[0]} record")
    print("Yaseen ozellikleri...")
    Xyas, yyas = load_external("yaseen_processed.csv", "label")
    print(f"  Yaseen: {Xyas.shape[0]} record")

    models = {
        "SVM-RBF": lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                               class_weight="balanced"),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=300, max_depth=None,
                                                       class_weight="balanced", n_jobs=-1,
                                                       random_state=42),
    }

    for mname, mfn in models.items():
        print("\n" + "="*60)
        print(f"MODEL: {mname}")
        print("="*60)
        ic_aucs, cinc_preds, yas_preds = [], [], []
        for fold in range(N_FOLDS):
            tr = folds != fold
            va = folds == fold
            scaler = StandardScaler().fit(Xc[tr])
            clf = mfn()
            clf.fit(scaler.transform(Xc[tr]), yc[tr])
            # ic val
            p_va = clf.predict_proba(scaler.transform(Xc[va]))[:, 1]
            ic_aucs.append(roc_auc_score(yc[va], p_va))
            # dis (her fold modeli, sonra ensemble)
            cinc_preds.append(clf.predict_proba(scaler.transform(Xcinc))[:, 1])
            yas_preds.append(clf.predict_proba(scaler.transform(Xyas))[:, 1])
        ic_auc = np.mean(ic_aucs)
        cinc_ens = np.mean(cinc_preds, 0)
        yas_ens = np.mean(yas_preds, 0)
        cinc_auc = roc_auc_score(ycinc, cinc_ens)
        yas_auc = roc_auc_score(yyas, yas_ens)
        # sensitivity/specificity (esik 0.5)
        def sens_spec(y, p):
            cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
            return cm[1,1]/max(cm[1].sum(),1), cm[0,0]/max(cm[0].sum(),1)
        cinc_se, cinc_sp = sens_spec(ycinc, cinc_ens)
        yas_se, yas_sp = sens_spec(yyas, yas_ens)
        print(f"  Ic (CirCor 5-fold): AUROC={ic_auc:.4f} (±{np.std(ic_aucs):.4f})")
        print(f"  Dis CinC (ens):     AUROC={cinc_auc:.4f} | Sens={cinc_se:.3f} Spec={cinc_sp:.3f}")
        print(f"  Dis Yaseen (ens):   AUROC={yas_auc:.4f} | Sens={yas_se:.3f} Spec={yas_sp:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    print("="*60)
    print("KLASIK ML BASELINE (SVM + Random Forest)")
    print("Oznitelikler: MFCC + DWT + spektral")
    print("="*60)
    if args.test:
        print("[TEST modu: ilk 20 record]")
        Xc, yc, folds = load_circor(test=True)
        print(f"  CirCor test: {Xc.shape}, etiketler: {np.bincount(yc)}")
        print("  [OK] oznitelik cikarimi calisiyor")
        return
    evaluate()
    print("\n[BITTI]")


if __name__ == "__main__":
    main()
