#!/usr/bin/env python3
"""
03_preprocess.py
PCG Q1 Manuscript — Hafta 2: Preprocessing Pipeline (SIFIR AUGMENTATION)

Politika:
- Sentetik veri uretimi YOK
- Augmentation YOK
- Yalnizca: resample, filtre, spike removal, normalize, pencereleme, mel-spec

Cikti:
- cache/cinc2016/<record_id>.npz   -> {raw_segs, mel_segs, label, patient, subset}
- cache/circor2022/<record_id>.npz -> {raw_segs, mel_segs, murmur, outcome, ...}
- splits/cinc2016_5fold.csv        -> patient/subset-aware stratified CV
- reports/preprocess_summary.txt
- reports/preprocess_examples.png  -> ornekleme
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt, resample_poly
from scipy.stats import iqr
from sklearn.model_selection import StratifiedGroupKFold
import librosa
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Yollar
# -----------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent
MANIFEST_DIR  = PROJECT_ROOT / "manifests"
CACHE_DIR     = PROJECT_ROOT / "cache"
SPLIT_DIR     = PROJECT_ROOT / "splits"
REPORT_DIR    = PROJECT_ROOT / "reports"
for d in (CACHE_DIR / "cinc2016", CACHE_DIR / "circor2022",
          SPLIT_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Sabitler — Q1 yayin standardi
# -----------------------------------------------------------------------------
TARGET_SR     = 2000     # ortak ornekleme orani (CinC native)
LOWCUT        = 25.0     # Hz, kalp sesi alt sinir
HIGHCUT       = 400.0    # Hz, kalp sesi ust sinir (S1, S2, ust murmur)
FILTER_ORDER  = 4
WIN_SEC       = 5.0      # 5 saniyelik pencere
HOP_SEC       = 2.5      # %50 overlap
N_FFT         = 1024
HOP_LEN       = int(0.025 * TARGET_SR)  # 50 sample = 25 ms
WIN_LEN       = int(0.050 * TARGET_SR)  # 100 sample = 50 ms
N_MELS        = 64
F_MIN         = 25.0
F_MAX         = 400.0    # Nyquist 1000, ama PCG icin 400 ust limit yeter
SPIKE_K       = 3.0      # IQR esigi (Schmidt 2010 yaklasimi)
RANDOM_STATE  = 42

# -----------------------------------------------------------------------------
# Sinyal isleme yardimcilari
# -----------------------------------------------------------------------------
def resample_to_target(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    from math import gcd
    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    return resample_poly(x, up, down).astype(np.float32)

def bandpass(x: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(FILTER_ORDER, [LOWCUT/nyq, HIGHCUT/nyq], btype="band")
    return filtfilt(b, a, x).astype(np.float32)

def remove_spikes(x: np.ndarray, k: float = SPIKE_K) -> np.ndarray:
    """Schmidt-2010 tarzi: IQR tabanli klip + clamp.
    Sentetik DEGIL: sadece donanim kaynakli ekstrem ornekleri kirpar."""
    q1, q3 = np.percentile(x, [25, 75])
    iq = q3 - q1
    lo, hi = q1 - k * iq, q3 + k * iq
    return np.clip(x, lo, hi).astype(np.float32)

def zscore(x: np.ndarray) -> np.ndarray:
    m, s = float(x.mean()), float(x.std() + 1e-8)
    return ((x - m) / s).astype(np.float32)

def window_signal(x: np.ndarray, sr: int = TARGET_SR,
                  win_sec: float = WIN_SEC, hop_sec: float = HOP_SEC) -> np.ndarray:
    """Sabit 5s pencereler, %50 overlap. Son pencere yetersizse zero-pad."""
    n_win = int(win_sec * sr)
    n_hop = int(hop_sec * sr)
    if len(x) < n_win:
        pad = np.zeros(n_win - len(x), dtype=np.float32)
        return np.stack([np.concatenate([x, pad])])[None, 0:1].squeeze(0).reshape(1, -1)
    starts = list(range(0, len(x) - n_win + 1, n_hop))
    if not starts:
        starts = [0]
    return np.stack([x[s:s+n_win] for s in starts])

def mel_spec(seg: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    return librosa.power_to_db(S + 1e-10, ref=np.max).astype(np.float32)

# -----------------------------------------------------------------------------
# Tek dosya isleme
# -----------------------------------------------------------------------------
def process_one(wav_path: Path) -> Optional[dict]:
    try:
        x, sr = sf.read(str(wav_path), always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = resample_to_target(x, sr_in=sr)
        x = bandpass(x)
        x = remove_spikes(x)
        x = zscore(x)
        raw_segs = window_signal(x)             # (n_seg, 10000)
        mel_segs = np.stack([mel_spec(s) for s in raw_segs])  # (n_seg, 64, T)
        return {"raw_segs": raw_segs, "mel_segs": mel_segs}
    except Exception as e:
        print(f"   HATA: {wav_path.name}: {e}")
        return None

# -----------------------------------------------------------------------------
# CinC 2016 isleme
# -----------------------------------------------------------------------------
def process_cinc(manifest: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== CinC 2016 preprocessing ({len(manifest)} kayit) ===")
    manifest = manifest[manifest["label"].isin(["normal","abnormal"])].copy()
    print(f"   Etiketli kayit: {len(manifest)}")

    cache_root = CACHE_DIR / "cinc2016"
    out_rows = []
    n_seg_total = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="   CinC"):
        wav = PROJECT_ROOT / row["path"]
        out_npz = cache_root / f"{row['record_id']}.npz"
        if out_npz.exists():
            d = np.load(out_npz, allow_pickle=True)
            n_seg = int(d["raw_segs"].shape[0])
        else:
            res = process_one(wav)
            if res is None:
                continue
            np.savez_compressed(
                out_npz,
                raw_segs=res["raw_segs"],
                mel_segs=res["mel_segs"],
                label=row["label"],
                subset=row["subset"],
                record_id=row["record_id"],
            )
            n_seg = int(res["raw_segs"].shape[0])
        n_seg_total += n_seg
        out_rows.append({
            "record_id": row["record_id"],
            "subset":    row["subset"],
            "label":     row["label"],
            "duration_s": row["duration_s"],
            "n_segments": n_seg,
            "cache_path": str(out_npz.relative_to(PROJECT_ROOT)),
        })

    df = pd.DataFrame(out_rows)
    out_csv = MANIFEST_DIR / "cinc2016_processed.csv"
    df.to_csv(out_csv, index=False)
    print(f"   Toplam segment: {n_seg_total}")
    print(f"   Cache yazildi: {cache_root}")
    print(f"   Manifest yazildi: {out_csv}")
    return df

# -----------------------------------------------------------------------------
# CirCor 2022 isleme
# -----------------------------------------------------------------------------
def process_circor(manifest: pd.DataFrame) -> pd.DataFrame:
    print(f"\n=== CirCor 2022 preprocessing ({len(manifest)} kayit) ===")
    cache_root = CACHE_DIR / "circor2022"
    out_rows = []
    n_seg_total = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="   CirCor"):
        wav = PROJECT_ROOT / row["path"]
        out_npz = cache_root / f"{row['record_id']}.npz"
        if out_npz.exists():
            d = np.load(out_npz, allow_pickle=True)
            n_seg = int(d["raw_segs"].shape[0])
        else:
            res = process_one(wav)
            if res is None:
                continue
            np.savez_compressed(
                out_npz,
                raw_segs=res["raw_segs"],
                mel_segs=res["mel_segs"],
                murmur=row["murmur"],
                outcome=row["outcome"],
                patient_id=row["patient_id"],
                location=row["location"],
                record_id=row["record_id"],
            )
            n_seg = int(res["raw_segs"].shape[0])
        n_seg_total += n_seg
        out_rows.append({
            "record_id":  row["record_id"],
            "patient_id": row["patient_id"],
            "location":   row["location"],
            "murmur":     row["murmur"],
            "outcome":    row["outcome"],
            "duration_s": row["duration_s"],
            "n_segments": n_seg,
            "cache_path": str(out_npz.relative_to(PROJECT_ROOT)),
        })

    df = pd.DataFrame(out_rows)
    out_csv = MANIFEST_DIR / "circor2022_processed.csv"
    df.to_csv(out_csv, index=False)
    print(f"   Toplam segment: {n_seg_total}")
    print(f"   Cache yazildi: {cache_root}")
    print(f"   Manifest yazildi: {out_csv}")
    return df

# -----------------------------------------------------------------------------
# CV split — Patient-disjoint, subset-aware, stratified
# -----------------------------------------------------------------------------
def make_cv_splits(cinc_df: pd.DataFrame, n_splits: int = 5) -> None:
    """
    CinC 2016 icin patient/subset-aware stratified CV.
    CinC'te 'patient_id' yok ama record_id zaten benzersiz hasta varsayimi
    yeterince ortak yapildi (literatur). Stratification: subset+label birlestirilmis.
    """
    print(f"\n=== {n_splits}-fold CV split ===")
    df = cinc_df.copy()
    df["strat_key"] = df["subset"].astype(str) + "_" + df["label"].astype(str)
    # CinC'te asagidaki onemli: training-e bias'i kirilmali
    # StratifiedGroupKFold: groups=record_id (her kayit ayri 'hasta' kabul)
    sgkf = StratifiedGroupKFold(n_splits=n_splits,
                                shuffle=True, random_state=RANDOM_STATE)
    fold_col = np.full(len(df), -1, dtype=np.int8)
    for fold_idx, (_, val_idx) in enumerate(
            sgkf.split(df, df["strat_key"], groups=df["record_id"])):
        fold_col[val_idx] = fold_idx
    df["fold"] = fold_col

    # Saglama
    print("   Fold dagilimi:")
    print(pd.crosstab(df["fold"], df["label"]).to_string())
    print("   Fold x subset:")
    print(pd.crosstab(df["fold"], df["subset"]).to_string())

    out = SPLIT_DIR / "cinc2016_5fold.csv"
    df[["record_id","subset","label","fold","n_segments"]].to_csv(out, index=False)
    print(f"   Yazildi: {out}")

# -----------------------------------------------------------------------------
# Sinif agirliklari
# -----------------------------------------------------------------------------
def compute_class_weights(cinc_df: pd.DataFrame) -> dict:
    counts = cinc_df["label"].value_counts().to_dict()
    total = sum(counts.values())
    n_cls = len(counts)
    weights = {lab: total / (n_cls * c) for lab, c in counts.items()}
    print(f"\n=== Sinif agirliklari (record-bazli) ===")
    for lab, w in weights.items():
        print(f"   {lab}: count={counts[lab]}, weight={w:.4f}")
    out = REPORT_DIR / "class_weights.txt"
    out.write_text("\n".join(f"{k}\t{v:.6f}" for k, v in weights.items()))
    print(f"   Yazildi: {out}")
    return weights

# -----------------------------------------------------------------------------
# Goruntu ornekleri (1 normal, 1 abnormal mel-spektrogram)
# -----------------------------------------------------------------------------
def plot_examples(cinc_df: pd.DataFrame) -> None:
    cache_root = CACHE_DIR / "cinc2016"
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for i, lab in enumerate(["normal", "abnormal"]):
        sub = cinc_df[cinc_df["label"] == lab]
        if sub.empty: continue
        rec = sub.iloc[0]
        d = np.load(cache_root / f"{rec['record_id']}.npz", allow_pickle=True)
        raw = d["raw_segs"][0]
        mel = d["mel_segs"][0]
        axes[i, 0].plot(np.arange(len(raw)) / TARGET_SR, raw, lw=0.6)
        axes[i, 0].set_title(f"{lab} - {rec['record_id']} - 1. pencere (5s)")
        axes[i, 0].set_xlabel("Zaman (s)")
        axes[i, 0].set_ylabel("Genlik (z-score)")
        im = axes[i, 1].imshow(mel, aspect="auto", origin="lower",
                               extent=[0, WIN_SEC, F_MIN, F_MAX])
        axes[i, 1].set_title(f"{lab} - log-mel (64 bin)")
        axes[i, 1].set_xlabel("Zaman (s)")
        axes[i, 1].set_ylabel("Frekans (Hz)")
        plt.colorbar(im, ax=axes[i, 1], format="%+2.0f dB")
    plt.tight_layout()
    out = REPORT_DIR / "preprocess_examples.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   Ornek figur: {out}")

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"TARGET_SR={TARGET_SR}, WIN={WIN_SEC}s, HOP={HOP_SEC}s")
    print(f"Mel: n_mels={N_MELS}, hop={HOP_LEN}, win={WIN_LEN}, fmax={F_MAX}")

    cinc_man   = pd.read_csv(MANIFEST_DIR / "cinc2016_manifest.csv")
    circor_man = pd.read_csv(MANIFEST_DIR / "circor2022_manifest.csv")

    cinc_df   = process_cinc(cinc_man)
    circor_df = process_circor(circor_man)

    make_cv_splits(cinc_df, n_splits=5)
    compute_class_weights(cinc_df)
    plot_examples(cinc_df)

    # Ozet
    summary = REPORT_DIR / "preprocess_summary.txt"
    with summary.open("w") as f:
        f.write("PCG Preprocessing Ozeti\n")
        f.write("="*60 + "\n")
        f.write(f"CinC kayit: {len(cinc_df)}\n")
        f.write(f"CinC segment: {int(cinc_df['n_segments'].sum())}\n")
        f.write(f"CirCor kayit: {len(circor_df)}\n")
        f.write(f"CirCor segment: {int(circor_df['n_segments'].sum())}\n")
        f.write(f"\nParams: SR={TARGET_SR}, BP=[{LOWCUT}-{HIGHCUT}]Hz, "
                f"win={WIN_SEC}s, hop={HOP_SEC}s\n")
        f.write(f"Mel: n_mels={N_MELS}, fmin={F_MIN}, fmax={F_MAX}\n")
        f.write("\nAUGMENTATION POLICY: NONE (zero synthetic, raw real data)\n")
    print(f"\nOzet: {summary}")
    print("\n=== TAMAMLANDI ===")

if __name__ == "__main__":
    main()
