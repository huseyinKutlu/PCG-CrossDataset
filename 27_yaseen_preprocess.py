#!/usr/bin/env python3
"""
27_yaseen_preprocess.py
Yaseen 2018 dataset'ini bizim .npz formatina cevir (CinC pipeline ile AYNI).

Yaseen: 1000 wav, 8000 Hz, ~2.84s (2.39-3.55s), int16, mono.
  5 sinif x 200: AS, MR, MS, MVP, N. Klasor: {SINIF}_New_3주기/New_{SINIF}_{NNN}.wav

Pipeline (CirCor/CinC ile birebir ayni olmali — adil cross-dataset icin):
  - 8000 -> 2000 Hz resample
  - 25-400 Hz bandpass (Butterworth)
  - z-score
  - 5s pencere (kisa kayit -> padding; %50 overlap ama kisa oldugu icin 1 segment)
  - log-mel: 64 bin, fmin=25, fmax=400, n_fft/hop CirCor ile ayni

Etiketler:
  - ikili: N->normal(0), {AS,MR,MS,MVP}->abnormal(1)
  - 5-sinif: hastalik-spesifik analiz icin saklanir (AS/MR/MS/MVP/N)

Cikti:
  cache/yaseen/*.npz  (raw_segs, mel_segs, label, class5, record_id)
  manifests/yaseen_processed.csv

Kullanim:
  python 27_yaseen_preprocess.py
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample_poly

# librosa mel icin (CirCor ile ayni kullanildiysa). Yoksa pip install librosa.
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

PROJECT_ROOT = Path(__file__).resolve().parent
YASEEN_DIR = PROJECT_ROOT / "datasets" / "yaseen"
CACHE_DIR = PROJECT_ROOT / "cache" / "yaseen"; CACHE_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = PROJECT_ROOT / "manifests"; MANIFEST_DIR.mkdir(exist_ok=True)

# === parametreler (CirCor/CinC ile AYNI olmali) ===
TARGET_SR = 2000
LOWCUT, HIGHCUT = 25, 400
WIN_SEC = 5.0
WIN_LEN = int(WIN_SEC * TARGET_SR)   # 10000
HOP_SEC = 2.5                         # %50 overlap
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 50                      # CirCor mel (64,201) -> 10000/50≈200+1
FMIN, FMAX = 25, 400

CLASS5 = {"N": 0, "AS": 1, "MR": 2, "MS": 3, "MVP": 4}
CLASS_TO_BINARY = {"N": 0, "AS": 1, "MR": 1, "MS": 1, "MVP": 1}  # N=normal, digerleri=abnormal


def bandpass(sig, sr, lo=LOWCUT, hi=HIGHCUT, order=4):
    ny = 0.5 * sr
    b, a = butter(order, [lo / ny, min(hi / ny, 0.99)], btype="band")
    return filtfilt(b, a, sig)


def make_mel(sig, sr):
    """log-mel (64, ~201) — CirCor ile ayni parametreler."""
    if not HAS_LIBROSA:
        raise ImportError("librosa gerekli: pip install librosa")
    S = librosa.feature.melspectrogram(
        y=sig.astype(np.float32), sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)   # -80..0 dB
    return S_db.astype(np.float32)


def process_wav(path, sr_orig=8000):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(1)
    data = data.astype(np.float64)
    # 8000 -> 2000 (down by 4)
    if sr != TARGET_SR:
        from math import gcd
        g = gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // g, sr // g)
    # bandpass
    data = bandpass(data, TARGET_SR)
    # z-score
    data = (data - data.mean()) / (data.std() + 1e-8)
    # 5s pencere: kisa kayit -> tek pencere + padding; uzunsa overlap'li bol
    segs = []
    if len(data) <= WIN_LEN:
        seg = np.zeros(WIN_LEN, dtype=np.float32)
        seg[:len(data)] = data
        segs.append(seg)
    else:
        hop = int(HOP_SEC * TARGET_SR)
        for start in range(0, len(data) - WIN_LEN + 1, hop):
            segs.append(data[start:start + WIN_LEN].astype(np.float32))
        if not segs:
            segs.append(data[:WIN_LEN].astype(np.float32))
    raw_segs = np.stack(segs)                       # (n, 10000)
    mel_segs = np.stack([make_mel(s, TARGET_SR) for s in segs])  # (n, 64, ~201)
    return raw_segs, mel_segs


def main():
    wavs = sorted(glob.glob(str(YASEEN_DIR / "**" / "*.wav"), recursive=True))
    print("="*70)
    print(f"YASEEN PREPROCESSING | {len(wavs)} wav -> .npz (CinC pipeline)")
    print(f"librosa: {HAS_LIBROSA} | target_sr={TARGET_SR} | win={WIN_SEC}s")
    print("="*70)

    rows = []
    mel_shapes = set()
    for i, w in enumerate(wavs):
        fname = Path(w).stem                         # New_AS_125
        parts = fname.split("_")
        cls = parts[1] if len(parts) >= 2 else "?"   # AS/MR/MS/MVP/N
        if cls not in CLASS5:
            print(f"  [atla] taninmayan sinif: {fname}"); continue
        try:
            raw_segs, mel_segs = process_wav(w)
        except Exception as e:
            print(f"  [hata] {fname}: {e}"); continue
        mel_shapes.add(mel_segs.shape[1:])
        rid = fname
        out = CACHE_DIR / f"{rid}.npz"
        np.savez_compressed(
            out, raw_segs=raw_segs.astype(np.float32),
            mel_segs=mel_segs.astype(np.float32),
            label="abnormal" if CLASS_TO_BINARY[cls] else "normal",
            class5=cls, record_id=rid)
        rows.append({
            "record_id": rid, "class5": cls,
            "label": "abnormal" if CLASS_TO_BINARY[cls] else "normal",
            "n_segments": len(raw_segs),
            "cache_path": str(out.relative_to(PROJECT_ROOT))})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(wavs)} islendi...")

    import pandas as pd
    df = pd.DataFrame(rows)
    manifest = MANIFEST_DIR / "yaseen_processed.csv"
    df.to_csv(manifest, index=False)

    print(f"\n[OK] {len(rows)} kayit islendi.")
    print(f"Mel shape(ler): {mel_shapes}  (CirCor (64,201) ile uyumlu olmali!)")
    print(f"\nSinif dagilimi:")
    print(df["class5"].value_counts().to_string())
    print(f"\nIkili dagilim:")
    print(df["label"].value_counts().to_string())
    print(f"\nManifest: {manifest}")
    print(f"Cache: {CACHE_DIR}")
    print("\n=== TAMAMLANDI ===")


if __name__ == "__main__":
    main()
