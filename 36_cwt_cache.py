#!/usr/bin/env python3
"""
36_cwt_cache.py
CWT scalogram cikarir ve mevcut .npz cache'lerine 'cwt_segs' olarak ekler.

PCG-MambaConformer'in 3. dali (CWT scalogram -> ConvNeXt/Swin) icin.
ssqueezepy ile Morlet CWT, mel ile ayni boyuta (64 x 201) indirgenir.

Stratejide onemli: CWT'yi AYRI dosyaya yazmiyoruz; mevcut .npz'leri
'cwt_segs' ekleyerek yeniden yaziyoruz (raw_segs, mel_segs korunur).

Kullanim:
  python 36_cwt_cache.py --dataset circor   # CirCor cache
  python 36_cwt_cache.py --dataset cinc
  python 36_cwt_cache.py --dataset yaseen
  python 36_cwt_cache.py --dataset all
  python 36_cwt_cache.py --dataset circor --test  # ilk 3 dosyada dene
"""
from __future__ import annotations
import argparse
from pathlib import Path
import glob
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE = PROJECT_ROOT / "cache"
SR = 2000
N_FREQ = 64      # mel ile ayni frekans ekseni
N_TIME = 201     # mel ile ayni zaman ekseni
RAW_LEN = 10000  # 5 sn @ 2kHz


def compute_cwt_scalogram(raw):
    """Tek segment (10000,) -> CWT scalogram (64, 201) log-magnitude, normalize."""
    from ssqueezepy import cwt
    # Morlet CWT (ssqueezepy varsayilan 'gmw' yerine morlet kullan)
    Wx, scales = cwt(raw, wavelet="morlet", nv=8)
    # |Wx|: (n_scales, 10000) -> guc
    mag = np.abs(Wx).astype(np.float32)
    # frekans eksenini N_FREQ'e indir (ortalama havuzlama bloklari)
    n_sc = mag.shape[0]
    if n_sc >= N_FREQ:
        idx = np.linspace(0, n_sc, N_FREQ + 1).astype(int)
        mag_f = np.stack([mag[idx[i]:idx[i+1]].mean(0) for i in range(N_FREQ)])
    else:
        mag_f = np.repeat(mag, int(np.ceil(N_FREQ / n_sc)), axis=0)[:N_FREQ]
    # zaman eksenini N_TIME'e indir
    t = mag_f.shape[1]
    idx_t = np.linspace(0, t, N_TIME + 1).astype(int)
    scal = np.stack([mag_f[:, idx_t[i]:idx_t[i+1]].mean(1) for i in range(N_TIME)], axis=1)
    # log + per-sample normalize (mel ile tutarli)
    scal = np.log1p(scal)
    scal = (scal - scal.mean()) / (scal.std() + 1e-6)
    return scal.astype(np.float32)  # (64, 201)


def process_npz(path, overwrite=False):
    """Bir .npz'ye cwt_segs ekle (raw_segs'ten hesapla)."""
    d = dict(np.load(path, allow_pickle=True))
    if "cwt_segs" in d and not overwrite:
        return "var"  # zaten islenmis
    if "raw_segs" not in d:
        return "raw_yok"
    raw = d["raw_segs"]  # (n_seg, 10000)
    cwts = np.stack([compute_cwt_scalogram(raw[i]) for i in range(len(raw))])
    d["cwt_segs"] = cwts.astype(np.float32)  # (n_seg, 64, 201)
    np.savez(path, **d)
    return "ok"


def dataset_paths(name):
    if name == "circor":
        return sorted(glob.glob(str(CACHE / "circor*" / "*.npz")))
    if name == "cinc":
        return sorted(glob.glob(str(CACHE / "cinc2016" / "*.npz")))
    if name == "yaseen":
        return sorted(glob.glob(str(CACHE / "yaseen" / "*.npz")))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="circor",
                    choices=["circor", "cinc", "yaseen", "all"])
    ap.add_argument("--test", action="store_true", help="ilk 3 dosyada dene")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    datasets = ["circor", "cinc", "yaseen"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        paths = dataset_paths(ds)
        if args.test:
            paths = paths[:3]
        print(f"\n=== {ds}: {len(paths)} dosya ===")
        if not paths:
            print("  dosya bulunamadi, atla"); continue
        stats = {"ok": 0, "var": 0, "raw_yok": 0}
        for i, p in enumerate(paths):
            r = process_npz(p, overwrite=args.overwrite)
            stats[r] = stats.get(r, 0) + 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(paths)} islendi...")
        print(f"  TAMAM: yeni={stats['ok']} zaten-var={stats['var']} "
              f"raw-yok={stats['raw_yok']}")
        # bir ornek dogrula
        if args.test and paths:
            d = np.load(paths[0], allow_pickle=True)
            if "cwt_segs" in d:
                print(f"  ornek cwt_segs sekli: {d['cwt_segs'].shape} "
                      f"(beklenen: (n_seg, {N_FREQ}, {N_TIME}))")
                print(f"  deger araligi: [{d['cwt_segs'].min():.2f}, "
                      f"{d['cwt_segs'].max():.2f}]")
    print("\n[BITTI]")


if __name__ == "__main__":
    main()
