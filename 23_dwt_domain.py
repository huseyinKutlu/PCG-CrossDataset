#!/usr/bin/env python3
"""
23_dwt_domain.py
DWT (wavelet) tabanli domain-farki TESHISI.

FIKIR: Pediatrik (CirCor) ve yetiskin (CinC) PCG arasindaki domain farki
belki belirli FREKANS BANTLARINDA yogunlasiyor. DWT ile sinyali alt-bantlara
ayirip her bandin enerji dagilimini kiyaslarsak, domain-farkini tasiyan
bantlari gorebiliriz. Eger net bir bant-farki varsa, DWT-tabanli domain-robust
temsil (farki tasiyan bandi zayiflat / domain-ortak bandi one cikar) mantikli.

Bu bir TESHIS — temsil degisikliginden once "fark nerede?" sorusunu cevaplar.

Kullanim:
  python 23_dwt_domain.py
"""
from __future__ import annotations
from pathlib import Path
from importlib import import_module
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def load_some_raw(cache_glob, n_files=40):
    import glob, os
    files = glob.glob(os.path.expanduser(cache_glob))[:n_files]
    segs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        rs = d["raw_segs"]
        for i in range(min(3, len(rs))):   # her dosyadan birkac segment
            segs.append(rs[i].astype(np.float32))
    return np.array(segs)   # (N, 10000)


def dwt_band_energy(sig, wavelet="db4", level=5):
    """Bir sinyalin DWT alt-bant enerji dagilimi (normalize)."""
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    # coeffs[0]=approx (en dusuk frek), coeffs[1:]=detay (artan frek)
    energies = np.array([float((c ** 2).sum()) for c in coeffs])
    return energies / (energies.sum() + 1e-12)


def fft_band_energy(sig, fs=2000,
                    bands=((0, 25), (25, 50), (50, 100), (100, 200), (200, 400))):
    f = np.fft.rfftfreq(len(sig), 1 / fs)
    P = np.abs(np.fft.rfft(sig)) ** 2
    e = np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in bands])
    return e / (e.sum() + 1e-12)


def main():
    print("="*70)
    print("DWT DOMAIN-FARKI TESHISI (CirCor vs CinC)")
    print("="*70)

    cir = load_some_raw("~/Desktop/pcg_project/cache/circor*/*.npz")
    cinc = load_some_raw("~/Desktop/pcg_project/cache/cinc2016/*.npz")
    print(f"CirCor segment: {len(cir)} | CinC segment: {len(cinc)}")

    if HAS_PYWT:
        print(f"\n--- DWT alt-bant enerji (db4, 5 seviye) ---")
        cir_e = np.mean([dwt_band_energy(s) for s in cir], axis=0)
        cinc_e = np.mean([dwt_band_energy(s) for s in cinc], axis=0)
        labels = ["A5(en dusuk)", "D5", "D4", "D3", "D2", "D1(en yuksek)"]
        print(f'{"bant":>14} {"CirCor":>9} {"CinC":>9} {"|fark|":>8}')
        for lab, c, n in zip(labels, cir_e, cinc_e):
            print(f'{lab:>14} {c:>9.3f} {n:>9.3f} {abs(c-n):>8.3f}')
        diff = np.abs(cir_e - cinc_e)
        print(f"\nEn buyuk domain-farki bandi: {labels[diff.argmax()]} (fark={diff.max():.3f})")
    else:
        print("\n[!] PyWavelets yok (pip install pywavelets). FFT-bant ile devam:")

    print(f"\n--- FFT frekans-bant enerji (yedek/dogrulama) ---")
    bands = ["0-25Hz", "25-50Hz", "50-100Hz", "100-200Hz", "200-400Hz"]
    cir_f = np.mean([fft_band_energy(s) for s in cir], axis=0)
    cinc_f = np.mean([fft_band_energy(s) for s in cinc], axis=0)
    print(f'{"bant":>10} {"CirCor":>9} {"CinC":>9} {"|fark|":>8}')
    for lab, c, n in zip(bands, cir_f, cinc_f):
        print(f'{lab:>10} {c:>9.3f} {n:>9.3f} {abs(c-n):>8.3f}')
    fdiff = np.abs(cir_f - cinc_f)
    print(f"\nEn buyuk domain-farki bandi: {bands[fdiff.argmax()]} (fark={fdiff.max():.3f})")

    print("\n" + "="*70)
    if fdiff.max() > 0.15:
        print("[SONUC] Belirgin bant-farki var -> DWT-tabanli domain-robust temsil")
        print("        mantikli. Domain-farkini tasiyan bandi zayiflat/normalize et.")
    else:
        print("[SONUC] Bant-farki dagilmis/kucuk -> tek bir bandi hedeflemek zor.")
        print("        Domain farki spektral degil, muhtemelen morfolojik/yapisal.")
        print("        DWT-bant yaklasimi sinirli fayda saglar.")
    print("="*70)


if __name__ == "__main__":
    main()
