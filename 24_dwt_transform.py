#!/usr/bin/env python3
"""
24_dwt_transform.py
DWT-tabanli domain-robust sinyal donusumu.

GEREKCE (23_dwt_domain.py teshisi): CirCor↔CinC domain farki FREKANS
BANTLARINDA lokalize:
  - 25-50 Hz: CinC'te baskin (yetiskin, dusuk frek)  [fark 0.262]
  - 200-400 Hz: CirCor'da baskin (cocuk, yuksek frek) [fark 0.180]
  - DWT D5 bandi: en buyuk fark (0.181)

FIKIR: Sinyali DWT alt-bantlarina ayir, her bandi normalize et (domain'e
ozgu bant-enerji profilini sil), yeniden insa et. Boylece model domain-ortak
YAPIYI gorur, domain-spesifik bant-dengesini degil. -> domain-robust temsil.

Bu sinyal-isleme tarafindan bir cozum (DANN gibi ogrenme-tarafi degil).
Ozgun katki potansiyeli: "DWT alt-bant normalizasyonu ile domain-robust PCG".

Kullanim (modul olarak 16'ya baglanir veya standalone test):
  python 24_dwt_transform.py   # birim test
"""
from __future__ import annotations
import numpy as np

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def dwt_band_normalize(sig, wavelet="db4", level=5, mode="energy"):
    """Sinyali DWT alt-bantlarina ayir, her bandi normalize et, yeniden insa.

    mode='energy': her bandi birim enerjiye normalize (bant-profilini esitler)
    mode='zscore': her bandi z-score (mean0/std1)

    Sekil korunur (giris uzunlugu = cikis uzunlugu).
    """
    if not HAS_PYWT:
        raise ImportError("pywt gerekli: pip install pywavelets")
    n = len(sig)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    out = []
    for c in coeffs:
        if mode == "energy":
            e = np.sqrt(float((c ** 2).sum()) + 1e-8)
            out.append(c / e)
        elif mode == "zscore":
            out.append((c - c.mean()) / (c.std() + 1e-8))
        else:
            out.append(c)
    rec = pywt.waverec(out, wavelet)
    rec = rec[:n] if len(rec) >= n else np.pad(rec, (0, n - len(rec)))
    # son olarak global z-score (model girdi olcegi tutarli kalsin)
    rec = (rec - rec.mean()) / (rec.std() + 1e-8)
    return rec.astype(np.float32)


def dwt_band_reweight(sig, target_profile, wavelet="db4", level=5):
    """Alternatif: her bandin enerjisini SABIT hedef profile esle.
    target_profile: (level+1,) normalize enerji oranlari (domain-ortak referans).
    """
    if not HAS_PYWT:
        raise ImportError("pywt gerekli")
    n = len(sig)
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    energies = np.array([float((c ** 2).sum()) + 1e-8 for c in coeffs])
    total = energies.sum()
    out = []
    for c, e, tgt in zip(coeffs, energies, target_profile):
        # bandi once birim enerjiye, sonra hedef orana olcekle
        scale = np.sqrt(tgt * total / e)
        out.append(c * scale)
    rec = pywt.waverec(out, wavelet)
    rec = rec[:n] if len(rec) >= n else np.pad(rec, (0, n - len(rec)))
    rec = (rec - rec.mean()) / (rec.std() + 1e-8)
    return rec.astype(np.float32)


if __name__ == "__main__":
    print("="*60)
    print("DWT domain-robust donusum testi")
    print("="*60)
    if not HAS_PYWT:
        print("[!] pywt yok — pip install pywavelets")
        raise SystemExit

    fs = 2000
    t = np.arange(5 * fs) / fs
    rng = np.random.default_rng(0)
    # pediatrik vs yetiskin benzeri
    ped = (np.sin(2*np.pi*2*t) + 0.6*np.sin(2*np.pi*250*t)
           + 0.3*np.sin(2*np.pi*40*t) + 0.05*rng.standard_normal(len(t)))
    adt = (np.sin(2*np.pi*1.2*t) + 0.1*np.sin(2*np.pi*250*t)
           + 0.9*np.sin(2*np.pi*40*t) + 0.05*rng.standard_normal(len(t)))

    def band_e(x, w="db4", lv=5):
        c = pywt.wavedec(x, w, level=lv)
        e = np.array([float((ci**2).sum()) for ci in c]); return e/e.sum()

    pe, ae = band_e(ped), band_e(adt)
    print(f"\nDonusum ONCESI bant-farki: {np.abs(pe-ae).sum():.3f}")

    pn = dwt_band_normalize(ped); an = dwt_band_normalize(adt)
    assert pn.shape == ped.shape, "sekil bozuldu!"
    pen, aen = band_e(pn), band_e(an)
    print(f"Donusum SONRASI bant-farki: {np.abs(pen-aen).sum():.3f}")
    print(f"\nSekil korundu: {pn.shape} | nan yok: {np.isfinite(pn).all()}")
    if np.abs(pen-aen).sum() < np.abs(pe-ae).sum():
        print("[OK] DWT band-normalize domain bant-farkini azaltti")
    print("\n=== test gecti ===")
