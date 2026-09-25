#!/usr/bin/env python3
"""
17_augment.py
PCG Q1 Manuscript — on-the-fly augmentation (domain-shift dayanikliligi).

Hipotez: cross-dataset cokusun kaynagi dar egitim dagilimi. Augmentation
modeli farkli kayit kosullarina (genlik, gurultu, tempo, spektral profil)
karsi degismez yapar -> pediatrik->yetiskin transferini iyilestirebilir.

Augmentation'lar PCG-uygun ve domain-shift'i taklit eder:
  - amp_scale   : farkli kayit seviyeleri (genlik)
  - add_noise   : farkli cihaz/ortam gurultusu
  - time_stretch: farkli kalp hizlari (cocuk hizli, yetiskin yavas)
  - spec_augment: spektral maskeleme (SpecAugment)

YALNIZCA egitimde uygulanir. Test/dis-validasyonda KAPALI.
Sekil her zaman korunur (raw 10000, mel 64x201).
"""
from __future__ import annotations
import numpy as np


class PCGAugment:
    """Egitim-anı PCG augmentation. raw ve mel uzerinde calisir.
    Her augmentation bagimsiz olasilikla (p) uygulanir."""
    def __init__(self, p_amp=0.5, p_noise=0.5, p_stretch=0.3, p_spec=0.5,
                 amp_range=(0.7, 1.3), snr_db_range=(15, 30),
                 stretch_range=(0.9, 1.1), spec_freq=8, spec_time=20,
                 seed=None):
        self.p_amp, self.p_noise = p_amp, p_noise
        self.p_stretch, self.p_spec = p_stretch, p_spec
        self.amp_range = amp_range
        self.snr_db_range = snr_db_range
        self.stretch_range = stretch_range
        self.spec_freq, self.spec_time = spec_freq, spec_time
        self.rng = np.random.default_rng(seed)

    def _amp_scale(self, raw):
        return raw * self.rng.uniform(*self.amp_range)

    def _add_noise(self, raw):
        snr = self.rng.uniform(*self.snr_db_range)
        sig_p = float((raw ** 2).mean()) + 1e-12
        noise_p = sig_p / (10 ** (snr / 10))
        noise = self.rng.standard_normal(raw.shape).astype(np.float32) * np.sqrt(noise_p)
        return raw + noise

    def _time_stretch(self, raw):
        rate = self.rng.uniform(*self.stretch_range)
        n = len(raw); new_n = max(1, int(n / rate))
        idx = np.linspace(0, n - 1, new_n).astype(int)
        y = raw[idx]
        if len(y) >= n:
            return y[:n].astype(np.float32)
        return np.pad(y, (0, n - len(y))).astype(np.float32)

    def _spec_augment(self, mel):
        mel = mel.copy()
        fill = float(mel.min())
        if mel.shape[0] > self.spec_freq:
            f0 = self.rng.integers(0, mel.shape[0] - self.spec_freq)
            mel[f0:f0 + self.spec_freq, :] = fill
        if mel.shape[1] > self.spec_time:
            t0 = self.rng.integers(0, mel.shape[1] - self.spec_time)
            mel[:, t0:t0 + self.spec_time] = fill
        return mel

    def __call__(self, raw, mel):
        """raw: (10000,) veya (1,10000); mel: (64,201) veya (1,64,201).
        Sekli koruyarak augmente eder. raw uzerindeki zaman-augmentlerinin
        mel'i degistirmedigine dikkat (mel onceden hesaplanmis); bu yuzden
        raw-augment ve mel-augment bagimsiz uygulanir (pratik basitlik)."""
        # raw squeeze
        raw_sq = raw.squeeze() if raw.ndim > 1 else raw
        mel_sq = mel.squeeze() if mel.ndim > 2 else mel

        if self.rng.random() < self.p_amp:
            raw_sq = self._amp_scale(raw_sq)
        if self.rng.random() < self.p_noise:
            raw_sq = self._add_noise(raw_sq)
        if self.rng.random() < self.p_stretch:
            raw_sq = self._time_stretch(raw_sq)
        if self.rng.random() < self.p_spec:
            mel_sq = self._spec_augment(mel_sq)

        return raw_sq.astype(np.float32), mel_sq.astype(np.float32)


# birim test
if __name__ == "__main__":
    print("="*60); print("PCG Augment testleri"); print("="*60)
    aug = PCGAugment(seed=0)
    raw = np.random.standard_normal(10000).astype(np.float32)
    mel = (np.random.standard_normal((64, 201)).astype(np.float32) * 10 - 25)
    for i in range(5):
        r, m = aug(raw, mel)
        assert r.shape == (10000,), f"raw sekil bozuldu: {r.shape}"
        assert m.shape == (64, 201), f"mel sekil bozuldu: {m.shape}"
        assert np.isfinite(r).all() and np.isfinite(m).all(), "nan/inf!"
    print("[OK] 5 augment cagrisi: sekil korundu, nan yok")
    # 1,10000 ve 1,64,201 formati da calismali
    r, m = aug(raw[None, :], mel[None, :, :])
    assert r.shape == (10000,) and m.shape == (64, 201)
    print("[OK] (1,N) / (1,H,W) girdi formati da calisiyor")
    print("\n=== Augment testleri gecti ===")
