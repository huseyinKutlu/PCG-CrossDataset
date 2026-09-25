#!/usr/bin/env python3
"""
05_dataset.py
PCG Q1 Manuscript — CirCor 2022 segment-bazli Dataset + DataLoader.

TASARIM:
  - Egitim SEGMENT duzeyinde (her .npz icinde n_seg adet 5s pencere).
  - Degerlendirme HASTA duzeyinde raporlanir (CirCor metrigi hasta duzeyi).
    -> Bu yuzden her segment record_id + patient_id tasir; agregasyon
       yardimcilari (segment -> record -> patient) burada tanimli.
  - Hibrit girdi: raw (1,10000) + mel (1,64,201).
  - Resmi etiket sirasi: Present=0, Unknown=1, Absent=2.
  - Lazy-load: npz __getitem__'de acilir (24k segment RAM'e zorlanmaz),
    opsiyonel LRU cache ile hizlandirma.

Kullanim:
  from importlib import import_module
  ds_mod = import_module("05_dataset")  # dosya adi rakamla basladigi icin
  train_ds = ds_mod.CircorSegmentDataset(cv_csv, fold=0, split="train",
                                         project_root=".")
"""
from __future__ import annotations
import sys
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Resmi murmur sinif sirasi
MURMUR_TO_IDX = {"Present": 0, "Unknown": 1, "Absent": 2}
IDX_TO_MURMUR = {v: k for k, v in MURMUR_TO_IDX.items()}
N_CLASSES = 3

RAW_LEN = 10000      # 5s * 2000 Hz
MEL_SHAPE = (64, 201)


# -----------------------------------------------------------------------------
# npz okuma (lazy + cache)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=256)
def _load_npz_cached(abs_path: str):
    """Bir kaydin tum segmentlerini yukler. LRU cache ile tekrar okumayi onler.
    256 kayit ~ birkac yuz MB; RTX 6000 / 128GB RAM icin guvenli."""
    d = np.load(abs_path, allow_pickle=True)
    return d["raw_segs"].astype(np.float32), d["mel_segs"].astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CircorSegmentDataset(Dataset):
    """
    Segment-duzeyi CirCor dataset.

    Parametreler
    ----------
    split_csv   : 5fold_cv.csv veya heldout_test.csv yolu
    project_root: cache_path'lerin coz uldugu kok (npz'ler buraya gore)
    fold        : CV kullaniminda hangi fold (0..4); None ise tum CSV.
    split       : "train" | "val" | "all"
                  - "train": fold != verilen fold (CV egitimi)
                  - "val"  : fold == verilen fold (CV dogrulama)
                  - "all"  : tum satirlar (held-out test icin)
    return_mel  : mel dondurulsun mu (hibrit model icin True)
    return_raw  : raw dondurulsun mu (hibrit model icin True)
    """
    def __init__(self,
                 split_csv: str | Path,
                 project_root: str | Path,
                 fold: Optional[int] = None,
                 split: str = "all",
                 return_raw: bool = True,
                 return_mel: bool = True):
        self.project_root = Path(project_root)
        self.return_raw = return_raw
        self.return_mel = return_mel

        df = pd.read_csv(split_csv)
        df["patient_id"] = df["patient_id"].astype(str)

        if split == "train":
            assert fold is not None and "fold" in df.columns
            df = df[df["fold"] != fold].reset_index(drop=True)
        elif split == "val":
            assert fold is not None and "fold" in df.columns
            df = df[df["fold"] == fold].reset_index(drop=True)
        elif split == "all":
            df = df.reset_index(drop=True)
        else:
            raise ValueError(f"Bilinmeyen split: {split}")

        self.records = df

        # Segment indeksi insa et: (record_satir_idx, segment_idx_in_record)
        # n_segments manifest'te zaten var -> npz acmadan indeks kurulabilir.
        self.index = []   # list of (rec_row, seg_in_rec)
        for rec_row, n_seg in enumerate(df["n_segments"].astype(int).tolist()):
            for s in range(n_seg):
                self.index.append((rec_row, s))

        # Etiketler (segment duzeyinde, sampler/agirlik icin)
        self.seg_labels = np.array(
            [MURMUR_TO_IDX[df.iloc[r]["murmur"]] for (r, _) in self.index],
            dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        rec_row, seg_idx = self.index[i]
        rec = self.records.iloc[rec_row]
        abs_path = str((self.project_root / rec["cache_path"]).resolve())
        raw_all, mel_all = _load_npz_cached(abs_path)

        out = {
            "label": int(MURMUR_TO_IDX[rec["murmur"]]),
            "record_id": rec["record_id"],
            "patient_id": rec["patient_id"],
            "seg_idx": seg_idx,
        }
        if self.return_raw:
            raw = torch.from_numpy(raw_all[seg_idx]).float().unsqueeze(0)  # (1,10000)
            out["raw"] = raw
        if self.return_mel:
            mel = torch.from_numpy(mel_all[seg_idx]).float().unsqueeze(0)  # (1,64,201)
            out["mel"] = mel
        if getattr(self, "return_cwt", False):
            # CWT ayri yuklenir (cache fonksiyonunu bozmamak icin dogrudan)
            d = np.load(abs_path, allow_pickle=True)
            if "cwt_segs" in d:
                cwt = torch.from_numpy(d["cwt_segs"][seg_idx].astype(np.float32)).unsqueeze(0)
                out["cwt"] = cwt
        return out

    # --- sinif agirligi (segment-bazli) ---
    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.seg_labels, minlength=N_CLASSES).astype(np.float64)
        total = counts.sum()
        w = total / (N_CLASSES * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32)

    # --- WeightedRandomSampler icin ornek agirliklari ---
    def sample_weights(self) -> torch.DoubleTensor:
        cw = self.class_weights().numpy()
        sw = cw[self.seg_labels]
        return torch.DoubleTensor(sw)


# -----------------------------------------------------------------------------
# DataLoader fabrikalari
# -----------------------------------------------------------------------------
def make_train_loader(ds: CircorSegmentDataset, batch_size=64,
                      num_workers=8, balanced=True) -> DataLoader:
    if balanced:
        sampler = WeightedRandomSampler(ds.sample_weights(),
                                        num_samples=len(ds),
                                        replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def make_eval_loader(ds: CircorSegmentDataset, batch_size=128,
                     num_workers=8) -> DataLoader:
    # Eval'de shuffle YOK, drop_last YOK (her segment degerlendirilmeli)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# -----------------------------------------------------------------------------
# Segment -> Hasta agregasyonu (CirCor metrigi icin KRITIK)
# -----------------------------------------------------------------------------
def aggregate_to_patient(seg_probs: np.ndarray,
                         patient_ids: list[str],
                         method: str = "mean") -> tuple[np.ndarray, list[str]]:
    """
    Segment olasiliklarini hasta duzeyine toplar.

    seg_probs   : (n_segments, 3) softmax olasiliklar
    patient_ids : len n_segments, her segmentin hastasi
    method      : "mean" (olasilik ortalamasi) | "max_present" (Present
                  olasiliginin maksimumu — tarama/duyarlilik odakli)

    Donus: (patient_probs (n_patients,3), patient_id_sirasi)
    """
    df = pd.DataFrame(seg_probs, columns=["p_present", "p_unknown", "p_absent"])
    df["patient_id"] = patient_ids

    if method == "mean":
        agg = df.groupby("patient_id").mean()
    elif method == "max_present":
        # Present olasiligini maksimize et (kacirma maliyeti yuksek senaryosu)
        agg = df.groupby("patient_id").agg({
            "p_present": "max", "p_unknown": "mean", "p_absent": "mean"})
        # yeniden normalize
        agg = agg.div(agg.sum(axis=1), axis=0)
    else:
        raise ValueError(method)

    pids = agg.index.tolist()
    return agg[["p_present", "p_unknown", "p_absent"]].to_numpy(), pids


def patient_true_labels(split_csv: str | Path) -> dict[str, int]:
    """Hasta -> gercek murmur index. Agregasyon sonrasi karsilastirma icin."""
    df = pd.read_csv(split_csv)
    df["patient_id"] = df["patient_id"].astype(str)
    pat = df.groupby("patient_id")["murmur"].first()
    return {pid: MURMUR_TO_IDX[m] for pid, m in pat.items()}


# -----------------------------------------------------------------------------
# Kendi kendine test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="proje koku")
    ap.add_argument("--cv", default="splits/circor2022_5fold_cv.csv")
    ap.add_argument("--test", default="splits/circor2022_heldout_test.csv")
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / args.cv
    test_csv = root / args.test

    print("="*70)
    print("Dataset smoke test")
    print("="*70)

    tr = CircorSegmentDataset(cv_csv, project_root=root, fold=0, split="train")
    va = CircorSegmentDataset(cv_csv, project_root=root, fold=0, split="val")
    te = CircorSegmentDataset(test_csv, project_root=root, split="all")

    print(f"Train(fold0): {len(tr)} segment, {tr.records['patient_id'].nunique()} hasta")
    print(f"Val(fold0)  : {len(va)} segment, {va.records['patient_id'].nunique()} hasta")
    print(f"Test        : {len(te)} segment, {te.records['patient_id'].nunique()} hasta")

    print("\nSegment sinif dagilimi (train fold0):")
    for idx in range(N_CLASSES):
        print(f"   {IDX_TO_MURMUR[idx]}: {(tr.seg_labels==idx).sum()}")
    print("Sinif agirliklari (train fold0):", tr.class_weights().tolist())

    # patient-disjoint dogrula (train ile val ortak hasta olmamali)
    ov = set(tr.records['patient_id']) & set(va.records['patient_id'])
    assert len(ov) == 0, f"SIZINTI: train/val ortak hasta {list(ov)[:5]}"
    print("\n[OK] train/val patient-disjoint")

    # bir batch cek
    loader = make_train_loader(tr, batch_size=8, num_workers=0, balanced=True)
    batch = next(iter(loader))
    print("\nBir batch:")
    print("   raw:", tuple(batch["raw"].shape), batch["raw"].dtype)
    print("   mel:", tuple(batch["mel"].shape), batch["mel"].dtype)
    print("   label:", batch["label"].tolist())
    print("   patient_id ornek:", batch["patient_id"][:3])

    # agregasyon testi: rastgele olasilikla
    print("\nAgregasyon testi (val fold0, rastgele prob):")
    eval_loader = make_eval_loader(va, batch_size=128, num_workers=0)
    all_pids = []
    for b in eval_loader:
        all_pids.extend(b["patient_id"])
    rng = np.random.default_rng(0)
    fake_probs = rng.dirichlet([1,1,1], size=len(all_pids))
    pat_probs, pids = aggregate_to_patient(fake_probs, all_pids, method="mean")
    print(f"   {len(all_pids)} segment -> {len(pids)} hasta")
    print(f"   pat_probs shape: {pat_probs.shape}, satir toplami ~1: "
          f"{np.allclose(pat_probs.sum(1), 1.0)}")
    truth = patient_true_labels(cv_csv)
    print(f"   gercek etiket eslemesi: {len(truth)} hasta")
    print("\n=== Dataset testi gecti ===")
