#!/usr/bin/env python3
"""
diag_labels.py
Teshis: CinC ve Yaseen loader'larinda etiket alani dogru geliyor mu?
44/45 script'lerinde 'tek sinif' ve 'normal bulunamadi' sorununun kokeni.
"""
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader

ev = import_module("16_external_val")
PROJECT_ROOT = Path(__file__).resolve().parent

def inspect(name, ds, n=300):
    print(f"\n=== {name} ===")
    # ilk batch'in anahtarlari
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    b0 = next(iter(loader))
    print("  batch anahtarlari:", list(b0.keys()))
    for k,v in b0.items():
        if torch.is_tensor(v):
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype} ornek={v.flatten()[:6].tolist()}")
        else:
            print(f"    {k}: {type(v)} ornek={v[:4] if hasattr(v,'__getitem__') else v}")
    # etiket dagilimi
    if "label" in b0:
        labs = []
        loader2 = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        seen=0
        for b in loader2:
            raw_lab = b["label"]
            bl = ev.to_binary_labels(raw_lab)
            labs.append(bl.numpy()); seen+=len(raw_lab)
            if seen>=n: break
        labs = np.concatenate(labs)[:n]
        uniq, cnt = np.unique(labs, return_counts=True)
        print(f"  to_binary_labels dagilimi (ilk {n}): {dict(zip(uniq.tolist(), cnt.tolist()))}")
        # ham label da gor
        raw_all=[]
        seen=0
        for b in loader2:
            raw_all.append(b["label"].numpy() if torch.is_tensor(b["label"]) else np.array(b["label"]))
            seen+=len(raw_all[-1])
            if seen>=n: break
        raw_all=np.concatenate(raw_all)[:n]
        ru,rc=np.unique(raw_all,return_counts=True)
        print(f"  HAM label dagilimi (ilk {n}): {dict(zip(ru.tolist(), rc.tolist()))}")
    else:
        print("  !!! 'label' anahtari YOK — bu yuzden tek sinif gorunuyor")

# CinC
cinc = ev.CinCDataset(PROJECT_ROOT/"manifests"/"cinc2016_processed.csv", PROJECT_ROOT, norm_adapt="none")
inspect("CinC", cinc)
# Yaseen
yas = ev.YaseenDataset(PROJECT_ROOT/"manifests"/"yaseen_processed.csv", PROJECT_ROOT, norm_adapt="none")
inspect("Yaseen", yas)

print("\n=== to_binary_labels fonksiyonu ne yapiyor? ===")
import inspect as _ins
try:
    print(_ins.getsource(ev.to_binary_labels))
except Exception as e:
    print("kaynak alinamadi:", e)
