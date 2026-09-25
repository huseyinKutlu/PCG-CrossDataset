#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R0_inventory.py  --  JCM revizyonu icin veri envanteri

Kullanim:
    cd ~/Desktop/pcg_project
    python R0_inventory.py > R0_inventory_out.txt 2>&1

Hicbir seyi degistirmez, sadece okur ve ekrana yazar.
Cikan R0_inventory_out.txt dosyasini bana gonderin; hakemlerin
istedigi esik (threshold) / PR-curve / Wilson CI analizlerinin
kodunu tam olarak sizin dosya yapiniza gore yazacagim.
"""

import os
import sys
import json
import glob

ROOT = os.path.abspath(os.path.dirname(__file__))
SEARCH_DIRS = ["reports", "checkpoints", "figures", "manifests", "splits", "cache"]
EXTS = (".npz", ".npy", ".json", ".csv", ".pkl", ".pt")

def hr(title):
    print("\n" + "=" * 78)
    print("== " + title)
    print("=" * 78)

# ---------------------------------------------------------------- 1. dosyalar
hr("1. ILGILI DOSYALAR (npz / npy / json / csv / pkl)")
found = []
for d in SEARCH_DIRS:
    base = os.path.join(ROOT, d)
    if not os.path.isdir(base):
        continue
    for dirpath, dirnames, filenames in os.walk(base):
        for fn in sorted(filenames):
            if fn.lower().endswith(EXTS):
                p = os.path.join(dirpath, fn)
                rel = os.path.relpath(p, ROOT)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    sz = -1
                found.append((rel, sz))
for rel, sz in sorted(found):
    print("%10.1f KB  %s" % (sz / 1024.0, rel))
print("\nTOPLAM: %d dosya" % len(found))

# ------------------------------------------------------------------ 2. npz/npy
hr("2. NPZ / NPY ICERIKLERI (anahtar, shape, dtype, ilk degerler)")
try:
    import numpy as np
except Exception as e:
    print("numpy import edilemedi:", e)
    np = None

if np is not None:
    for rel, sz in sorted(found):
        if not rel.lower().endswith((".npz", ".npy")):
            continue
        p = os.path.join(ROOT, rel)
        print("\n--- %s" % rel)
        try:
            if rel.lower().endswith(".npz"):
                z = np.load(p, allow_pickle=True)
                for k in z.files:
                    a = z[k]
                    try:
                        head = np.asarray(a).ravel()[:5]
                    except Exception:
                        head = "?"
                    print("    %-28s shape=%-18s dtype=%-10s ilk5=%s"
                          % (k, str(getattr(a, "shape", "?")),
                             str(getattr(a, "dtype", "?")), head))
            else:
                a = np.load(p, allow_pickle=True)
                print("    shape=%s dtype=%s ilk5=%s"
                      % (getattr(a, "shape", "?"), getattr(a, "dtype", "?"),
                         np.asarray(a).ravel()[:5]))
        except Exception as e:
            print("    OKUNAMADI:", e)

# -------------------------------------------------------------------- 3. json
hr("3. JSON ICERIKLERI (tam icerik, 6000 karakterle sinirli)")
for rel, sz in sorted(found):
    if not rel.lower().endswith(".json"):
        continue
    p = os.path.join(ROOT, rel)
    print("\n--- %s  (%.1f KB)" % (rel, sz / 1024.0))
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        txt = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
        print(txt[:6000])
        if len(txt) > 6000:
            print("... [KISALTILDI, toplam %d karakter]" % len(txt))
    except Exception as e:
        print("    OKUNAMADI:", e)

# --------------------------------------------------------------------- 4. csv
hr("4. CSV BASLIKLARI VE ILK 3 SATIR")
for rel, sz in sorted(found):
    if not rel.lower().endswith(".csv"):
        continue
    p = os.path.join(ROOT, rel)
    print("\n--- %s  (%.1f KB)" % (rel, sz / 1024.0))
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 4:
                    break
                print("    " + line.rstrip()[:300])
    except Exception as e:
        print("    OKUNAMADI:", e)

# ------------------------------------------------------- 5. kod: kayit yerleri
hr("5. KODDA TAHMIN/OLASILIK KAYIT SATIRLARI (np.savez / to_csv / json.dump)")
PATTERNS = ("np.savez", "np.save(", "to_csv(", "json.dump", "torch.save",
            "probs", "y_prob", "y_score", "predict_proba")
for py in sorted(glob.glob(os.path.join(ROOT, "*.py"))):
    rel = os.path.relpath(py, ROOT)
    if rel == "R0_inventory.py":
        continue
    hits = []
    try:
        with open(py, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if any(pat in s for pat in PATTERNS):
                    hits.append("    %5d: %s" % (i, s[:200]))
    except Exception as e:
        hits = ["    OKUNAMADI: %s" % e]
    if hits:
        print("\n--- %s" % rel)
        print("\n".join(hits[:40]))
        if len(hits) > 40:
            print("    ... [%d satir daha]" % (len(hits) - 40))

# --------------------------------------------------- 6. onemli scriptlerin bas
hr("6. ANAHTAR SCRIPTLERIN ILK 60 SATIRI (arayuz/yol bilgisi icin)")
KEY = ["16_external_val.py", "30_external_stats.py", "34_roc_confusion.py",
       "47_statistics_revision.py", "06_metrics.py", "31_figures.py",
       "43_updated_figures.py", "49_reciprocal_transfer.py"]
for k in KEY:
    p = os.path.join(ROOT, k)
    if not os.path.exists(p):
        print("\n--- %s : YOK" % k)
        continue
    print("\n--- %s" % k)
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if i > 60:
                    break
                print("    %3d| %s" % (i, line.rstrip()[:200]))
    except Exception as e:
        print("    OKUNAMADI:", e)

hr("BITTI")
print("Bu ciktinin tamamini (R0_inventory_out.txt) bana gonderin.")
