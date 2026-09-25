#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R0_inventory.py  --  JCM revizyonu icin veri envanteri (v2, sinirlandirilmis)

Kullanim:
    cd ~/Desktop/pcg_project
    python R0_inventory.py > R0_inventory_out.txt 2>&1

Hicbir seyi degistirmez, sadece okur.
 - reports/ ve checkpoints/  -> TAM inceleme (npz anahtarlari, json icerikleri)
 - cache/                    -> sadece ozet (klasor basina dosya sayisi + 1 ornek)
"""

import os
import json
import glob

ROOT = os.path.abspath(os.path.dirname(__file__))
FULL_DIRS = ["reports", "checkpoints", "manifests", "splits"]
SUMMARY_DIRS = ["cache", "figures"]
EXTS = (".npz", ".npy", ".json", ".csv", ".pkl")

MAX_NPZ_FILES = 80        # tam incelenecek npz sayisi
MAX_JSON_CHARS = 6000     # json basina karakter
MAX_JSON_FILES = 60

def hr(title):
    print("\n" + "=" * 78)
    print("== " + title)
    print("=" * 78)

def collect(dirs):
    out = []
    for d in dirs:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in sorted(filenames):
                if fn.lower().endswith(EXTS):
                    p = os.path.join(dirpath, fn)
                    try:
                        sz = os.path.getsize(p)
                    except OSError:
                        sz = -1
                    out.append((os.path.relpath(p, ROOT), sz))
    return sorted(out)

try:
    import numpy as np
except Exception as e:
    print("numpy import edilemedi:", e)
    np = None

def describe_npz(rel):
    p = os.path.join(ROOT, rel)
    print("\n--- %s" % rel)
    if np is None:
        return
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

# ================================================== 1. TAM TARANAN KLASORLER
full = collect(FULL_DIRS)

hr("1. reports / checkpoints / manifests / splits  --  DOSYA LISTESI")
for rel, sz in full:
    print("%10.1f KB  %s" % (sz / 1024.0, rel))
print("\nTOPLAM: %d dosya" % len(full))

hr("2. NPZ / NPY ICERIKLERI (anahtar, shape, dtype, ilk degerler)")
npzs = [r for r in full if r[0].lower().endswith((".npz", ".npy"))]
for rel, _sz in npzs[:MAX_NPZ_FILES]:
    describe_npz(rel)
if len(npzs) > MAX_NPZ_FILES:
    print("\n[... %d npz/npy dosyasi daha var, listelenmedi]"
          % (len(npzs) - MAX_NPZ_FILES))

hr("3. JSON ICERIKLERI")
jsons = [r for r in full if r[0].lower().endswith(".json")]
for rel, sz in jsons[:MAX_JSON_FILES]:
    p = os.path.join(ROOT, rel)
    print("\n--- %s  (%.1f KB)" % (rel, sz / 1024.0))
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        txt = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
        print(txt[:MAX_JSON_CHARS])
        if len(txt) > MAX_JSON_CHARS:
            print("... [KISALTILDI, toplam %d karakter]" % len(txt))
    except Exception as e:
        print("    OKUNAMADI:", e)
if len(jsons) > MAX_JSON_FILES:
    print("\n[... %d json dosyasi daha var]" % (len(jsons) - MAX_JSON_FILES))

hr("4. CSV BASLIKLARI VE ILK 3 SATIR")
for rel, sz in full:
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

# ===================================================== 5. OZET KLASORLER
hr("5. cache / figures  --  SADECE OZET")
for d in SUMMARY_DIRS:
    base = os.path.join(ROOT, d)
    if not os.path.isdir(base):
        print("\n%s : YOK" % d)
        continue
    for dirpath, _dn, filenames in os.walk(base):
        rel = os.path.relpath(dirpath, ROOT)
        rel_files = [f for f in filenames if f.lower().endswith(EXTS)]
        if not rel_files:
            continue
        total = sum(os.path.getsize(os.path.join(dirpath, f))
                    for f in rel_files)
        print("\n%s : %d dosya, %.1f MB" % (rel, len(rel_files), total / 1e6))
        print("    ornekler: %s" % ", ".join(sorted(rel_files)[:3]))
        # bir ornek npz'nin yapisi
        sample = next((f for f in sorted(rel_files)
                       if f.lower().endswith(".npz")), None)
        if sample:
            describe_npz(os.path.join(rel, sample))

# ============================================ 6. KODDA KAYIT SATIRLARI
hr("6. KODDA TAHMIN/OLASILIK KAYIT SATIRLARI")
PATTERNS = ("np.savez", "np.save(", "to_csv(", "json.dump",
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
        print("\n".join(hits[:30]))
        if len(hits) > 30:
            print("    ... [%d satir daha]" % (len(hits) - 30))

# ======================================= 7. ANAHTAR SCRIPTLERIN BASLANGICI
hr("7. ANAHTAR SCRIPTLERIN ILK 60 SATIRI")
KEY = ["16_external_val.py", "30_external_stats.py", "34_roc_confusion.py",
       "47_statistics_revision.py", "06_metrics.py", "43_updated_figures.py",
       "49_reciprocal_transfer.py"]
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
print("R0_inventory_out.txt dosyasini gonderin.")
