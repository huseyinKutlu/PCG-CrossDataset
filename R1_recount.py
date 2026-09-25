#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1_recount.py -- Ek Tablo S4'un sayilarini manifest/split dosyalarindan
yeniden uretir. Hicbir seyi degistirmez.

Kullanim:
    cd ~/Desktop/pcg_project
    python R1_recount.py
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))

def p(*a):
    return os.path.join(ROOT, *a)

def line():
    print("-" * 70)

print("=" * 70)
print("EK TABLO S4 -- YENIDEN HESAPLANAN SAYILAR")
print("=" * 70)

# ---------------------------------------------------------- CirCor (ic)
line()
print("CirCor (internal)")
line()
for name, path in [("CV set (5-fold)", p("splits", "circor2022_5fold_cv.csv")),
                   ("CV set (unknown haric)",
                    p("splits", "circor2022_5fold_cv_nounknown.csv")),
                   ("Held-out test", p("splits", "circor2022_heldout_test.csv")),
                   ("Tum isle(n)mis", p("manifests", "circor2022_processed.csv"))]:
    if not os.path.exists(path):
        print("  %-24s : DOSYA YOK" % name)
        continue
    df = pd.read_csv(path)
    n_rec = len(df)
    n_pat = df["patient_id"].nunique() if "patient_id" in df else None
    n_seg = int(df["n_segments"].sum()) if "n_segments" in df else None
    print("  %-24s : %4d kayit | %s hasta | %s segment"
          % (name, n_rec, n_pat, n_seg))
    if "murmur" in df:
        print("      kayit duzeyi murmur : %s"
              % df["murmur"].value_counts().to_dict())
        pat = df.drop_duplicates("patient_id")
        print("      hasta duzeyi murmur : %s"
              % pat["murmur"].value_counts().to_dict())
        pat_bin = pat["murmur"].map(
            lambda m: "abnormal" if m in ("Present", "Unknown") else "normal")
        print("      hasta duzeyi ikili  : %s"
              % pat_bin.value_counts().to_dict())

# ------------------------------------------------------------ CinC (dis)
line()
print("CinC 2016 (external)")
line()
path = p("manifests", "cinc2016_processed.csv")
if os.path.exists(path):
    df = pd.read_csv(path)
    print("  kayit sayisi   : %d" % len(df))
    print("  segment sayisi : %d" % int(df["n_segments"].sum()))
    print("  sinif dagilimi : %s" % df["label"].value_counts().to_dict())
    print("  alt-veritabani : %s" % df["subset"].value_counts().to_dict())
    print("  sure (s)       : ort %.1f | medyan %.1f | min %.1f | maks %.1f"
          % (df["duration_s"].mean(), df["duration_s"].median(),
             df["duration_s"].min(), df["duration_s"].max()))
else:
    print("  DOSYA YOK")

# ---------------------------------------------------------- Yaseen (dis)
line()
print("Yaseen 2018 (external)")
line()
path = p("manifests", "yaseen_processed.csv")
if os.path.exists(path):
    df = pd.read_csv(path)
    print("  kayit sayisi   : %d" % len(df))
    print("  segment sayisi : %d" % int(df["n_segments"].sum()))
    print("  ikili dagilim  : %s" % df["label"].value_counts().to_dict())
    print("  5 sinif        : %s" % df["class5"].value_counts().to_dict())
else:
    print("  DOSYA YOK")

line()
print("Bu ciktiyi bana gonderin; Ek Tablo S4'un duzeltilmis halini yazayim.")
