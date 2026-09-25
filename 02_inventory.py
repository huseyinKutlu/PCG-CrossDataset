#!/usr/bin/env python3
"""02_inventory.py v2 - script'in bulundugu klasoru proje koku kabul eder"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

try:
    import soundfile as sf
except ImportError:
    print("HATA: pip install soundfile")
    sys.exit(1)

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
CINC_DIR     = RAW_DIR / "cinc2016"
CIRCOR_DIR   = RAW_DIR / "circor2022"
MANIFEST_DIR = PROJECT_ROOT / "manifests"
REPORT_DIR   = PROJECT_ROOT / "reports"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_LINES = []
def log(m):
    print(m); REPORT_LINES.append(m)

def inventory_cinc():
    log("\n" + "="*70 + "\nCinC 2016\n" + "="*70)
    if not CINC_DIR.exists():
        log(f"!! YOK: {CINC_DIR}"); return pd.DataFrame()
    roots = sorted({p for p in CINC_DIR.rglob("training-*") if p.is_dir()})
    if not roots:
        log("!! 'training-*' klasoru bulunamadi"); return pd.DataFrame()
    log(f"   training-* klasor: {len(roots)}")
    rows = []
    for sr_dir in roots:
        ref = sr_dir / "REFERENCE.csv"
        labels = {}
        if ref.exists():
            r = pd.read_csv(ref, header=None, names=["record","label"])
            labels = dict(zip(r["record"].astype(str), r["label"].astype(int)))
        wavs = sorted(sr_dir.glob("*.wav"))
        log(f"   {sr_dir.name}: {len(wavs)} wav | {len(labels)} etiket")
        for wp in tqdm(wavs, desc=f"   {sr_dir.name}", leave=False):
            try:
                info = sf.info(str(wp))
                lab = labels.get(wp.stem, np.nan)
                rows.append({
                    "dataset":"cinc2016", "subset":sr_dir.name,
                    "record_id":wp.stem, "path":str(wp.relative_to(PROJECT_ROOT)),
                    "sample_rate":info.samplerate,
                    "duration_s":info.frames/info.samplerate,
                    "label_raw":lab,
                    "label":{1:"abnormal",-1:"normal"}.get(lab,"unknown"),
                })
            except Exception as e:
                log(f"      hata: {wp.name}: {e}")
    df = pd.DataFrame(rows)
    if df.empty: return df
    log(f"\n   Toplam: {len(df)}")
    log(f"   SR: {dict(Counter(df['sample_rate']))}")
    log(f"   Sure(s): min={df['duration_s'].min():.1f}, max={df['duration_s'].max():.1f}, "
        f"medyan={df['duration_s'].median():.1f}, toplam={df['duration_s'].sum()/3600:.1f}h")
    log(f"   Etiket:\n{df['label'].value_counts().to_string()}")
    log(f"   Subset x Label:\n{pd.crosstab(df['subset'], df['label']).to_string()}")
    out = MANIFEST_DIR / "cinc2016_manifest.csv"
    df.to_csv(out, index=False)
    log(f"   manifest: {out}")
    return df

def inventory_circor():
    log("\n" + "="*70 + "\nCirCor 2022\n" + "="*70)
    if not CIRCOR_DIR.exists():
        log(f"!! YOK: {CIRCOR_DIR}"); return pd.DataFrame()
    tr_roots = [p for p in CIRCOR_DIR.rglob("training_data") if p.is_dir()]
    if not tr_roots:
        log("!! training_data klasoru yok"); return pd.DataFrame()
    tr = tr_roots[0]
    log(f"   training_data: {tr}")
    pcsv = list(CIRCOR_DIR.rglob("training_data.csv"))
    pdf = pd.DataFrame()
    if pcsv:
        pdf = pd.read_csv(pcsv[0])
        log(f"   patient CSV: {len(pdf)} hasta, sutunlar: {list(pdf.columns)}")
    pmap = {}
    if not pdf.empty:
        for _, r in pdf.iterrows():
            pmap[str(r.get("Patient ID",""))] = {
                "murmur":str(r.get("Murmur","")),
                "outcome":str(r.get("Outcome","")),
                "age":r.get("Age",np.nan),
                "sex":r.get("Sex",""),
            }
    rows = []
    wavs = sorted(tr.glob("*.wav"))
    log(f"   wav: {len(wavs)}")
    for wp in tqdm(wavs, desc="   CirCor", leave=False):
        parts = wp.stem.split("_")
        pid = parts[0]; loc = "_".join(parts[1:]) if len(parts)>1 else ""
        try:
            info = sf.info(str(wp))
            pi = pmap.get(pid, {})
            rows.append({
                "dataset":"circor2022", "subset":"training_data",
                "record_id":wp.stem, "patient_id":pid, "location":loc,
                "path":str(wp.relative_to(PROJECT_ROOT)),
                "sample_rate":info.samplerate,
                "duration_s":info.frames/info.samplerate,
                "murmur":pi.get("murmur","Unknown"),
                "outcome":pi.get("outcome","Unknown"),
                "age":pi.get("age",np.nan), "sex":pi.get("sex",""),
            })
        except Exception as e:
            log(f"      hata: {wp.name}: {e}")
    df = pd.DataFrame(rows)
    if df.empty: return df
    log(f"\n   Toplam kayit: {len(df)}, hasta: {df['patient_id'].nunique()}")
    log(f"   SR: {dict(Counter(df['sample_rate']))}")
    log(f"   Sure(s): min={df['duration_s'].min():.1f}, max={df['duration_s'].max():.1f}, "
        f"medyan={df['duration_s'].median():.1f}")
    log(f"   Murmur:\n{df['murmur'].value_counts().to_string()}")
    log(f"   Outcome:\n{df['outcome'].value_counts().to_string()}")
    log(f"   Lokasyon:\n{df['location'].value_counts().to_string()}")
    out = MANIFEST_DIR / "circor2022_manifest.csv"
    df.to_csv(out, index=False)
    log(f"   manifest: {out}")
    return df

def plot_dist(c, cc):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    if not c.empty:
        sns.countplot(data=c, x="label", ax=ax[0,0], order=c["label"].value_counts().index)
        ax[0,0].set_title(f"CinC 2016 (n={len(c)})")
        sns.countplot(data=c, x="subset", hue="label", ax=ax[0,1],
                      order=sorted(c["subset"].unique()))
        ax[0,1].set_title("CinC alt-set")
        ax[0,1].tick_params(axis="x", rotation=30)
    if not cc.empty:
        sns.countplot(data=cc, x="murmur", ax=ax[1,0], order=cc["murmur"].value_counts().index)
        ax[1,0].set_title(f"CirCor Murmur (n={len(cc)})")
        sns.countplot(data=cc, x="outcome", ax=ax[1,1], order=cc["outcome"].value_counts().index)
        ax[1,1].set_title("CirCor Outcome")
    plt.tight_layout()
    out = REPORT_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"   gorsel: {out}")

if __name__ == "__main__":
    log(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log(f"Python = {sys.executable}")
    c = inventory_cinc()
    cc = inventory_circor()
    if not c.empty or not cc.empty: plot_dist(c, cc)
    rp = REPORT_DIR / "inventory_report.txt"
    rp.write_text("\n".join(REPORT_LINES), encoding="utf-8")
    print(f"\nrapor: {rp}")
