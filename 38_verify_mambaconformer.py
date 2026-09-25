#!/usr/bin/env python3
"""
38_verify_mambaconformer.py
MambaConformer'in Yaseen'deki cok-iyi sonucunu (AUROC 0.972) DOGRULAR.

Bir Q1 hakeminin soracagi kontroller — yayindan ONCE biz yapariz.

Kontroller:
  1. LEAKAGE: Yaseen record'lari CirCor egitim setinde var mi? (olmamali)
  2. ETIKET YONU: Yaseen ham confusion matrix (Spec 1.00 gercek mi)
  3. PER-FOLD: her fold'un Yaseen AUROC'u ayri (ensemble sansli mi)
  4. CWT KONTROL: Yaseen CWT'leri makul mu (bos/bozuk degil)
  5. SKOR DAGILIMI: normal vs abnormal olasilik ayrisiyor mu
  6. CINC karsilastirma: ayni model CinC'te neden coktu

Kullanim:
  python 38_verify_mambaconformer.py
"""
from __future__ import annotations
from pathlib import Path
from importlib import import_module
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

ext_mod = import_module("16_external_val")
ds_mod = import_module("05_dataset")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
CKPT = PROJECT_ROOT / "checkpoints" / "binary"
CACHE = PROJECT_ROOT / "cache"


def log(m): print(m, flush=True)


# ---- Kontrol 1: LEAKAGE ----
def check_leakage():
    log("\n" + "="*60)
    log("KONTROL 1: LEAKAGE (Yaseen record'lari CirCor'da var mi?)")
    log("="*60)
    import pandas as pd
    cv = pd.read_csv(PROJECT_ROOT / "splits" / "circor2022_5fold_cv.csv")
    circor_ids = set(cv["record_id"].astype(str))
    yas_files = glob.glob(str(CACHE / "yaseen" / "*.npz"))
    yas_ids = set(Path(f).stem for f in yas_files)
    overlap = circor_ids & yas_ids
    log(f"  CirCor record sayisi: {len(circor_ids)}")
    log(f"  Yaseen record sayisi: {len(yas_ids)}")
    log(f"  ORTAK record (leakage): {len(overlap)}")
    if overlap:
        log(f"  !!! UYARI: {len(overlap)} ortak record! Ornekler: {list(overlap)[:5]}")
    else:
        log("  [OK] Hic ortak record yok — leakage YOK. Yaseen tamamen zero-shot.")
    # cache yollari da farkli mi?
    log(f"  CirCor cache: cache/circor2022/, Yaseen cache: cache/yaseen/ (ayri dizinler)")
    return len(overlap) == 0


# ---- Kontrol 4: CWT makul mu ----
def check_cwt():
    log("\n" + "="*60)
    log("KONTROL 4: Yaseen CWT'leri makul mu (bos/bozuk degil)")
    log("="*60)
    yas = sorted(glob.glob(str(CACHE / "yaseen" / "*.npz")))[:20]
    stats = []
    for f in yas:
        d = np.load(f, allow_pickle=True)
        if "cwt_segs" not in d:
            log(f"  !!! {Path(f).name}: cwt_segs YOK!"); continue
        c = d["cwt_segs"]
        stats.append((c.shape, float(c.mean()), float(c.std()), float(c.min()), float(c.max())))
    if stats:
        shapes = set(s[0] for s in stats)
        means = [s[1] for s in stats]; stds = [s[2] for s in stats]
        log(f"  Incelenen: {len(stats)} dosya")
        log(f"  CWT sekilleri: {shapes} (hepsi (n,64,201) olmali)")
        log(f"  mean araligi: [{min(means):.3f}, {max(means):.3f}] (normalize ~0)")
        log(f"  std araligi: [{min(stds):.3f}, {max(stds):.3f}] (normalize ~1)")
        # mel ile CWT farkli mi (yanlislikla mel kopyasi degil mi)
        d0 = np.load(yas[0], allow_pickle=True)
        if "mel_segs" in d0 and "cwt_segs" in d0:
            corr = np.corrcoef(d0["mel_segs"][0].ravel(), d0["cwt_segs"][0].ravel())[0, 1]
            log(f"  CWT-mel korelasyonu: {corr:.3f} (1.0 ise CWT=mel kopyasi=SORUN)")
            if abs(corr) > 0.95:
                log("  !!! UYARI: CWT mel'in kopyasi olabilir!")
            else:
                log("  [OK] CWT mel'den farkli (gercek CWT).")
    return True


# ---- Kontrol 2,3,5: tahminlerle ----
@torch.no_grad()
def predict_per_fold(ExtDS, manifest):
    """Her fold modeliyle ayri tahmin + ensemble. record-seviyesi."""
    ds = ExtDS(manifest, PROJECT_ROOT, norm_adapt="none")
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    per_fold_p, ref_y, ref_id = [], None, None
    for fold in range(N_FOLDS):
        ck = CKPT / f"mambaconformer_fold{fold}.pth"
        if not ck.exists():
            log(f"  !! fold{fold} checkpoint yok"); continue
        model, mode = ext_mod.build_binary_model("mambaconformer")
        model = model.to(DEVICE)
        sd = torch.load(ck, map_location=DEVICE)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        model.eval()
        sp, sy, si = [], [], []
        for b in loader:
            logits = model(b["raw"].to(DEVICE), b["mel"].to(DEVICE), b["cwt"].to(DEVICE))
            sp.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            sy.append(b["label"].numpy()); si.extend(b["record_id"])
        sp = np.concatenate(sp); sy = np.concatenate(sy); si = np.array(si)
        uniq = np.unique(si)
        rp = np.array([sp[si == u].mean() for u in uniq])
        ry = np.array([sy[si == u][0] for u in uniq])
        per_fold_p.append(rp)
        if ref_y is None: ref_y, ref_id = ry, uniq
    return np.array(per_fold_p), ref_y, ref_id


def check_predictions(name, manifest_name):
    log("\n" + "="*60)
    log(f"KONTROL 2,3,5: {name} tahmin analizi")
    log("="*60)
    ExtDS = ext_mod.YaseenDataset if "yaseen" in manifest_name else ext_mod.CinCDataset
    manifest = PROJECT_ROOT / "manifests" / manifest_name
    pf, y, ids = predict_per_fold(ExtDS, manifest)
    if pf is None or len(pf) == 0:
        log("  tahmin yok"); return
    # Kontrol 3: per-fold AUROC
    log("  -- Kontrol 3: per-fold AUROC (ensemble sansli mi?) --")
    for i, p in enumerate(pf):
        log(f"     fold {i}: AUROC={roc_auc_score(y, p):.4f}")
    ens = pf.mean(0)
    log(f"     ENSEMBLE: AUROC={roc_auc_score(y, ens):.4f}")
    log(f"     per-fold std: {np.std([roc_auc_score(y, p) for p in pf]):.4f}")
    # Kontrol 2: ham confusion matrix (esik 0.5)
    log("  -- Kontrol 2: ham confusion matrix (ensemble, esik 0.5) --")
    pred = (ens >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    log(f"     True\\Pred  Normal  Abnormal")
    log(f"     Normal     {cm[0,0]:5d}   {cm[0,1]:5d}")
    log(f"     Abnormal   {cm[1,0]:5d}   {cm[1,1]:5d}")
    sens = cm[1,1]/max(cm[1].sum(),1); spec = cm[0,0]/max(cm[0].sum(),1)
    log(f"     Sens={sens:.3f} Spec={spec:.3f}")
    log(f"     normal n={int((y==0).sum())}, abnormal n={int((y==1).sum())}")
    # Kontrol 5: skor dagilimi
    log("  -- Kontrol 5: skor dagilimi (siniflar ayrisiyor mu?) --")
    p_norm = ens[y == 0]; p_abn = ens[y == 1]
    log(f"     normal kayitlar olasilik: ort={p_norm.mean():.3f} (dusuk olmali)")
    log(f"     abnormal kayitlar olasilik: ort={p_abn.mean():.3f} (yuksek olmali)")
    log(f"     ayrisma (abn-norm): {p_abn.mean()-p_norm.mean():.3f} (buyuk=iyi)")


def main():
    log("#"*60)
    log("# MAMBACONFORMER YASEEN SONUCU DOGRULAMA")
    log("#"*60)
    no_leak = check_leakage()
    check_cwt()
    check_predictions("YASEEN", "yaseen_processed.csv")
    check_predictions("CINC (karsilastirma)", "cinc2016_processed.csv")
    log("\n" + "#"*60)
    log("# DOGRULAMA TAMAMLANDI — yukaridaki kontrolleri yorumlayalim")
    log("#"*60)


if __name__ == "__main__":
    main()
