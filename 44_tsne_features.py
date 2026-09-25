#!/usr/bin/env python3
"""
44_tsne_features.py
XAI-A: Oznitelik-uzayi gorsellestirme (t-SNE).

MambaConformer'in fusion-oncesi (penultimate) ozelliklerini uc dataset
icin cikarir, 2B t-SNE ile gosterir. Amac: domain hizalanmasini gormek —
Yaseen (homojen) CirCor'a yakin mi, CinC (heterojen) dagilmis mi?

Cikti: figures/F16_tsne_domains.png (+pdf), 300 dpi
  Panel A: dataset'e gore renklendirme (CirCor/CinC/Yaseen)
  Panel B: sinif'a gore renklendirme (normal/abnormal)

Kullanim:
  python 44_tsne_features.py            # fold 0 checkpoint, ~2000 ornek/dataset
  python 44_tsne_features.py --max 1500
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev = import_module("16_external_val")
ds_mod = import_module("05_dataset")
mc_mod = import_module("37_mambaconformer")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = PROJECT_ROOT / "figures"; FIG.mkdir(exist_ok=True)
CKPT = PROJECT_ROOT / "checkpoints" / "binary"


def load_mc_feature_extractor(fold=0):
    """MambaConformer'i yukle, fusion-oncesi ozellik cikaran sarmalayici dondur."""
    model, _ = ev.build_binary_model("mambaconformer")
    ck = CKPT / f"mambaconformer_fold{fold}.pth"
    sd = torch.load(ck, map_location=DEVICE)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model = model.to(DEVICE).eval()
    return model


@torch.no_grad()
def extract_features(model, loader, mode_keys, max_n, is_circor=False):
    """Fusion-oncesi ozellikleri topla.
    is_circor=True: label 3-sinif (CirCor) -> to_binary_labels ile cevir
    is_circor=False: label zaten ikili (CinC/Yaseen) -> dogrudan kullan
    """
    feats, labels = [], []
    n = 0
    for b in loader:
        raw = b["raw"].to(DEVICE)
        mel = b["mel"].to(DEVICE) if "mel" in b else None
        cwt = b["cwt"].to(DEVICE) if "cwt" in b else None
        if hasattr(model, "extract_features"):
            f = model.extract_features(raw, mel, cwt)
        else:
            f = model(raw, mel, cwt)
        feats.append(f.cpu().numpy())
        if "label" in b:
            if is_circor:
                lab = ev.to_binary_labels(b["label"]).numpy()
            else:
                lab = b["label"].numpy()  # CinC/Yaseen zaten ikili (0=normal,1=abnormal)
        else:
            lab = np.zeros(len(raw))
        labels.append(lab)
        n += len(raw)
        if n >= max_n: break
    X = np.concatenate(feats)[:max_n]
    y = np.concatenate(labels)[:max_n]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--max", type=int, default=1500)
    args = ap.parse_args()

    print("MambaConformer ozellik cikarici yukleniyor...")
    model = load_mc_feature_extractor(args.fold)
    return_cwt = True

    # CirCor (internal) — val fold
    cv = PROJECT_ROOT / "splits" / "circor2022_5fold_cv.csv"
    circ = ds_mod.CircorSegmentDataset(cv, project_root=PROJECT_ROOT, fold=args.fold, split="val")
    circ.return_cwt = return_cwt
    circ_loader = ds_mod.make_eval_loader(circ, batch_size=64, num_workers=4)
    # CinC + Yaseen
    cinc = ev.CinCDataset(PROJECT_ROOT/"manifests"/"cinc2016_processed.csv", PROJECT_ROOT, norm_adapt="none")
    yas = ev.YaseenDataset(PROJECT_ROOT/"manifests"/"yaseen_processed.csv", PROJECT_ROOT, norm_adapt="none")
    cinc_loader = DataLoader(cinc, batch_size=64, shuffle=True, num_workers=4)
    yas_loader = DataLoader(yas, batch_size=64, shuffle=True, num_workers=4)

    print("Ozellikler cikariliyor (3 dataset)...")
    Xc, yc = extract_features(model, circ_loader, None, args.max, is_circor=True)
    Xi, yi = extract_features(model, cinc_loader, None, args.max, is_circor=False)
    Xy, yy = extract_features(model, yas_loader, None, args.max, is_circor=False)
    print(f"  CirCor {Xc.shape}, CinC {Xi.shape}, Yaseen {Xy.shape}")
    # etiket dagilimi kontrolu (tek sinif uyarisi)
    for nm, yy_ in [("CirCor",yc),("CinC",yi),("Yaseen",yy)]:
        u,c = np.unique(yy_.astype(int), return_counts=True)
        print(f"    {nm} sinif dagilimi: {dict(zip(u.tolist(), c.tolist()))}")

    # birlestir
    X = np.concatenate([Xc, Xi, Xy])
    dataset_lbl = np.array([0]*len(Xc) + [1]*len(Xi) + [2]*len(Xy))
    class_lbl = np.concatenate([yc, yi, yy]).astype(int)

    # standardize + t-SNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    Xs = StandardScaler().fit_transform(X)
    print("t-SNE calisiyor (biraz surebilir)...")
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42,
               max_iter=1000).fit_transform(Xs)

    # ciz
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # Panel A: dataset
    dcolors = ["#999999", "#4878CF", "#6ACC65"]; dnames = ["CirCor (internal)","CinC (heterogeneous)","Yaseen (homogeneous)"]
    for d in range(3):
        m = dataset_lbl == d
        axes[0].scatter(emb[m,0], emb[m,1], s=6, alpha=0.5, c=dcolors[d], label=dnames[d])
    axes[0].set_title("(A) Feature space by dataset"); axes[0].legend(fontsize=8, markerscale=2)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    # Panel B: sinif
    ccolors = ["#4878CF", "#D65F5F"]; cnames = ["Normal","Abnormal"]
    for c in range(2):
        m = class_lbl == c
        axes[1].scatter(emb[m,0], emb[m,1], s=6, alpha=0.5, c=ccolors[c], label=cnames[c])
    axes[1].set_title("(B) Feature space by class"); axes[1].legend(fontsize=8, markerscale=2)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.suptitle("Learned feature space (PCG-MambaConformer, pre-fusion): domain alignment of the homogeneous target", fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.95])
    for ext in ["png","pdf"]:
        fig.savefig(FIG/f"F16_tsne_domains.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F16_tsne_domains.png + .pdf -> {FIG}")

    # ek metrik: her dataset'te SINIF AYRISABILIRLIGI (transferi bu belirliyor)
    # merkez mesafesi yaniltici; asil onemli olan hedefin normal/abnormal ayrilabilirligi
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    print("\n=== Sinif ayrisabilirligi (orijinal 512-B ozellik uzayinda) ===")
    for d, name in [(0,"CirCor"),(1,"CinC"),(2,"Yaseen")]:
        m = dataset_lbl == d
        Xd, yd = Xs[m], class_lbl[m]
        if len(np.unique(yd)) < 2:
            print(f"  {name}: tek sinif, atlandi"); continue
        try:
            sil = silhouette_score(Xd, yd)
        except Exception:
            sil = float("nan")
        # lineer ayrilabilirlik: 5-fold LR dogrulugu
        try:
            lin = cross_val_score(LogisticRegression(max_iter=1000, class_weight="balanced"),
                                  Xd, yd, cv=5, scoring="roc_auc").mean()
        except Exception:
            lin = float("nan")
        print(f"  {name:8s}: silhouette(normal/abnormal)={sil:.3f}, linear-separability AUROC={lin:.3f}")
    print("  (Yuksek deger = sinrflar daha ayrilabilir = daha iyi transfer beklenir)")
    # merkez mesafesi de raporlanir ama ikincil
    cen_c = emb[dataset_lbl==0].mean(0); cen_i = emb[dataset_lbl==1].mean(0); cen_y = emb[dataset_lbl==2].mean(0)
    print(f"\n  (ikincil) t-SNE merkez mesafeleri: CirCor-Yaseen={np.linalg.norm(cen_c-cen_y):.1f}, "
          f"CirCor-CinC={np.linalg.norm(cen_c-cen_i):.1f}")
    print("  NOT: merkez mesafesi transferi ACIKLAMAZ; sinif ayrisabilirligi aciklar.")


if __name__ == "__main__":
    main()
