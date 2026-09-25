#!/usr/bin/env python3
"""
46_test_time_adaptation.py
Test-time adaptation (TTA) — HEDEF ETIKETI HIC KULLANMADAN.

Iki yontem:
  (1) BN-adapt: BatchNorm katmanlarini train moduna alip hedef-batch
      istatistiklerini kullanmak (Schneider 2020). Gradyan/guncelleme YOK.
  (2) TENT: yalnizca norm-katmani affine parametrelerini (gamma/beta),
      hedef tahminlerinin entropisini minimize ederek online gunceller
      (Wang 2021). Etiket kullanilmaz.

Her iki yontem de iki dis dataset'te (CinC heterojen, Yaseen homojen)
naive baseline ile karsilastirilir; cnn_bilstm uzerinde (DANN ile birebir
karsilastirma icin). 5-fold ensemble, record-level AUROC/Sens/Spec.

Cikti:
  figures/F18_tta.png (+pdf)  — Naive vs BN-adapt vs TENT, CinC & Yaseen
  konsola ozet tablo

Kullanim:
  python 46_test_time_adaptation.py
  python 46_test_time_adaptation.py --tta_bs 64 --tent_lr 1e-3 --tent_steps 1
  python 46_test_time_adaptation.py --folds 0,1,2,3,4
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev = import_module("16_external_val")
me_mod = import_module("06_metrics")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = ev.DEVICE
FIG = PROJECT_ROOT / "figures"; FIG.mkdir(exist_ok=True)

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
NORM_TYPES = BN_TYPES + (nn.GroupNorm, nn.LayerNorm,
                         nn.InstanceNorm1d, nn.InstanceNorm2d)


# ---------------------------------------------------------------- yardimcilar
def load_model(name, fold):
    model, mode = ev.build_binary_model(name)
    ckpt = ev.CKPT_DIR / f"{name}_fold{fold}.pth"
    sd = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    return model.to(DEVICE), mode


def fwd(model, b, mode):
    """predict_grouped ile ayni forward mantigi."""
    if mode == "tri":
        return model(b["raw"].to(DEVICE), b["mel"].to(DEVICE), b["cwt"].to(DEVICE))
    if mode == "dual":
        return model(b["raw"].to(DEVICE), b["mel"].to(DEVICE))
    return model(b[mode].to(DEVICE))


def count_norms(model):
    n_bn = sum(isinstance(m, BN_TYPES) for m in model.modules())
    n_other = sum(isinstance(m, NORM_TYPES) and not isinstance(m, BN_TYPES)
                  for m in model.modules())
    return n_bn, n_other


def set_bn_train(model, train=True):
    """Sadece BatchNorm katmanlarini train/eval moduna al (batch istatistigi)."""
    for m in model.modules():
        if isinstance(m, BN_TYPES):
            m.train(train)


def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * logits.log_softmax(1)).sum(1).mean()


def grouped_dict(seg_pa, ids, seg_y):
    """Segment olasiliklarini record bazinda ortala -> {id: (p_abn, y)}."""
    ids = np.asarray(ids); seg_pa = np.asarray(seg_pa); seg_y = np.asarray(seg_y)
    d = {}
    for u in np.unique(ids):
        m = ids == u
        d[u] = (float(seg_pa[m].mean()), int(seg_y[m][0]))
    return d


# ---------------------------------------------------------------- TTA passes
@torch.no_grad()
def pass_bn_adapt(model, loader, mode):
    """BN-adapt: BN train modunda (batch stats), guncelleme yok."""
    model.eval(); set_bn_train(model, True)
    seg_pa, ids, seg_y = [], [], []
    for b in loader:
        if b[next(k for k in ('raw','mel','cwt') if k in b)].size(0) == 1:
            set_bn_train(model, False)            # tek-ornek batch: BN eval
            logits = fwd(model, b, mode)
            set_bn_train(model, True)
        else:
            logits = fwd(model, b, mode)
        seg_pa.extend(logits.softmax(1)[:, 1].cpu().numpy().tolist())
        ids.extend(b["record_id"])
        seg_y.extend(b["label"].numpy().tolist())  # circor=False: dogrudan ikili
    return grouped_dict(seg_pa, ids, seg_y)


def pass_tent(model, loader, mode, lr, steps):
    """TENT: norm affine paramlarini entropy-min ile online gunceller.

    NOT: cnn_bilstm icindeki LSTM, cuDNN nedeniyle backward'i yalnizca TRAIN
    modunda calistirabilir. Bu yuzden modeli train() moduna aliriz ama TUM
    parametrelerin requires_grad'ini kapatip yalnizca norm affine'lerini aciariz
    -> LSTM agirliklari guncellenmez (grad kapali), sadece norm gamma/beta adapte olur.
    BatchNorm yoksa running-stats sorunu da olmaz.
    """
    model.train()                          # LSTM backward icin sart (mod != grad)
    for p in model.parameters():
        p.requires_grad_(False)            # hicbir agirlik guncellenmesin
    params = []
    for m in model.modules():
        if isinstance(m, NORM_TYPES):
            if getattr(m, "weight", None) is not None:
                m.weight.requires_grad_(True); params.append(m.weight)
            if getattr(m, "bias", None) is not None:
                m.bias.requires_grad_(True); params.append(m.bias)
    if not params:
        # adapte edilecek norm parametresi yok -> naive'e dus
        return None
    opt = torch.optim.Adam(params, lr=lr)

    seg_pa, ids, seg_y = [], [], []
    for b in loader:
        # online adaptasyon (tahmin guncel paramlarla, sonra entropy-min adimi)
        logits = fwd(model, b, mode)
        prob = logits.softmax(1)[:, 1].detach().cpu().numpy()
        for _ in range(steps):
            opt.zero_grad()
            loss = softmax_entropy(fwd(model, b, mode) if _ > 0 else logits)
            loss.backward()
            opt.step()
        seg_pa.extend(prob.tolist())
        ids.extend(b["record_id"]); seg_y.extend(b["label"].numpy().tolist())
    return grouped_dict(seg_pa, ids, seg_y)


def ensemble_dicts(dicts):
    """Fold-bazli {id:(p,y)} sozluklerini record bazinda ortala."""
    common = set(dicts[0])
    for d in dicts[1:]:
        common &= set(d)
    ids = sorted(common)
    p = np.array([np.mean([d[i][0] for d in dicts]) for i in ids])
    y = np.array([dicts[0][i][1] for i in ids])
    return p, y


# ---------------------------------------------------------------- main
def run_dataset(name_model, ext, manifest, folds, args):
    ExtDS = ev.YaseenDataset if ext == "yaseen" else ev.CinCDataset
    ds = ExtDS(manifest, PROJECT_ROOT, norm_adapt=args.norm_adapt)
    # naive: shuffle=False, tum segmentler (yayinlanan baseline'i tikraralar)
    naive_loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
    # TTA: sinif-karisik batch'ler icin shuffle (Yaseen sirali!), sabit seed
    g = torch.Generator(); g.manual_seed(2025)
    tta_loader = DataLoader(ds, batch_size=args.tta_bs, shuffle=True,
                            num_workers=4, generator=g, drop_last=True)

    naive_d, bn_d, tent_d = [], [], []
    for fold in folds:
        model, mode = load_model(name_model, fold)
        if fold == folds[0]:
            nbn, noth = count_norms(model)
            print(f"  [{ext}] {name_model}: BatchNorm={nbn}, diger-norm={noth}")
            if nbn == 0:
                print("  UYARI: BatchNorm yok -> BN-adapt naive ile ayni olabilir; "
                      "TENT yalnizca varsa diger norm affine'lerini adapte eder.")
        # NAIVE (eval, running stats) — predict_grouped ile birebir
        rp, ry, ids = ev.predict_grouped(model, naive_loader, mode, "record_id", circor=False)
        naive_d.append({i: (float(p), int(y)) for i, p, y in zip(ids, rp, ry)})
        # BN-adapt
        model_bn, _ = load_model(name_model, fold)
        bn_d.append(pass_bn_adapt(model_bn, tta_loader, mode))
        # TENT
        model_te, _ = load_model(name_model, fold)
        td = pass_tent(model_te, tta_loader, mode, args.tent_lr, args.tent_steps)
        tent_d.append(td if td is not None else bn_d[-1])

    out = {}
    for label, dl in [("Naive", naive_d), ("BN-adapt", bn_d), ("TENT", tent_d)]:
        p, y = ensemble_dicts(dl)
        m = me_mod.binary_metrics(y, p)
        out[label] = (m["auroc"], m["sensitivity"], m["specificity"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn_bilstm")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tta_bs", type=int, default=64)
    ap.add_argument("--tent_lr", type=float, default=1e-3)
    ap.add_argument("--tent_steps", type=int, default=1)
    ap.add_argument("--norm_adapt", default="none")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",")]

    results = {}
    for ext, manifest in [("cinc", PROJECT_ROOT/"manifests"/"cinc2016_processed.csv"),
                          ("yaseen", PROJECT_ROOT/"manifests"/"yaseen_processed.csv")]:
        print(f"\n=== {ext.upper()} ===")
        results[ext] = run_dataset(args.model, ext, manifest, folds, args)

    # ---- ozet tablo
    print("\n" + "="*64)
    print(f"TEST-TIME ADAPTATION ozet ({args.model}, 5-fold ensemble, etiketsiz)")
    print("="*64)
    print(f"{'Dataset':<10}{'Method':<12}{'AUROC':>8}{'Sens':>8}{'Spec':>8}")
    for ext in ("cinc", "yaseen"):
        for meth in ("Naive", "BN-adapt", "TENT"):
            a, se, sp = results[ext][meth]
            print(f"{ext:<10}{meth:<12}{a:>8.3f}{se:>8.3f}{sp:>8.3f}")
    print("="*64)
    print("Yorum: CinC (heterojen) vs Yaseen (homojen) farkina bakin;")
    print("DANN'da oldugu gibi TTA da homojen hedefte yardim edip heterojende")
    print("notr/zararli ise, 'domain-kosullu' tez ucuncu kez dogrulanir.")
    print("NOT: cnn_bilstm'de BatchNorm yok; BN-adapt bu mimaride naive ile")
    print("ozdes olabilir. Asil etkin yontem TENT (norm-affine entropy-min).")

    # ---- figur: Naive/BN/TENT, CinC & Yaseen
    methods = ["Naive", "BN-adapt", "TENT"]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ext, title in [(axes[0], "cinc", "CinC (heterogeneous)"),
                           (axes[1], "yaseen", "Yaseen (homogeneous)")]:
        aurocs = [results[ext][m][0] for m in methods]
        bars = ax.bar(methods, aurocs, color=colors, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, aurocs):
            ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.set_title(title); ax.set_ylabel("External AUROC")
        ax.set_ylim(0.45, 1.0)
    fig.suptitle("Test-time adaptation (no target labels): Naive vs BN-adapt vs TENT", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext_ in ("png", "pdf"):
        fig.savefig(FIG/f"F18_tta.{ext_}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] F18_tta.png + .pdf -> {FIG}")


if __name__ == "__main__":
    main()
