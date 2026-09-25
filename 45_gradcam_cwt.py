#!/usr/bin/env python3
"""
45_gradcam_cwt.py  (REVISED)
XAI-B: Grad-CAM CWT branch saliency.

Duzeltmeler:
  - Normal VE abnormal ornekleri GARANTILI topla (sinif-bazli, erken break yok)
  - Daha KESKIN harita: son 7x7 depthwise conv yerine daha erken/yuksek-cozunurluklu
    bir conv katmani hedeflenir (--layer ile secilebilir; varsayilan: mid)
  - Bulunan ornek sayisi kadar subplot (BOS PANEL YOK)

Cikti: figures/F17_gradcam_cwt.png (+pdf), 300 dpi

Kullanim:
  python 45_gradcam_cwt.py                 # 2 normal + 2 abnormal, orta conv
  python 45_gradcam_cwt.py --layer last    # son conv (eski davranis)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ev = import_module("16_external_val")
mc_mod = import_module("37_mambaconformer")
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIG = PROJECT_ROOT / "figures"; FIG.mkdir(exist_ok=True)
CKPT = PROJECT_ROOT / "checkpoints" / "binary"


def list_convs(module):
    return [(n, m) for n, m in module.named_modules() if isinstance(m, torch.nn.Conv2d)]


def pick_target_conv(cwt_branch, which="mid"):
    convs = list_convs(cwt_branch)
    if not convs:
        return None, None
    if which == "last":
        return convs[-1]
    if which == "first":
        return convs[0]
    idx = max(0, len(convs)//2 - 1)
    return convs[idx]


def _get_cwt_branch(model):
    for name, mod in model.named_modules():
        if "cwt" in name.lower() and len(list(mod.modules())) > 1:
            return mod
    return getattr(model, "cwt", model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--layer", choices=["last","mid","first"], default="mid")
    ap.add_argument("--scan", type=int, default=4000)
    args = ap.parse_args()

    print("Model yukleniyor...")
    model, _ = ev.build_binary_model("mambaconformer")
    sd = torch.load(CKPT/f"mambaconformer_fold{args.fold}.pth", map_location=DEVICE)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model = model.to(DEVICE).eval()

    cwt_branch = _get_cwt_branch(model)
    tname, target = pick_target_conv(cwt_branch, args.layer)
    print(f"  Grad-CAM hedef conv ({args.layer}): {tname}")
    print(f"    {target}")

    acts, grads = {}, {}
    def fwd_hook(m, i, o): acts["v"] = o.detach()
    def bwd_hook(m, gi, go): grads["v"] = go[0].detach()
    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    yas = ev.YaseenDataset(PROJECT_ROOT/"manifests"/"yaseen_processed.csv", PROJECT_ROOT, norm_adapt="none")
    from torch.utils.data import DataLoader
    g = torch.Generator(); g.manual_seed(123)
    loader = DataLoader(yas, batch_size=1, shuffle=True, num_workers=2, generator=g)

    need = args.n_per_class
    picked = {0: [], 1: []}
    scanned = 0
    print("Dogru siniflanan NORMAL ve ABNORMAL ornekler araniyor...")
    for b in loader:
        scanned += 1
        if scanned > args.scan: break
        if all(len(picked[c]) >= need for c in (0,1)): break
        # Yaseen label zaten ikili (0=normal, 1=abnormal); to_binary_labels KULLANMA
        y = int(b["label"][0].item()) if "label" in b else 0
        if len(picked[y]) >= need:
            continue
        raw = b["raw"].to(DEVICE); mel = b["mel"].to(DEVICE); cwt = b["cwt"].to(DEVICE)
        model.zero_grad()
        logits = model(raw, mel, cwt)
        pred = int(logits.argmax(1)[0].item())
        if pred != y:
            continue
        score = logits[0, pred]
        score.backward()
        A = acts["v"][0]; G = grads["v"][0]
        w = G.mean(dim=(1,2), keepdim=True)
        cam = F.relu((w * A).sum(0))
        cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
        scal = cwt[0,0].cpu().numpy() if cwt.dim()==4 else cwt[0].cpu().numpy()
        prob = float(torch.softmax(logits,1)[0,1].item())
        picked[y].append((scal, cam.cpu().numpy(), prob))
        print(f"    bulundu: sinif={'Normal' if y==0 else 'Abnormal'} p={prob:.2f} "
              f"(N:{len(picked[0])}/{need}, A:{len(picked[1])}/{need})")

    h1.remove(); h2.remove()

    items_ordered = [(0,it) for it in picked[0]] + [(1,it) for it in picked[1]]
    n_found = len(items_ordered)
    if n_found == 0:
        print("[HATA] hic dogru-siniflanan ornek bulunamadi"); return
    print(f"Toplam {n_found} ornek cizilecek (Normal:{len(picked[0])}, Abnormal:{len(picked[1])})")

    rows = n_found
    fig, axes = plt.subplots(rows, 2, figsize=(7, 2.6*rows))
    if rows == 1: axes = axes[None,:]
    for ri,(cls,(scal,cam,p)) in enumerate(items_ordered):
        label = "Normal" if cls==0 else "Abnormal"
        H,W = scal.shape
        cam_t = torch.tensor(cam)[None,None].float()
        cam_rs = F.interpolate(cam_t, size=(H,W), mode="bilinear", align_corners=False)[0,0].numpy()
        axes[ri,0].imshow(scal, aspect="auto", origin="lower", cmap="viridis")
        axes[ri,0].set_title(f"{label} — CWT scalogram", fontsize=9)
        axes[ri,0].set_xticks([]); axes[ri,0].set_yticks([])
        axes[ri,1].imshow(scal, aspect="auto", origin="lower", cmap="gray")
        axes[ri,1].imshow(cam_rs, aspect="auto", origin="lower", cmap="jet", alpha=0.55)
        axes[ri,1].set_title(f"{label} — Grad-CAM (p_abn={p:.2f})", fontsize=9)
        axes[ri,1].set_xticks([]); axes[ri,1].set_yticks([])
    fig.suptitle("Grad-CAM on the CWT branch (PCG-MambaConformer): salient time-frequency regions for correctly classified Yaseen recordings", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.97])
    for ext in ["png","pdf"]:
        fig.savefig(FIG/f"F17_gradcam_cwt.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] F17_gradcam_cwt.png + .pdf -> {FIG}")


if __name__ == "__main__":
    main()
