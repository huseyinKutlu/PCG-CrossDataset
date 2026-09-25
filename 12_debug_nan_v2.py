#!/usr/bin/env python3
"""
12_debug_nan_v2.py
nan'i GERCEK egitim kosullarinda (sampler + optimizer adimlari) yakalar.
10 adim degil — nan cikana kadar kosar, tam o anda patlayan yeri raporlar.

Forward hook'lar ile her alt-modulun ciktisi izlenir; nan ilk nerede dogarsa
oraya isaret eder. Ayrica Mamba ic parametrelerini (A_log, dt) izler.

Kullanim:
  python 12_debug_nan_v2.py --fold 0 --lr 5e-4 --max_steps 400
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ds_mod = import_module("05_dataset")
hy_mod = import_module("09_model_hybrid")

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_CLASSES = 3


def make_sampler(ds, beta=0.5):
    c = np.bincount(ds.seg_labels, minlength=N_CLASSES).astype(float)
    cw = (c.sum() / (N_CLASSES * np.maximum(c, 1))) ** beta
    return WeightedRandomSampler(torch.DoubleTensor(cw[ds.seg_labels]),
                                 num_samples=len(ds), replacement=True)


def param_health(model, tag):
    """Tum parametrelerin |max| degerini tara, en buyukleri dondur."""
    worst = []
    for n, p in model.named_parameters():
        m = p.detach().abs().max().item()
        finite = torch.isfinite(p).all().item()
        worst.append((m, n, finite))
    worst.sort(reverse=True)
    print(f"\n   [{tag}] en buyuk |parametre| degerleri:")
    for m, n, f in worst[:6]:
        flag = "" if f else "  <-- NAN/INF!"
        print(f"      {m:.3e}  {n}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--head", default="mlp")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"
    print("="*70)
    print(f"nan TESHIS v2 — gercek egitim kosullari (lr={args.lr}, beta={args.beta})")
    print("="*70)

    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="train")
    sampler = make_sampler(tr, args.beta)
    loader = DataLoader(tr, batch_size=64, sampler=sampler, num_workers=4,
                        pin_memory=True, drop_last=True)

    torch.manual_seed(42)
    model = hy_mod.HybridPCG(head=args.head, d_model=128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    # forward hook: ilk nan ureten alt-modulu yakala
    nan_module = {"name": None}
    def hook(mod, inp, out):
        if nan_module["name"] is not None:
            return
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t) and not torch.isfinite(t).all():
            # bu modulun adini bul
            for n, m in model.named_modules():
                if m is mod:
                    nan_module["name"] = n; break
    for m in model.modules():
        m.register_forward_hook(hook)

    # Mamba ic parametrelerini izle (A_log, dt_bias vb.)
    mamba_internal = [n for n, _ in model.named_parameters()
                      if "A_log" in n or "dt" in n or "D" == n.split(".")[-1]]
    print(f"\nIzlenen Mamba ic parametreleri: {len(mamba_internal)} adet")
    for n in mamba_internal[:4]:
        print(f"   {n}")

    print(f"\nEgitim basliyor (max {args.max_steps} adim, nan'da duracak)...")
    print("adim | loss | grad_norm | en_buyuk_param")
    print("-"*60)

    model.train()
    step = 0
    it = iter(loader)
    while step < args.max_steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        raw = b["raw"].to(DEVICE); mel = b["mel"].to(DEVICE)
        y = b["label"].to(DEVICE)

        opt.zero_grad()
        out = model(raw, mel)

        if nan_module["name"] is not None:
            print(f"\n>>> adim {step}: ILK nan ureten modul: '{nan_module['name']}'")
            param_health(model, f"adim {step} (forward nan)")
            break

        loss = crit(out, y)
        if not torch.isfinite(loss):
            print(f"\n>>> adim {step}: loss nan (ama forward ciktisi sonluydu — "
                  f"muhtemelen label_smoothing/log)")
            param_health(model, f"adim {step}")
            break

        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        opt.step()

        # max parametre
        pmax = max(p.detach().abs().max().item() for p in model.parameters())

        if step % 20 == 0 or gnorm > 100:
            mark = "  <-- grad buyuk!" if gnorm > 100 else ""
            print(f"{step:4d} | {loss.item():.4f} | {gnorm:8.2e} | {pmax:.3e}{mark}")

        # erken uyari: parametre patlamaya basladi mi
        if pmax > 1e4:
            print(f"\n>>> adim {step}: parametre patlamasi basladi (|max|={pmax:.3e})")
            param_health(model, f"adim {step} (param patlamasi)")
            # bir adim daha at, nan'a donusecek mi gor
            opt.zero_grad()
            out2 = model(raw, mel)
            print(f"   sonraki forward sonlu mu: {torch.isfinite(out2).all().item()}")
            break

        step += 1

    if nan_module["name"] is None and step >= args.max_steps:
        print(f"\n[OK] {args.max_steps} adim boyunca nan YOK (lr={args.lr}).")
        print("   -> Bu lr/beta ile stabil. Egitimde sorun baska parametrede olabilir.")

    print("\n" + "="*70)
    print("Teshis v2 bitti.")


if __name__ == "__main__":
    main()
