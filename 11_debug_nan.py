#!/usr/bin/env python3
"""
11_debug_nan.py
nan/inf'in hangi katmanda dogdugunu tespit eder. Tek batch, adim adim.

Kullanim:
  python 11_debug_nan.py --fold 0
"""
from __future__ import annotations
import argparse
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn

ds_mod = import_module("05_dataset")
hy_mod = import_module("09_model_hybrid")

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stat(name, t):
    """Bir tensorun saglik durumu."""
    finite = torch.isfinite(t).all().item()
    flag = "OK " if finite else "!! NAN/INF"
    print(f"   [{flag}] {name:24s} shape={tuple(t.shape)} "
          f"min={t.float().min().item():.3e} max={t.float().max().item():.3e} "
          f"mean={t.float().mean().item():.3e}")
    return finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--head", default="mlp")
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    cv_csv = root / "splits" / "circor2022_5fold_cv.csv"

    print("="*70)
    print("nan TESHIS — katman katman")
    print("="*70)

    # 1) VERI saglik kontrolu — once girdide nan/inf var mi?
    print("\n[1] Veri saglik kontrolu (ilk 200 segment):")
    tr = ds_mod.CircorSegmentDataset(cv_csv, project_root=root, fold=args.fold, split="train")
    loader = ds_mod.make_eval_loader(tr, batch_size=64, num_workers=0)
    bad_raw = bad_mel = 0; checked = 0; rmax = mmax = 0.0
    for b in loader:
        raw, mel = b["raw"], b["mel"]
        if not torch.isfinite(raw).all(): bad_raw += 1
        if not torch.isfinite(mel).all(): bad_mel += 1
        rmax = max(rmax, raw.abs().max().item())
        mmax = max(mmax, mel.abs().max().item())
        checked += len(raw)
        if checked >= 200: break
    print(f"   kontrol edilen segment: {checked}")
    print(f"   raw nan/inf batch: {bad_raw} | mel nan/inf batch: {bad_mel}")
    print(f"   raw |max|: {rmax:.3e} | mel |max|: {mmax:.3e}")
    if bad_raw or bad_mel:
        print("   !! VERIDE nan/inf VAR — preprocessing kaynakli. Model degil.")
    else:
        print("   [OK] veride nan/inf yok, girdi temiz.")

    # 2) MODEL ileri gecis — katman katman
    print("\n[2] Model ileri gecis (tek batch, katman katman):")
    torch.manual_seed(0)
    model = hy_mod.HybridPCG(head=args.head, d_model=128).to(DEVICE)
    model.eval()
    b = next(iter(loader))
    raw = b["raw"].to(DEVICE); mel = b["mel"].to(DEVICE)
    stat("girdi raw", raw); stat("girdi mel", mel)

    with torch.no_grad():
        # Mamba kolu adim adim
        print("\n   --- Mamba kolu ---")
        h = model.mamba.stem(raw); stat("mamba.stem", h)
        h = h.transpose(1, 2)
        for i, blk in enumerate(model.mamba.blocks):
            h2 = blk(h)
            ok = stat(f"mamba.block[{i}]", h2)
            h = h2 + h
            if not ok:
                print(f"   >>> nan KAYNAGI: mamba.block[{i}]")
                break
        f1 = model.mamba.norm(h).mean(1) if torch.isfinite(h).all() else h.mean(1)
        stat("mamba cikti (f1)", f1)

        # Swin kolu
        print("\n   --- Swin/Spec kolu ---")
        f2 = model.spec(mel); stat("spec cikti (f2)", f2)

        # Fuzyon + head
        print("\n   --- Fuzyon + head ---")
        if torch.isfinite(f1).all() and torch.isfinite(f2).all():
            z = model.fuse(torch.cat([f1, f2], 1)); stat("fuse", z)
            out = model.head(z); stat("head cikti", out)

    # 3) Geri gecis — gradyan patlamasi var mi
    print("\n[3] Geri gecis (gradyan saglik):")
    model.train()
    out = model(raw, mel)
    if torch.isfinite(out).all():
        loss = nn.functional.cross_entropy(out, b["label"].to(DEVICE))
        loss.backward()
        gnorm = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()
                               if p.grad is not None))
        print(f"   loss={loss.item():.4f} | toplam grad-norm={gnorm.item():.3e}")
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"   !! nan/inf gradyan: {n}")
        if gnorm.item() > 1e3:
            print("   !! GRADYAN PATLAMASI (norm>1000) — clip + dusuk lr sart")
        elif torch.isfinite(gnorm):
            print("   [OK] gradyanlar sonlu")
    else:
        print("   !! ileri geciste zaten nan var, geri gecis anlamsiz")

    # 4) TRAIN MODU ardisik adimlar — BatchNorm running-stats bozulmasi yakalama
    print("\n[4] Train modu ardisik adimlar (10 adim, gercek egitim taklidi):")
    torch.manual_seed(0)
    model2 = hy_mod.HybridPCG(head=args.head, d_model=128).to(DEVICE)
    model2.train()
    opt = torch.optim.AdamW(model2.parameters(), lr=5e-4)
    train_loader = ds_mod.make_eval_loader(tr, batch_size=64, num_workers=0)
    it = iter(train_loader)
    nan_step = -1
    for step in range(10):
        try:
            bb = next(it)
        except StopIteration:
            it = iter(train_loader); bb = next(it)
        r = bb["raw"].to(DEVICE); m_ = bb["mel"].to(DEVICE)
        y = bb["label"].to(DEVICE)
        opt.zero_grad()
        o = model2(r, m_)
        if not torch.isfinite(o).all():
            print(f"   !! adim {step}: model CIKTISI nan (train modu)")
            nan_step = step; break
        loss = nn.functional.cross_entropy(o, y)
        if not torch.isfinite(loss):
            print(f"   !! adim {step}: loss nan")
            nan_step = step; break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt.step()
        # eval moduna gecip ayni batch'i kontrol et (running-stats etkisi)
        model2.eval()
        with torch.no_grad():
            o_eval = model2(r, m_)
        model2.train()
        eflag = "OK" if torch.isfinite(o_eval).all() else "NAN(eval/running-stats!)"
        print(f"   adim {step}: train_loss={loss.item():.4f} | eval-mode cikti: {eflag}")
    if nan_step < 0:
        print("   [OK] 10 train adimi boyunca nan YOK — duzeltme calisiyor.")
    else:
        print(f"   >>> nan adim {nan_step}'de basladi.")

    print("\n" + "="*70)
    print("Teshis bitti. nan kaynagi yukarida '>>> nan KAYNAGI' veya '!!' ile isaretli.")


if __name__ == "__main__":
    main()
