#!/usr/bin/env python3
"""
35_timm_models.py
Guclu modern mimariler (timm): EfficientNet-B0, MaxViT-tiny.
Mel-spektrogramini (64x201) goruntu olarak isler, ImageNet-pretrained omurga.

Amac: "mimari karmasikligi cross-dataset genellemeyi cozmuyor" tezini
EN GUCLU modern mimarilerle test etmek. Eger bunlar da ic'te iyi ama
dis'te coker ise, tez demir gibi olur (hakem itirazini kapatir).

Kullanim (model olarak import edilir):
  from importlib import import_module
  tm = import_module("35_timm_models")
  model = tm.TimmPCG("efficientnet_b0", n_classes=3)  # mel girdi
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TimmPCG(nn.Module):
    """Mel-spektrogram (B,1,64,201) -> timm omurga -> sinif.
    Gerekirse girdiyi modele uygun boyuta interpolate eder."""
    def __init__(self, backbone="efficientnet_b0", n_classes=3,
                 pretrained=True, img_size=224):
        super().__init__()
        self.backbone_name = backbone
        # MaxViT gibi katı modeller sabit boyut ister -> resize gerekli
        self.needs_resize = "maxvit" in backbone or "tf_" in backbone
        self.img_size = img_size
        self.net = timm.create_model(
            backbone, pretrained=pretrained, num_classes=n_classes, in_chans=1)

    def forward(self, mel):
        # mel: (B, 64, 201) veya (B, 1, 64, 201)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)          # (B,1,64,201)
        if self.needs_resize:
            mel = F.interpolate(mel, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
        return self.net(mel)


def build(backbone, n_classes=3, pretrained=True):
    """Fabrika: model + mod ('mel') dondur (16'nin build_binary_model uyumu)."""
    return TimmPCG(backbone, n_classes=n_classes, pretrained=pretrained), "mel"


# hizli kendi-testi (torch varsa)
if __name__ == "__main__":
    print("timm modelleri test ediliyor...")
    for bb in ["efficientnet_b0", "maxvit_tiny_tf_224"]:
        m = TimmPCG(bb, n_classes=2, pretrained=False)
        x = torch.randn(2, 64, 201)   # 2 ornek mel
        y = m(x)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {bb}: cikti {tuple(y.shape)} | {n_params:.1f}M parametre")
    print("[OK] timm modelleri calisiyor")
