#!/usr/bin/env python3
"""
39_resnet1d.py
ResNet1D: ham PCG sinyali icin derin residual 1D-CNN baseline.
"Guclu standart 1D baseline" — derin oğrenme kiyasi icin.

Mimari: Conv stem + 4 residual stage (her stage 2 blok) + GAP + FC.
Ham 1D sinyal girdisi (B, 10000) veya (B, 1, 10000).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.act(out + identity)


class ResNet1D(nn.Module):
    """Ham 1D PCG -> derin residual CNN -> sinif."""
    def __init__(self, n_classes=2, base_ch=32, layers=(2, 2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        in_ch = base_ch
        stages = []
        for i, (out_ch, n_blk) in enumerate(zip(chs, layers)):
            stride = 1 if i == 0 else 2
            blocks = [BasicBlock1D(in_ch, out_ch, stride=stride)]
            for _ in range(n_blk - 1):
                blocks.append(BasicBlock1D(out_ch, out_ch))
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_ch, n_classes))

    def forward(self, raw):
        if raw.dim() == 2:
            x = raw.unsqueeze(1)
        elif raw.dim() == 3:
            x = raw
        else:
            x = raw.view(raw.size(0), 1, -1)
        x = self.stem(x)
        x = self.stages(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


if __name__ == "__main__":
    m = ResNet1D(n_classes=2)
    for shape in [(2, 10000), (2, 1, 10000)]:
        x = torch.randn(*shape)
        y = m(x)
        print(f"  girdi {shape} -> cikti {tuple(y.shape)}")
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  {n:.1f}M parametre")
    print("[OK] ResNet1D calisiyor")
