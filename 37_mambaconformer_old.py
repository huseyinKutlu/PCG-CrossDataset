#!/usr/bin/env python3
"""
37_mambaconformer.py
PCG-MambaConformer: 4-dalli cok-temsilli model.

Mimari (Hocam'in tarifi, feature-level fusion):
  Dal 1: Ham 1D -> InceptionTime      (lokal kalp sesi morfolojisi)
  Dal 2: Log-Mel -> Conformer         (zaman-frekans murmur paterni)
  Dal 3: CWT scalogram -> ConvNeXt    (cok-olcekli gecici patern)
  Dal 4: Ham sekans -> Mamba          (uzun-donem temporal bagimlilik)
  Fusion: [4 ozellik concat] -> LayerNorm -> Dropout -> MLP -> softmax

Girdi: raw (B,10000), mel (B,64,201), cwt (B,64,201)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ DAL 1: InceptionTime (ham 1D) ============
class InceptionModule1D(nn.Module):
    def __init__(self, in_ch, n_filters=32, kernels=(9, 19, 39)):
        super().__init__()
        bottleneck = 32
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, padding=0, bias=False) \
            if in_ch > 1 else nn.Identity()
        bn_out = bottleneck if in_ch > 1 else in_ch
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_out, n_filters, k, padding=k // 2, bias=False) for k in kernels])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, n_filters, 1, padding=0, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernels) + 1))
        self.act = nn.ReLU()

    def forward(self, x):
        b = self.bottleneck(x)
        outs = [c(b) for c in self.convs]
        outs.append(self.conv_pool(self.maxpool(x)))
        return self.act(self.bn(torch.cat(outs, dim=1)))


class InceptionTime(nn.Module):
    """Ham 1D PCG -> ozellik vektoru."""
    def __init__(self, in_ch=1, n_filters=32, depth=3, out_dim=128):
        super().__init__()
        # once downsample (10000 -> ~625), aksi halde cok agir
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 16, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 16, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(16), nn.ReLU())
        chs = 16
        blocks = []
        for _ in range(depth):
            blocks.append(InceptionModule1D(chs, n_filters))
            chs = n_filters * 4
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(chs, out_dim)

    def forward(self, raw):
        # raw: (B,10000) veya (B,1,10000)
        if raw.dim() == 2:
            x = raw.unsqueeze(1)            # (B,1,10000)
        elif raw.dim() == 3:
            x = raw                         # (B,1,10000) zaten
        else:
            x = raw.view(raw.size(0), 1, -1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return self.proj(x)  # (B, out_dim)


# ============ DAL 2: Conformer (mel) ============
class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, conv_kernel=15, ff_mult=2, drop=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(), nn.Dropout(drop), nn.Linear(d_model * ff_mult, d_model))
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop, batch_first=True)
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model), nn.SiLU(),
            nn.Conv1d(d_model, d_model, 1), nn.Dropout(drop))
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(), nn.Dropout(drop), nn.Linear(d_model * ff_mult, d_model))
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x):  # x: (B, T, d)
        x = x + 0.5 * self.ff1(x)
        a = self.ln_attn(x)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        c = self.ln_conv(x).transpose(1, 2)
        x = x + self.conv(c).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class ConformerBranch(nn.Module):
    """Mel (B,64,201) -> Conformer -> ozellik."""
    def __init__(self, n_mels=64, d_model=128, n_blocks=2, out_dim=128):
        super().__init__()
        # mel'i zaman dizisi olarak: her zaman adimi 64-boyutlu
        self.proj_in = nn.Linear(n_mels, d_model)
        self.pos = nn.Parameter(torch.randn(1, 201, d_model) * 0.02)
        self.blocks = nn.ModuleList([ConformerBlock(d_model) for _ in range(n_blocks)])
        self.gap_proj = nn.Linear(d_model, out_dim)

    def forward(self, mel):  # (B,64,201) veya (B,1,64,201)
        if mel.dim() == 4:
            mel = mel.squeeze(1)           # (B,64,201)
        x = mel.transpose(1, 2)            # (B, 201, 64)
        x = self.proj_in(x) + self.pos[:, :x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.gap_proj(x.mean(1))    # (B, out_dim)


# ============ DAL 3: ConvNeXt (CWT scalogram) ============
class ConvNeXtBranch(nn.Module):
    """CWT scalogram (B,64,201) -> timm ConvNeXt-tiny -> ozellik."""
    def __init__(self, out_dim=128, pretrained=True):
        super().__init__()
        import timm
        self.net = timm.create_model("convnext_tiny", pretrained=pretrained,
                                     num_classes=0, in_chans=1)  # num_classes=0 -> ozellik
        feat = self.net.num_features
        self.proj = nn.Linear(feat, out_dim)

    def forward(self, cwt):  # (B,64,201) veya (B,1,64,201)
        if cwt.dim() == 3:
            x = cwt.unsqueeze(1)   # (B,1,64,201)
        else:
            x = cwt                # (B,1,64,201) zaten
        f = self.net(x)
        return self.proj(f)  # (B, out_dim)


# ============ DAL 4: Mamba (ham sekans) ============
class MambaBlock(nn.Module):
    """Pure-PyTorch basitlestirilmis SSM/Mamba blogu (mamba_ssm gereksiz)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32)
                              .log().unsqueeze(0).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.d_state = d_state

    def forward(self, x):  # (B, L, d_model)
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)        # her biri (B,L,d_inner)
        xi = xi.transpose(1, 2)
        xi = self.conv1d(xi)[..., :L].transpose(1, 2)
        xi = F.silu(xi)
        # SSM parametreleri
        x_dbl = self.x_proj(xi)            # (B,L,2*d_state+1)
        dt, Bm, Cm = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B,L,d_inner)
        A = -torch.exp(self.A)             # (d_inner,d_state)
        # basit ardisik tarama (egitim icin yeterli, hiz ikincil)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)          # (B,d_inner,d_state)
            dB = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)  # (B,d_inner,d_state)
            h = dA * h + dB * xi[:, t].unsqueeze(-1)
            y = (h * Cm[:, t].unsqueeze(1)).sum(-1)              # (B,d_inner)
            ys.append(y)
        y = torch.stack(ys, dim=1)         # (B,L,d_inner)
        y = y + xi * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBranch(nn.Module):
    """Ham -> CNN stem (downsample) -> Mamba -> ozellik."""
    def __init__(self, d_model=128, n_blocks=2, out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=8, padding=7), nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(64, d_model, 5, stride=2, padding=2), nn.GELU())
        # 10000 -> /8 /4 /2 = ~156 zaman adimi
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, raw):  # (B,10000) veya (B,1,10000)
        if raw.dim() == 2:
            x = raw.unsqueeze(1)
        elif raw.dim() == 3:
            x = raw
        else:
            x = raw.view(raw.size(0), 1, -1)
        x = self.stem(x).transpose(1, 2)   # (B, L, d_model)
        for blk in self.blocks:
            x = x + blk(self.norm(x))
        return self.proj(x.mean(1))         # (B, out_dim)


# ============ TAM MODEL: feature-level fusion ============
class PCGMambaConformer(nn.Module):
    def __init__(self, n_classes=2, branch_dim=128, pretrained=True,
                 use_branches=("incep", "conf", "convnext", "mamba")):
        super().__init__()
        self.use = use_branches
        if "incep" in use_branches:
            self.incep = InceptionTime(out_dim=branch_dim)
        if "conf" in use_branches:
            self.conf = ConformerBranch(out_dim=branch_dim)
        if "convnext" in use_branches:
            self.convnext = ConvNeXtBranch(out_dim=branch_dim, pretrained=pretrained)
        if "mamba" in use_branches:
            self.mamba = MambaBranch(out_dim=branch_dim)
        fused = branch_dim * len(use_branches)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused), nn.Dropout(0.3),
            nn.Linear(fused, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes))

    def forward(self, raw, mel, cwt):
        feats = []
        if "incep" in self.use:    feats.append(self.incep(raw))
        if "conf" in self.use:     feats.append(self.conf(mel))
        if "convnext" in self.use: feats.append(self.convnext(cwt))
        if "mamba" in self.use:    feats.append(self.mamba(raw))
        return self.fusion(torch.cat(feats, dim=-1))


# hizli kendi-testi
if __name__ == "__main__":
    print("PCG-MambaConformer test ediliyor...")
    m = PCGMambaConformer(n_classes=2, pretrained=False)
    # gercek dataset (B,1,...) boyutlariyla test (kanal boyutu var)
    raw = torch.randn(2, 1, 10000)
    mel = torch.randn(2, 1, 64, 201)
    cwt = torch.randn(2, 1, 64, 201)
    y = m(raw, mel, cwt)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  (B,1,...) girdi -> cikti: {tuple(y.shape)} | {n_params:.1f}M parametre")
    # ayrica (B,...) boyutlariyla da (kanal yok)
    raw2 = torch.randn(2, 10000); mel2 = torch.randn(2, 64, 201); cwt2 = torch.randn(2, 64, 201)
    y2 = m(raw2, mel2, cwt2)
    print(f"  (B,...) girdi   -> cikti: {tuple(y2.shape)} OK")
    for br in ["incep", "conf", "convnext", "mamba"]:
        mb = PCGMambaConformer(n_classes=2, pretrained=False, use_branches=(br,))
        yb = mb(raw, mel, cwt)
        print(f"  sadece {br}: {tuple(yb.shape)} OK")
    print("[OK] model calisiyor")
