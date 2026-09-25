#!/usr/bin/env python3
"""
37_mambaconformer_revised.py
PCG-MambaConformer: multi-representation PCG classifier for CirCor-style data.

Key revisions vs original:
  1) Feature/intermediate fusion is kept as the main architecture.
  2) Conformer positional encoding is dynamic sinusoidal encoding, not fixed to T=201.
  3) Mamba branch uses official mamba_ssm.Mamba when available; otherwise falls back to
     a clearly named MambaInspiredSSMBlock.
  4) CWT image branch uses timm ConvNeXt when available; otherwise uses a lightweight
     ConvNeXt-like CNN fallback, so the file runs without timm.
  5) Inception branch includes residual projection.
  6) Optional auscultation-site embedding can be added at fusion level.
  7) Includes patient-level aggregation utility for CirCor evaluation.

Expected segment-level inputs:
  raw: (B, L) or (B, 1, L), e.g. L=10000 for 10 s at 1000 Hz
  mel: (B, M, T) or (B, 1, M, T), e.g. (B,64,201)
  cwt: (B, F, T) or (B, 1, F, T), e.g. (B,64,201)
  site_id optional: LongTensor (B,), e.g. AV/PV/TV/MV encoded as 0..3

Output:
  logits: (B, n_classes)
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_raw_3d(raw: torch.Tensor) -> torch.Tensor:
    """Return raw PCG as (B, 1, L)."""
    if raw.dim() == 2:
        return raw.unsqueeze(1)
    if raw.dim() == 3:
        return raw
    return raw.view(raw.size(0), 1, -1)


def _ensure_img_4d(x: torch.Tensor) -> torch.Tensor:
    """Return time-frequency representation as (B, 1, F, T)."""
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape={tuple(x.shape)}")


def sinusoidal_positional_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create sinusoidal positional encoding with shape (1, length, dim)."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


# -----------------------------------------------------------------------------
# Branch 1: Raw waveform -> InceptionTime-inspired residual branch
# -----------------------------------------------------------------------------
class InceptionModule1D(nn.Module):
    def __init__(self, in_ch: int, n_filters: int = 32, kernels: Sequence[int] = (9, 19, 39), bottleneck: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        conv_in = bottleneck if in_ch > 1 else in_ch
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, kernel_size=k, padding=k // 2, bias=False) for k in kernels
        ])
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, kernel_size=1, bias=False),
        )
        out_ch = n_filters * (len(kernels) + 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.res_bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        y = torch.cat([conv(z) for conv in self.convs] + [self.pool_conv(x)], dim=1)
        y = self.bn(y)
        r = self.res_bn(self.residual(x))
        return self.act(y + r)


class InceptionTimeBranch(nn.Module):
    """Raw 1D PCG -> feature vector."""
    def __init__(self, in_ch: int = 1, n_filters: int = 32, depth: int = 3, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        ch = 16
        blocks = []
        for _ in range(depth):
            blocks.append(InceptionModule1D(ch, n_filters=n_filters))
            ch = n_filters * 4
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.LayerNorm(ch), nn.Linear(ch, out_dim))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = _ensure_raw_3d(raw)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.proj(x)


# -----------------------------------------------------------------------------
# Branch 2: Log-Mel -> Conformer branch
# -----------------------------------------------------------------------------
class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, conv_kernel: int = 15, ff_mult: int = 2, drop: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(drop),
        )
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop, batch_first=True)
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(drop),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_mult),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(drop),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        q = self.ln_attn(x)
        x = x + self.attn(q, q, q, need_weights=False)[0]
        c = self.ln_conv(x).transpose(1, 2)
        x = x + self.conv(c).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class ConformerBranch(nn.Module):
    """Log-Mel spectrogram (B,M,T) or (B,1,M,T) -> feature vector."""
    def __init__(self, n_mels: int = 64, d_model: int = 128, n_blocks: int = 2, out_dim: int = 128, drop: float = 0.1):
        super().__init__()
        self.n_mels = n_mels
        self.proj_in = nn.Linear(n_mels, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, drop=drop) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, out_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 4:
            if mel.size(1) != 1:
                raise ValueError("Mel tensor with 4 dims must have channel size 1: (B,1,M,T)")
            mel = mel.squeeze(1)
        if mel.dim() != 3:
            raise ValueError(f"Expected mel shape (B,M,T) or (B,1,M,T), got {tuple(mel.shape)}")
        if mel.size(1) != self.n_mels:
            raise ValueError(f"Expected {self.n_mels} mel bins, got {mel.size(1)}")
        x = mel.transpose(1, 2)  # (B,T,M)
        x = self.proj_in(x)
        x = x + sinusoidal_positional_encoding(x.size(1), x.size(2), x.device, x.dtype)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.proj_out(x.mean(dim=1))


# -----------------------------------------------------------------------------
# Branch 3: CWT scalogram -> ConvNeXt/timm or local ConvNeXt-like fallback
# -----------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int = 4, drop: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * mlp_ratio, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(channels * mlp_ratio, channels, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.norm(y)
        y = self.pw2(self.act(self.pw1(y)))
        return x + self.drop(y)


class ConvNeXtLiteBranch(nn.Module):
    """Dependency-free ConvNeXt-like CWT branch."""
    def __init__(self, out_dim: int = 128, widths: Sequence[int] = (32, 64, 128), depths: Sequence[int] = (1, 1, 2)):
        super().__init__()
        stages = []
        in_ch = 1
        for width, depth in zip(widths, depths):
            stages += [
                nn.Conv2d(in_ch, width, kernel_size=3, stride=2, padding=1, bias=False),
                LayerNorm2d(width),
                nn.GELU(),
            ]
            stages += [ConvNeXtLiteBlock(width) for _ in range(depth)]
            in_ch = width
        self.net = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Flatten(), nn.LayerNorm(widths[-1]), nn.Linear(widths[-1], out_dim))

    def forward(self, cwt: torch.Tensor) -> torch.Tensor:
        x = _ensure_img_4d(cwt)
        x = self.net(x)
        x = self.pool(x)
        return self.proj(x)


class CWTImageBranch(nn.Module):
    """CWT scalogram branch. Uses timm ConvNeXt if installed, else ConvNeXtLite."""
    def __init__(self, out_dim: int = 128, pretrained: bool = True, backbone: str = "convnext_tiny"):
        super().__init__()
        self.backend = "convnext_lite"
        try:
            import timm  # type: ignore
            self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=1)
            feat = self.net.num_features
            self.proj = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, out_dim))
            self.backend = f"timm:{backbone}"
        except Exception:
            self.net = ConvNeXtLiteBranch(out_dim=out_dim)
            self.proj = None

    def forward(self, cwt: torch.Tensor) -> torch.Tensor:
        x = _ensure_img_4d(cwt)
        f = self.net(x)
        if self.proj is not None:
            f = self.proj(f)
        return f


# -----------------------------------------------------------------------------
# Branch 4: Raw waveform -> Mamba or Mamba-inspired fallback
# -----------------------------------------------------------------------------
class MambaInspiredSSMBlock(nn.Module):
    """Fallback selective-SSM-inspired block.

    This is not the official Mamba implementation. Use it only when mamba_ssm is
    unavailable, and describe it in manuscripts as a 'Mamba-inspired SSM block'.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, dim=-1)
        xi = self.conv1d(xi.transpose(1, 2))[..., :L].transpose(1, 2)
        xi = F.silu(xi)

        x_dbl = self.x_proj(xi)
        dt, Bm, Cm = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dB = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)
            h = dA * h + dB * xi[:, t].unsqueeze(-1)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)
        y = y + xi * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBranch(nn.Module):
    """Raw PCG -> CNN patch stem -> official Mamba if installed, else fallback SSM -> feature vector."""
    def __init__(
        self,
        d_model: int = 128,
        n_blocks: int = 2,
        out_dim: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        prefer_official_mamba: bool = True,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=8, padding=7), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3), nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2), nn.GELU(),
        )
        blocks = []
        self.backend = "mamba_inspired_fallback"
        MambaClass = None
        if prefer_official_mamba:
            try:
                from mamba_ssm import Mamba as MambaClass  # type: ignore
                self.backend = "official_mamba_ssm"
            except Exception:
                MambaClass = None
        for _ in range(n_blocks):
            if MambaClass is not None:
                blocks.append(MambaClass(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            else:
                blocks.append(MambaInspiredSSMBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = _ensure_raw_3d(raw)
        x = self.stem(x).transpose(1, 2)
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        x = self.out_norm(x)
        return self.proj(x.mean(dim=1))


# -----------------------------------------------------------------------------
# Full model: feature-level fusion
# -----------------------------------------------------------------------------
class PCGMambaConformer(nn.Module):
    def __init__(
        self,
        n_classes: int = 2,
        branch_dim: int = 128,
        pretrained: bool = True,
        use_branches: Sequence[str] = ("incep", "conf", "cwt", "mamba"),
        n_sites: int = 0,
        site_emb_dim: int = 16,
        dropout: float = 0.3,
        prefer_official_mamba: bool = True,
    ):
        super().__init__()
        valid = {"incep", "conf", "cwt", "mamba"}
        unknown = set(use_branches) - valid
        if unknown:
            raise ValueError(f"Unknown branch names: {sorted(unknown)}. Valid={sorted(valid)}")
        self.use_branches = tuple(use_branches)
        self.n_sites = n_sites

        if "incep" in self.use_branches:
            self.incep = InceptionTimeBranch(out_dim=branch_dim)
        if "conf" in self.use_branches:
            self.conf = ConformerBranch(out_dim=branch_dim)
        if "cwt" in self.use_branches:
            self.cwt = CWTImageBranch(out_dim=branch_dim, pretrained=pretrained)
        if "mamba" in self.use_branches:
            self.mamba = MambaBranch(out_dim=branch_dim, prefer_official_mamba=prefer_official_mamba)

        fused_dim = branch_dim * len(self.use_branches)
        if n_sites > 0:
            self.site_emb = nn.Embedding(n_sites, site_emb_dim)
            fused_dim += site_emb_dim
        else:
            self.site_emb = None

        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def extract_features(
        self,
        raw: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
        cwt: Optional[torch.Tensor] = None,
        site_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        if "incep" in self.use_branches:
            feats.append(self.incep(raw))
        if "conf" in self.use_branches:
            if mel is None:
                raise ValueError("mel input is required when 'conf' branch is enabled")
            feats.append(self.conf(mel))
        if "cwt" in self.use_branches:
            if cwt is None:
                raise ValueError("cwt input is required when 'cwt' branch is enabled")
            feats.append(self.cwt(cwt))
        if "mamba" in self.use_branches:
            feats.append(self.mamba(raw))
        fused = torch.cat(feats, dim=-1)
        if self.site_emb is not None:
            if site_id is None:
                raise ValueError("site_id is required because n_sites > 0")
            fused = torch.cat([fused, self.site_emb(site_id.long())], dim=-1)
        return fused

    def forward(
        self,
        raw: torch.Tensor,
        mel: Optional[torch.Tensor] = None,
        cwt: Optional[torch.Tensor] = None,
        site_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.fusion(self.extract_features(raw, mel, cwt, site_id))

    def branch_backends(self) -> Dict[str, str]:
        out = {}
        if hasattr(self, "cwt"):
            out["cwt"] = getattr(self.cwt, "backend", "unknown")
        if hasattr(self, "mamba"):
            out["mamba"] = getattr(self.mamba, "backend", "unknown")
        return out


# -----------------------------------------------------------------------------
# Patient-level aggregation for CirCor-style evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def aggregate_patient_probabilities(
    logits: torch.Tensor,
    patient_ids: Sequence[Union[str, int]],
    method: str = "mean_prob",
) -> Dict[Union[str, int], torch.Tensor]:
    """Aggregate segment/recording-level logits into patient-level probabilities.

    Args:
        logits: Tensor with shape (N, C).
        patient_ids: length-N patient identifiers.
        method: 'mean_prob', 'max_prob', or 'mean_logit'.

    Returns:
        dict patient_id -> probability tensor with shape (C,).
    """
    if logits.size(0) != len(patient_ids):
        raise ValueError("logits and patient_ids length mismatch")
    groups: Dict[Union[str, int], List[int]] = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        groups[pid].append(i)

    probs = torch.softmax(logits, dim=-1)
    out: Dict[Union[str, int], torch.Tensor] = {}
    for pid, idxs in groups.items():
        idx = torch.tensor(idxs, device=logits.device)
        if method == "mean_prob":
            out[pid] = probs.index_select(0, idx).mean(dim=0)
        elif method == "max_prob":
            out[pid] = probs.index_select(0, idx).max(dim=0).values
        elif method == "mean_logit":
            out[pid] = torch.softmax(logits.index_select(0, idx).mean(dim=0), dim=-1)
        else:
            raise ValueError("method must be one of: mean_prob, max_prob, mean_logit")
    return out


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    print("Testing revised PCG-MambaConformer...")

    # Small tensors keep the CPU smoke test fast. Use real lengths during training.
    raw = torch.randn(1, 1, 512)
    mel = torch.randn(1, 1, 64, 32)
    cwt = torch.randn(1, 1, 64, 32)
    site = torch.tensor([0])

    model = PCGMambaConformer(n_classes=2, branch_dim=32, pretrained=False, n_sites=4)
    model.eval()
    with torch.no_grad():
        logits = model(raw, mel, cwt, site_id=site)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Output: {tuple(logits.shape)} | params={n_params:.2f}M | backends={model.branch_backends()}")

    patient_probs = aggregate_patient_probabilities(logits, patient_ids=["p1"], method="mean_prob")
    print(f"Patient aggregation keys={list(patient_probs.keys())}, prob_shape={tuple(patient_probs['p1'].shape)}")
    print("[OK] revised model runs")
