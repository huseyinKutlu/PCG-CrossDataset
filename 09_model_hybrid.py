#!/usr/bin/env python3
"""
09_model_hybrid.py
PCG Q1 Manuscript — Hibrit govde + degistirilebilir head'ler.

MIMARI:
  Govde (sabit):
    - MambaBranch : ham sinyal (1,10000) -> conv on-yuz -> Mamba bloklari -> (B, D)
    - SpecBranch  : log-mel (1,64,201) -> patch embed -> shifted-window
                    transformer enkoder -> (B, D)
    - Fusion      : concat + projeksiyon -> (B, D_f)
  Head (degistirilebilir):
    - "mlp" : sade Linear head (baz cizgisi)
    - "kan" : saf-PyTorch KAN head (ozgunluk adayi, ek bagimlilik yok)
  Ensemble: head degil, egitim stratejisi (ayni govde+mlp farkli seed) ->
            ayri script'te (10) ele alinacak.

NOT (Swin): Gercek Swin-Transformer ImageNet-onegitimli, 224x224 bekler;
64x201 mel'e zorlama olur. Bunun yerine Swin'in kaydirmali-pencere mantigini
mel boyutuna uyarlayan, SIFIRDAN egitilen sade bir enkoder kullaniyoruz.
Makalede durustce "shifted-window transformer encoder" olarak adlandirilir.

Bagimlilik: torch, mamba_ssm (CUDA). KAN ve SpecBranch saf-torch.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False

N_CLASSES = 3


# =============================================================================
# 1) Mamba kolu — ham sinyal
# =============================================================================
class MambaBranch(nn.Module):
    """Ham PCG (B,1,10000) -> conv on-yuz ile dizi kisaltma -> Mamba -> (B,D).

    Conv on-yuz: 10000 ornegi dogrudan Mamba'ya vermek israf; once stride'li
    conv ile ~625 uzunlukta, d_model kanalli bir dizi uretiriz, sonra Mamba.
    """
    def __init__(self, d_model=128, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # 10000 -> /4 /4 /4 = ~156 zaman adimi, d_model kanal
        # NOT: BatchNorm DEGIL GroupNorm — dizi/Mamba modellerinde batch
        # istatistikleri kararsiz (kucuk efektif batch + agresif sampler ->
        # running_var patlamasi -> nan). GroupNorm batch'ten bagimsizdir.
        def cb(ci, co, k=7, s=4, p=3, groups=8):
            return nn.Sequential(nn.Conv1d(ci, co, k, s, p),
                                 nn.GroupNorm(min(groups, co), co), nn.GELU())
        self.stem = nn.Sequential(
            cb(1, 32), cb(32, 64), cb(64, d_model))   # (B, d_model, ~156)
        if not _HAS_MAMBA:
            raise ImportError("mamba_ssm bulunamadi. Bu kol gercek Mamba gerektirir.")
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)])
        # Her blok icin AYRI pre-LayerNorm — residual birikmesini engeller
        # (4 blok boyunca normalize edilmemis toplam -> mamba.norm'da patlama).
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_out = d_model

    def forward(self, x):              # x: (B,1,10000)
        h = self.stem(x)               # (B, d_model, T)
        h = h.transpose(1, 2)          # (B, T, d_model) — Mamba (B,L,D) bekler
        # Pre-LN residual: her blok ONCESI normalize -> birikme/patlama engellenir
        for blk, bn in zip(self.blocks, self.block_norms):
            h = h + blk(bn(h))         # pre-norm residual (transformer-standardi)
        h = self.norm(h)
        return h.mean(dim=1)           # (B, d_model) zaman havuzu


# =============================================================================
# 2) Shifted-window transformer kolu — log-mel
# =============================================================================
class WindowAttention(nn.Module):
    """Pencere-ici cok-basli oz-dikkat (Swin tarzi, sade)."""
    def __init__(self, dim, n_heads, win):
        super().__init__()
        self.dim, self.n_heads, self.win = dim, n_heads, win
        self.scale = (dim // n_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):              # x: (num_win*B, win*win, dim)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, C // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    """Bir shifted-window transformer blogu (pencere + kaydirmali pencere mantigi)."""
    def __init__(self, dim, n_heads, win=4, shift=0):
        super().__init__()
        self.win, self.shift = win, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, n_heads, win)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                 nn.Linear(dim*2, dim))

    def _partition(self, x, H, W):
        # x: (B,H,W,C) -> (nW*B, win*win, C)
        B, _, _, C = x.shape
        x = x.view(B, H//self.win, self.win, W//self.win, self.win, C)
        x = x.permute(0,1,3,2,4,5).contiguous()
        return x.view(-1, self.win*self.win, C)

    def _reverse(self, win_x, H, W, B):
        C = win_x.shape[-1]
        x = win_x.view(B, H//self.win, W//self.win, self.win, self.win, C)
        x = x.permute(0,1,3,2,4,5).contiguous()
        return x.view(B, H, W, C)

    def forward(self, x, H, W):        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1,2))
        win = self._partition(x, H, W)
        win = self.attn(win)
        x = self._reverse(win, H, W, B)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1,2))
        x = x.view(B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SpecBranch(nn.Module):
    """log-mel (B,1,64,201) -> patch embed -> Swin bloklari -> (B,D)."""
    def __init__(self, d_model=128, n_heads=4, win=4, depth=4, patch=4):
        super().__init__()
        # patch embed: 4x4 patch conv -> (B, d_model, 16, ~50)
        self.patch = nn.Conv2d(1, d_model, kernel_size=patch, stride=patch)
        self.d_model = d_model; self.win = win
        # bloklar: alternatif shift (0, win/2, 0, win/2 ...)
        self.blocks = nn.ModuleList([
            SwinBlock(d_model, n_heads, win, shift=(0 if i%2==0 else win//2))
            for i in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.d_out = d_model

    def forward(self, x):              # (B,1,64,201)
        h = self.patch(x)              # (B, d_model, H, W)
        B, C, H, W = h.shape
        # H,W pencereye tam bolunmeli: gerekirse pad
        padH = (self.win - H % self.win) % self.win
        padW = (self.win - W % self.win) % self.win
        if padH or padW:
            h = F.pad(h, (0, padW, 0, padH))
            H, W = h.shape[2], h.shape[3]
        h = h.flatten(2).transpose(1, 2)   # (B, H*W, C)
        for blk in self.blocks:
            h = blk(h, H, W)
        h = self.norm(h)
        return h.mean(dim=1)               # (B, d_model)


# =============================================================================
# 3) Head'ler
# =============================================================================
class MLPHead(nn.Module):
    def __init__(self, d_in, n_classes=N_CLASSES, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in//2), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(d_in//2, n_classes))
    def forward(self, x): return self.net(x)


class KANLayer(nn.Module):
    """Saf-PyTorch KAN katmani (B-spline tabanli, ek bagimlilik yok).
    Basitlestirilmis Efficient-KAN yaklasimi: ogrenilebilir spline + taban."""
    def __init__(self, in_f, out_f, grid=5, k=3):
        super().__init__()
        self.in_f, self.out_f, self.grid, self.k = in_f, out_f, grid, k
        # spline grid
        g = torch.linspace(-1, 1, grid + 1)
        self.register_buffer("grid_pts", g)
        self.base_w = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        self.spline_w = nn.Parameter(torch.randn(out_f, in_f, grid + k) * 0.1)
        self.base_act = nn.SiLU()

    def _bspline(self, x):
        # x: (B, in_f) -> (B, in_f, grid+k) basit RBF-benzeri taban
        x = torch.tanh(x)                       # [-1,1] araligina sik
        xc = x.unsqueeze(-1)                    # (B,in_f,1)
        centers = torch.linspace(-1, 1, self.grid + self.k,
                                 device=x.device)  # (grid+k,)
        width = 2.0 / (self.grid + self.k)
        return torch.exp(-((xc - centers) / width) ** 2)  # (B,in_f,grid+k)

    def forward(self, x):                       # (B, in_f)
        base = F.linear(self.base_act(x), self.base_w)        # (B,out_f)
        bs = self._bspline(x)                                 # (B,in_f,grid+k)
        spline = torch.einsum("big,oig->bo", bs, self.spline_w)  # (B,out_f)
        return base + spline


class KANHead(nn.Module):
    def __init__(self, d_in, n_classes=N_CLASSES):
        super().__init__()
        self.k1 = KANLayer(d_in, d_in // 2)
        self.k2 = KANLayer(d_in // 2, n_classes)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        return self.k2(self.drop(self.k1(x)))


class EvidentialHead(nn.Module):
    """Evidential Deep Learning basi (Sensoy 2018).
    Logit yerine 'evidence' (>=0) uretir; alpha=evidence+1 Dirichlet param.
    Tek forward'ta hem tahmin hem belirsizlik. OOD'de belirsizlik->1.
    forward() ham logit doner (evidence=softplus(logit) egitim/cikarimda alinir)."""
    def __init__(self, d_in, n_classes=N_CLASSES, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in // 2), nn.GELU(),
                                 nn.Dropout(p), nn.Linear(d_in // 2, n_classes))
    def forward(self, x):
        return self.net(x)   # ham logit; evidence = softplus(logit)


def edl_evidence(logits):
    """Logit -> evidence (>=0). softplus kullanilir."""
    return F.softplus(logits)


def edl_predict(logits, n_classes=None):
    """EDL: logits -> (olasiliklar, belirsizlik). Tek forward'tan.
    belirsizlik u = K/S, S=sum(alpha), alpha=evidence+1."""
    evidence = edl_evidence(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    K = logits.shape[-1]
    p = alpha / S
    u = K / S.squeeze(-1)
    return p, u


def edl_loss(logits, target, epoch=0, anneal=10, n_classes=None):
    """EDL kaybi (Bayes risk + KL annealing). target: (B,) sinif indeksi."""
    evidence = edl_evidence(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    K = logits.shape[1]
    y = F.one_hot(target, num_classes=K).float()
    # Bayes risk (kare hata beklentisi)
    p = alpha / S
    err = ((y - p) ** 2).sum(dim=1)
    var = (p * (1 - p) / (S + 1)).sum(dim=1)
    loss_data = (err + var).mean()
    # KL annealing: yanlis kanitlari cezalandir (erken egitimde kapali)
    lam = min(1.0, epoch / float(anneal))
    alpha_tilde = y + (1 - y) * alpha   # dogru sinif kaniti korunur
    kl = _dirichlet_kl(alpha_tilde, K)
    return loss_data + lam * kl.mean()


def _dirichlet_kl(alpha, K):
    """KL(Dir(alpha) || Dir(1)) — uniform Dirichlet'e."""
    S = alpha.sum(dim=1, keepdim=True)
    t1 = torch.lgamma(S.squeeze(1)) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
    t2 = (torch.lgamma(alpha)).sum(dim=1)   # - 0 (lgamma(1)=0)
    t3 = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return t1 - t2 + t3


HEADS = {"mlp": MLPHead, "kan": KANHead, "edl": EvidentialHead}


# =============================================================================
# 4) Tam model
# =============================================================================
class HybridPCG(nn.Module):
    def __init__(self, head="mlp", d_model=128, n_classes=N_CLASSES):
        super().__init__()
        self.mamba = MambaBranch(d_model=d_model)
        self.spec  = SpecBranch(d_model=d_model)
        d_fused = self.mamba.d_out + self.spec.d_out
        self.fuse = nn.Sequential(nn.Linear(d_fused, d_model), nn.GELU(),
                                  nn.Dropout(0.2))
        assert head in HEADS, f"Bilinmeyen head: {head}"
        self.head = HEADS[head](d_model, n_classes)
        self.head_name = head

    def forward(self, raw, mel):       # raw:(B,1,10000) mel:(B,1,64,201)
        f1 = self.mamba(raw)
        f2 = self.spec(mel)
        z = self.fuse(torch.cat([f1, f2], dim=1))
        return self.head(z)


# =============================================================================
# YENI: BiLSTM kolu — domain shift'e dayanikli raw kodlayici
# =============================================================================
class BiLSTMBranch(nn.Module):
    """Ham PCG (B,1,10000) -> conv on-yuz -> BiLSTM -> (B,D).
    Mamba kolunun yerine: dis-validasyonda BiLSTM raw'da daha dayanikli cikti.
    Cikti boyutu d_model (MambaBranch ile ayni — fuzyon uyumu icin)."""
    def __init__(self, d_model=128, lstm_layers=2):
        super().__init__()
        def cb(ci, co, k=7, s=4, p=3, groups=8):
            return nn.Sequential(nn.Conv1d(ci, co, k, s, p),
                                 nn.GroupNorm(min(groups, co), co), nn.GELU())
        self.stem = nn.Sequential(cb(1, 32), cb(32, 64), cb(64, 128))  # (B,128,~156)
        # BiLSTM hidden = d_model//2 -> bidir concat = d_model (boyut korunur)
        self.lstm = nn.LSTM(input_size=128, hidden_size=d_model // 2,
                            num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.norm = nn.LayerNorm(d_model)
        self.d_out = d_model

    def forward(self, x):              # (B,1,10000)
        h = self.stem(x)               # (B,128,T)
        h = h.transpose(1, 2)          # (B,T,128)
        out, _ = self.lstm(h)          # (B,T,d_model)
        return self.norm(out.mean(dim=1))   # (B,d_model)


class HybridPCG2(nn.Module):
    """YENI ozgun mimari: BiLSTM(raw zaman) + Swin(mel spektral) fuzyon + head.
    Gerekce: Mamba(raw) dis-validasyonda cokuyordu; BiLSTM(raw) dayanikli.
    Iki tamamlayici kodlayici (recurrent + attention), iki modalite (zaman+spektral)."""
    def __init__(self, head="kan", d_model=128, n_classes=N_CLASSES):
        super().__init__()
        self.bilstm = BiLSTMBranch(d_model=d_model)
        self.spec = SpecBranch(d_model=d_model)
        d_fused = self.bilstm.d_out + self.spec.d_out
        self.fuse = nn.Sequential(nn.Linear(d_fused, d_model), nn.GELU(),
                                  nn.Dropout(0.2))
        assert head in HEADS, f"Bilinmeyen head: {head}"
        self.head = HEADS[head](d_model, n_classes)
        self.head_name = head

    def forward(self, raw, mel):
        f1 = self.bilstm(raw)
        f2 = self.spec(mel)
        z = self.fuse(torch.cat([f1, f2], dim=1))
        return self.head(z)


# =============================================================================
# YENI (ozgun): HybridPCG3 — UC kol + attention-gated fuzyon
#   BiLSTM(raw) + Mamba(raw) + Swin(mel), kollar attention ile agirliklanir.
#   Gerekce: domain shift'te cokmus kol dusuk agirlik alir -> fuzyonu bozmaz.
#   Ozgunluk: 3 tamamlayici kodlayici (recurrent+state-space+windowed-attn)
#             + ogrenilen kol gating + KAN belirsizlik basi.
# =============================================================================
class AttentionGatedFusion(nn.Module):
    """K kolu (her biri (B,d)) ogrenilen softmax-agirliklarla birlestirir.
    Cokmus/yararsiz kol dusuk agirlik alir."""
    def __init__(self, d_model, n_branches):
        super().__init__()
        # her kol icin skor ureten kucuk gating agi
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))
        self.n_branches = n_branches

    def forward(self, branch_feats):   # liste: K x (B, d_model)
        stacked = torch.stack(branch_feats, dim=0)        # (K, B, d)
        K, B, d = stacked.shape
        scores = self.gate(stacked).squeeze(-1)           # (K, B)
        attn = torch.softmax(scores, dim=0)               # (K, B) kollar arasi
        fused = (attn.unsqueeze(-1) * stacked).sum(0)     # (B, d)
        return fused, attn   # attn'i da don (yorumlanabilirlik/analiz icin)


class HybridPCG3(nn.Module):
    """Ozgun uc-kol mimari: BiLSTM(raw)+Mamba(raw)+Swin(mel), attention fuzyon."""
    def __init__(self, head="kan", d_model=128, n_classes=N_CLASSES):
        super().__init__()
        self.bilstm = BiLSTMBranch(d_model=d_model)
        self.mamba = MambaBranch(d_model=d_model)
        self.spec = SpecBranch(d_model=d_model)
        self.fusion = AttentionGatedFusion(d_model, n_branches=3)
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.2))
        assert head in HEADS, f"Bilinmeyen head: {head}"
        self.head = HEADS[head](d_model, n_classes)
        self.head_name = head
        self.last_attn = None   # son batch'in kol agirliklari (analiz icin)

    def forward(self, raw, mel):
        f_lstm = self.bilstm(raw)
        f_mamba = self.mamba(raw)
        f_spec = self.spec(mel)
        fused, attn = self.fusion([f_lstm, f_mamba, f_spec])
        self.last_attn = attn.detach()
        return self.head(self.proj(fused))


# =============================================================================
# YENI (ozgun): Cycle-Aware Transformer — kalp dongusu faz-bilincli attention
#   Raw sinyalden otokorelasyonla kalp dongusu periyodu tahmin edilir;
#   patch'lere FAZ-bilincli konumsal kodlama eklenir; boylece self-attention
#   ayni faza denk gelen (farkli dongulerdeki) patch'leri iliskilendirebilir
#   = CROSS-CYCLE attention. PCG'ye ozgu, fizyolojik motivasyonlu.
# =============================================================================
def estimate_cycle_period(raw_bt, fs=2000, min_bpm=40, max_bpm=200):
    """raw_bt: (B, L) tensor. Her ornek icin FFT-otokorelasyon ile dongu
    periyodunu (ornek cinsinden) tahmin et. Doner: (B,) periyot."""
    B, L = raw_bt.shape
    x = raw_bt - raw_bt.mean(dim=1, keepdim=True)
    # FFT otokorelasyon
    nfft = 2 * L
    f = torch.fft.rfft(x, n=nfft, dim=1)
    ac = torch.fft.irfft(f * torch.conj(f), dim=1)[:, :L]   # (B, L)
    min_lag = int(60.0 / max_bpm * fs)
    max_lag = min(int(60.0 / min_bpm * fs), L - 1)
    # gecerli lag araliginda tepe
    seg = ac[:, min_lag:max_lag]                            # (B, range)
    peak = seg.argmax(dim=1) + min_lag                      # (B,)
    return peak.clamp(min=min_lag).float()


def phase_positional_encoding(phases, d_model, n_harmonics=8):
    """phases: (B, T) her patch'in dongu fazi [0,1). Doner: (B, T, d_model)
    faz-bilincli sinuzoidal kodlama. n_harmonics SINIRLI tutulur (varsayilan 8)
    cunku yuksek harmonikler (d/2=64) asiri yuksek frekans -> gurultu; ayni faza
    yakin patchler cok farkli kodlanir, cross-cycle amaci bozulur. Dusuk frekansli
    kodlama 'ayni faz -> benzer kodlama' ozelligini korur."""
    B, T = phases.shape
    device = phases.device
    pe = torch.zeros(B, T, d_model, device=device)
    n_harm = min(n_harmonics, d_model // 2)
    harmonics = torch.arange(1, n_harm + 1, device=device).float()  # (n_harm,)
    ang = 2 * math.pi * phases.unsqueeze(-1) * harmonics            # (B,T,n_harm)
    # ilk 2*n_harm kanala sin/cos yaz, gerisi 0 (sonra ogrenilebilir alpha ile olcekle)
    pe[:, :, 0:2 * n_harm:2] = torch.sin(ang)
    pe[:, :, 1:2 * n_harm:2] = torch.cos(ang)
    return pe


class CycleAwareBranch(nn.Module):
    """Raw (B,1,10000) -> conv stem -> patch dizisi + FAZ kodlamasi ->
    transformer enkoder (cross-cycle) -> (B, d_model)."""
    def __init__(self, d_model=128, n_heads=4, depth=3, fs=2000):
        super().__init__()
        self.fs = fs
        def cb(ci, co, k=7, s=4, p=3, groups=8):
            return nn.Sequential(nn.Conv1d(ci, co, k, s, p),
                                 nn.GroupNorm(min(groups, co), co), nn.GELU())
        self.stem = nn.Sequential(cb(1, 32), cb(32, 64), cb(64, d_model))  # (B,d,T)
        self.d_model = d_model
        self.stem_stride = 4 * 4 * 4   # toplam downsample (T = L/64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.d_out = d_model
        # ogrenilebilir faz-kodlama olcegi: 0'dan baslar (once duz transformer,
        # egitimde faz bilgisini ihtiyac kadar devreye alir). pe'nin conv ozelligini
        # ezmesini onler.
        self.phase_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):              # x: (B,1,10000)
        B, _, L = x.shape
        h = self.stem(x)               # (B, d, T)
        h = h.transpose(1, 2)          # (B, T, d)
        T = h.shape[1]
        # her ornek icin dongu periyodu (raw uzerinden)
        with torch.no_grad():
            period = estimate_cycle_period(x.squeeze(1), fs=self.fs)   # (B,)
        # her patch'in zaman-merkezi (orijinal ornek cinsinden)
        patch_centers = (torch.arange(T, device=x.device).float() + 0.5) * self.stem_stride
        # faz = (zaman mod periyot) / periyot
        phase = (patch_centers.unsqueeze(0) % period.unsqueeze(1)) / period.unsqueeze(1)  # (B,T)
        pe = phase_positional_encoding(phase, self.d_model)            # (B,T,d)
        h = h + self.phase_alpha * pe   # ogrenilebilir olcek (0'dan baslar)
        h = self.encoder(h)
        h = self.norm(h)
        return h.mean(dim=1)


class HybridPCG4(nn.Module):
    """Ozgun: Cycle-Aware Transformer(raw) + Swin(mel) fuzyon + head.
    Raw kolu kalp dongusu faz-bilincli (cross-cycle attention)."""
    def __init__(self, head="kan", d_model=128, n_classes=N_CLASSES):
        super().__init__()
        self.cycle = CycleAwareBranch(d_model=d_model)
        self.spec = SpecBranch(d_model=d_model)
        d_fused = self.cycle.d_out + self.spec.d_out
        self.fuse = nn.Sequential(nn.Linear(d_fused, d_model), nn.GELU(),
                                  nn.Dropout(0.2))
        assert head in HEADS, f"Bilinmeyen head: {head}"
        self.head = HEADS[head](d_model, n_classes)
        self.head_name = head

    def forward(self, raw, mel):
        f1 = self.cycle(raw)
        f2 = self.spec(mel)
        z = self.fuse(torch.cat([f1, f2], dim=1))
        return self.head(z)


# =============================================================================
# Sekil testi (govde Mamba gerektirir; KAN/Spec ayri test edilebilir)
# =============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=list(HEADS), default="mlp")
    ap.add_argument("--cpu_test_no_mamba", action="store_true",
                    help="Mamba'siz: sadece Spec+KAN sekil testi (CPU)")
    args = ap.parse_args()

    if args.cpu_test_no_mamba:
        print("=== Mamba'siz parca testleri (CPU) ===")
        # SpecBranch
        spec = SpecBranch(d_model=64)
        mel = torch.randn(2, 1, 64, 201)
        fs = spec(mel)
        print(f"SpecBranch: {tuple(mel.shape)} -> {tuple(fs.shape)} (beklenen (2,64))")
        assert fs.shape == (2, 64)
        # KAN head
        kan = KANHead(64)
        out = kan(fs)
        print(f"KANHead: (2,64) -> {tuple(out.shape)} (beklenen (2,3))")
        assert out.shape == (2, 3)
        # MLP head
        mlp = MLPHead(64)
        print(f"MLPHead: (2,64) -> {tuple(mlp(fs).shape)} (beklenen (2,3))")
        print("\n[OK] Spec + KAN + MLP sekilleri dogru (Mamba'siz parcalar)")
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"=== Tam HybridPCG testi (head={args.head}, device={dev}) ===")
        model = HybridPCG(head=args.head).to(dev)
        n = sum(p.numel() for p in model.parameters())
        print(f"Parametre: {n:,}")
        raw = torch.randn(2, 1, 10000).to(dev)
        mel = torch.randn(2, 1, 64, 201).to(dev)
        out = model(raw, mel)
        print(f"Forward: raw{tuple(raw.shape)} + mel{tuple(mel.shape)} "
              f"-> {tuple(out.shape)} (beklenen (2,3))")
        assert out.shape == (2, 3)
        loss = F.cross_entropy(out, torch.tensor([0, 2]).to(dev))
        loss.backward()
        print(f"Backward OK, loss={loss.item():.4f}")
        print("\n[OK] Tam hibrit model forward+backward calisiyor")
