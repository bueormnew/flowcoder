from __future__ import annotations
import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CoreConfig


class SinusoidalTimeEmbedding(nn.Module):
    """Embedding sinusoidal continuo para el paso temporal t in [0, 1]."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: shape (B,) o (B, 1) con valores continuos [0, 1]
        t = t.view(-1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None, :] * 1000.0
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Aplica escalado adaptativo y desplazamiento (AdaLN)."""
    return x * (1.0 + scale) + shift


class ConvNeXtBlock(nn.Module):
    """
    Bloque convolucional depthwise 7x7 ultra-optimizado para aceleradores y móviles.
    Combina gran campo receptivo (7x7) con complejidad lineal O(HW).
    """
    def __init__(self, dim: int, cdim: int, mlp_ratio: float = 4.0, kernel_size: int = 7):
        super().__init__()
        self.dim = dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.GroupNorm(1, dim, affine=False)
        hidden_dim = int(dim * mlp_ratio)
        self.pw1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(hidden_dim, dim)
        self.mod = nn.Linear(cdim, 3 * dim)
        # Inicialización en cero para compuerta residual
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: vector de acondicionamiento (B, cdim)
        s, b, gate = self.mod(F.silu(c))[:, :, None, None].chunk(3, dim=1)
        h = self.dwconv(x)
        h = modulate(self.norm(h), b, s)
        h = h.permute(0, 2, 3, 1) # B, H, W, C
        h = self.pw2(self.act(self.pw1(h)))
        h = h.permute(0, 3, 1, 2) # B, C, H, W
        return x + gate * h


class CrossAttentionBlock(nn.Module):
    """Cross-Attention modulada sobre tokens de texto."""
    def __init__(self, dim: int, txt_dim: int, cdim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(txt_dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mod = nn.Linear(cdim, 3 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, txt_ctx: torch.Tensor, mask: Optional[torch.Tensor], c: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # B, N, C
        s, b, gate = self.mod(F.silu(c))[:, None, :].chunk(3, dim=-1)

        norm_t = modulate(self.norm(t), b, s)
        q = self.to_q(norm_t).view(B, H * W, self.heads, C // self.heads).transpose(1, 2)
        k, v = self.to_kv(txt_ctx).chunk(2, dim=-1)
        k = k.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, -1, self.heads, C // self.heads).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            # mask: (B, L) bool
            attn_mask = mask[:, None, None, :] # (B, 1, 1, L)
            if attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
                attn_mask = attn_mask.bool()

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        t = t + gate * out
        return t.transpose(1, 2).reshape(B, C, H, W)


class SelfAttentionBlock(nn.Module):
    """Auto-atención espacial para coherencia geométrica global."""
    def __init__(self, dim: int, cdim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mod = nn.Linear(cdim, 3 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        s, b, gate = self.mod(F.silu(c))[:, None, :].chunk(3, dim=-1)

        norm_t = modulate(self.norm(t), b, s)
        q, k, v = self.qkv(norm_t).chunk(3, dim=-1)
        q = q.view(B, H * W, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, H * W, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, H * W, self.heads, C // self.heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        t = t + gate * out
        return t.transpose(1, 2).reshape(B, C, H, W)


class HybridCoreBlock(nn.Module):
    """Bloque híbrido configurable: ConvNeXt + Cross-Attn opcional + Self-Attn opcional."""
    def __init__(
        self,
        dim: int,
        txt_dim: int,
        cdim: int,
        heads: int,
        has_cross_attn: bool = True,
        has_self_attn: bool = False,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.conv = ConvNeXtBlock(dim, cdim, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
        self.cross_attn = CrossAttentionBlock(dim, txt_dim, cdim, heads) if has_cross_attn else None
        self.self_attn = SelfAttentionBlock(dim, cdim, heads) if has_self_attn else None

    def forward(
        self,
        x: torch.Tensor,
        txt_ctx: torch.Tensor,
        mask: Optional[torch.Tensor],
        c: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv(x, c)
        if self.cross_attn is not None:
            x = self.cross_attn(x, txt_ctx, mask, c)
        if self.self_attn is not None:
            x = self.self_attn(x, c)
        return x


class MobileCoreDenoiser(nn.Module):
    """
    CORE híbrido Rectified Flow para inferencia ultra-rápida (1-8 pasos).
    Integra Patchify (2x2), Convolución Depthwise ConvNeXt y Atención Eficiente.
    Predice el vector de velocidad v_t = dx_t / dt.
    """
    def __init__(self, config: Optional[CoreConfig] = None):
        super().__init__()
        self.config = config or CoreConfig()
        cfg = self.config

        # 1. Stem de Patchificación (reduce la resolución latente 2x2 para escala eficiente)
        self.patch_size = cfg.patch_size
        self.patch_dim = cfg.in_channels * (self.patch_size ** 2)
        self.stem = nn.Conv2d(cfg.in_channels, cfg.dim, kernel_size=self.patch_size, stride=self.patch_size)

        # 2. Acondicionamiento (Tiempo continuo t + Escala de Guía s + Pooled Text)
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        if cfg.use_guidance_embed:
            self.guidance_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
            self.guidance_mlp = nn.Sequential(
                nn.Linear(cfg.time_embed_dim, cfg.dim),
                nn.SiLU(),
                nn.Linear(cfg.dim, cfg.dim),
            )
        else:
            self.guidance_embed = None

        self.pool_text = nn.Linear(cfg.txt_dim, cfg.dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )

        # Token nulo aprendido para Classifier-Free Guidance auténtico
        self.null_text_token = nn.Parameter(torch.randn(1, 1, cfg.txt_dim) * 0.02)

        # 3. Bloques híbridos del CORE
        self.blocks = nn.ModuleList()
        for i in range(cfg.depth):
            has_cross = (i % cfg.cross_attn_interval == 0)
            has_self = (i in cfg.self_attn_layers)
            self.blocks.append(
                HybridCoreBlock(
                    dim=cfg.dim,
                    txt_dim=cfg.txt_dim,
                    cdim=cfg.dim,
                    heads=cfg.heads,
                    has_cross_attn=has_cross,
                    has_self_attn=has_self,
                    mlp_ratio=cfg.mlp_ratio,
                    kernel_size=cfg.dw_kernel_size,
                )
            )

        # 4. Head de Des-patchificación (PixelShuffle para recuperar latentes)
        self.out_norm = nn.GroupNorm(1, cfg.dim, affine=False)
        self.out_linear = nn.Linear(cfg.dim, cfg.dim)
        self.unpatchify = nn.Sequential(
            nn.Conv2d(cfg.dim, cfg.out_channels * (self.patch_size ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(self.patch_size),
        )
        # Inicialización a cero del head para estabilidad en transfer learning
        nn.init.zeros_(self.unpatchify[0].weight)
        nn.init.zeros_(self.unpatchify[0].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        txt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: Latente con ruido (B, in_channels, H, W)
        t: Paso de tiempo continuo [0, 1] (B,) o float
        txt: Embeddings de texto (B, L, txt_dim)
        mask: Máscara booleana de atención de texto (B, L)
        guidance: Escala de guía CFG para 1-pass guidance distillation (B,) o float
        """
        B = x.shape[0]
        if not isinstance(t, torch.Tensor):
            t = torch.full((B,), float(t), device=x.device, dtype=torch.float32)
        elif t.dim() == 0:
            t = t.repeat(B)

        # 1. Proyección de tiempo
        c = self.time_mlp(self.time_embed(t).to(dtype=x.dtype))

        # 2. Proyección de escala de guía (si está habilitado para 1-pass CFG)
        if self.guidance_embed is not None and guidance is not None:
            if not isinstance(guidance, torch.Tensor):
                guidance = torch.full((B,), float(guidance), device=x.device, dtype=torch.float32)
            elif guidance.dim() == 0:
                guidance = guidance.repeat(B)
            c = c + self.guidance_mlp(self.guidance_embed(guidance).to(dtype=x.dtype))

        # 3. Pooling de texto y modulación global c
        if txt.dtype != x.dtype:
            txt = txt.to(dtype=x.dtype)
        if mask is not None:
            m = mask.unsqueeze(-1).to(txt.dtype)
            pooled = (txt * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            pooled = txt.mean(dim=1)
        c = self.cond_mlp(c + self.pool_text(pooled))

        # 4. Patchify Stem
        h = self.stem(x)

        # 5. Pasada a través de bloques híbridos
        for block in self.blocks:
            h = block(h, txt, mask, c)

        # 6. Unpatchify Head
        h_norm = self.out_norm(h)
        return self.unpatchify(F.silu(h_norm))
