"""Attention mechanisms for Causal-to-Spatial Perceiver Bridge model."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricRoPECrossAttention(nn.Module):
    """Multi-head cross-attention with decoupled Rotary Positional Embeddings (RoPE)."""

    def __init__(self, dim: int = 1024, heads: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head**-0.5

        # Projections (bias=False for better performance)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier Uniform for stable early gradients."""
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)
        nn.init.xavier_uniform_(self.to_out.weight)

    def _apply_rotary_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies RoPE to a tensor."""
        # position: [seq_len, 1]
        postion = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)

        # div_term: [dim_head / 2]
        div_term = torch.exp(
            torch.arange(0, self.dim_head, 2, dtype=torch.float32, device=x.device)
            * (-math.log(10000.0) / self.dim_head)
        )

        # freqs: [seq_len, dim_head / 2]
        freqs = postion * div_term

        # emb: [1, 1, seq_len, dim_head]
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0).unsqueeze(0)

        # Cast RoPE embeddings to match input dtype
        emb = emb.to(dtype=x.dtype)

        # Split features for rotation
        x1, x2 = x[..., : self.dim_head // 2], x[..., self.dim_head // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)

        return (x * emb.cos()) + (x_rotated * emb.sin())

    def forward(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the cross-attention mechanism."""
        b, n_q, _ = q_x.shape
        _, n_kv, _ = kv_x.shape

        # 1. Linear projections and reshaping
        # q: [B, heads, N_q, dim_head]
        q = self.to_q(q_x).view(b, n_q, self.heads, self.dim_head).transpose(1, 2)

        # kv: [B, N_kv, D * 2] -> split -> 2 * [B, heads, N_kv, dim_head]
        kv = self.to_kv(kv_x).chunk(2, dim=-1)
        k, v = [t.view(b, n_kv, self.heads, self.dim_head).transpose(1, 2) for t in kv]

        # 2. Apply Decoupled RoPE
        q = self._apply_rotary_emb(q, n_q)
        k = self._apply_rotary_emb(k, n_kv)

        # 3. Scaled dot-product attention
        # dots: [B, heads, N_q, N_kv]
        dots = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # mask: [B, N_kv] -> [B, 1, 1, N_kv]
            mask = mask.view(b, 1, 1, n_kv)
            # Fill masked positions with negative infinity
            dots = dots.masked_fill(~mask, torch.finfo(dots.dtype).min)

        # attn: [B, heads, N_q, N_kv]
        attn = dots.softmax(dim=-1)

        # 4. Value aggregation and output projection
        # out: [B, heads, N_q, dim_head] -> [B, N-q, heads * dim_head]
        out = (attn @ v).transpose(1, 2).reshape(b, n_q, -1)

        # return: [B, N_q, D]
        return self.to_out(out)
